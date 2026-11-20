"""Compare the three approved 5.2 query prompts on identical document vectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from sqlalchemy import text

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    VectorIndexManifest,
    VectorRecord,
    VectorSearchQuery,
)
from ai_learning_rag.embedding import EmbeddingConfig, Qwen3Embedder
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview

EXPERIMENT_DATABASE = "ai_learning_rag_stage52"
PDF_EVIDENCE_PROMPT = (
    "Instruct: Given a question, retrieve passages from the PDF collection "
    "that provide direct evidence to answer it.\nQuery:"
)
QUERY_PROFILES = (
    ("default", None),
    ("pdf-evidence", PDF_EVIDENCE_PROMPT),
    ("no-instruction", ""),
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_runtime_versions(audit_metadata: dict[str, str]) -> None:
    for package, audit_field in (
        ("torch", "torch"),
        ("sentence-transformers", "sentence_transformers"),
    ):
        expected = audit_metadata.get(audit_field)
        if expected is None:
            raise ValueError(f"validated baseline is missing runtime metadata: {audit_field}")
        installed = version(package)
        if installed != expected:
            raise ValueError(
                f"runtime {package} differs from the validated baseline: {installed} != {expected}"
            )


def load_baseline_embedder(baseline: VectorIndexManifest) -> Qwen3Embedder:
    check_runtime_versions(baseline.audit_metadata)
    return Qwen3Embedder(
        EmbeddingConfig(
            model_id=baseline.model_id,
            model_revision=baseline.model_revision,
            tokenizer_id=baseline.tokenizer_id,
            tokenizer_revision=baseline.tokenizer_revision,
            device=baseline.audit_metadata["device"],
            dimension=baseline.embedding_dimension,
            max_sequence_length=baseline.max_input_tokens,
            normalize_embeddings=baseline.normalized,
            pooling=baseline.pooling,
            dtype=baseline.dtype,
            query_profile=baseline.query_profile,
            adapter_version=baseline.adapter_version,
            local_files_only=True,
        )
    )


def records_by_chunk(records: tuple[VectorRecord, ...]) -> dict[ChunkIdentity, VectorRecord]:
    result = {
        ChunkIdentity(record.document_id, record.chunk_fingerprint): record for record in records
    }
    if len(result) != len(records):
        raise ValueError("an active index contains duplicate chunk identities")
    return result


def copy_index(
    index: PostgresVectorIndex,
    baseline: VectorIndexManifest,
    source_records: tuple[VectorRecord, ...],
    name: str,
) -> tuple[VectorIndexManifest, int]:
    identity = baseline.identity_payload | {
        "logical_name": f"stage52-embedding-{name}-v1",
        "query_profile": f"qwen-{name}-v1",
    }
    target = VectorIndexManifest.create(
        **identity,
        required_pgvector_version=baseline.required_pgvector_version,
        audit_metadata=baseline.audit_metadata,
    )
    current = index.get_manifest(target.index_id)
    if current is None:
        index.ensure_index(target)
        current = target
    if current.status not in {IndexStatus.BUILDING, IndexStatus.READY, IndexStatus.ACTIVE}:
        raise RuntimeError(f"candidate has an unusable index status: {current.status}")

    by_document = defaultdict(list)
    for record in source_records:
        by_document[record.document_id].append(record)

    written = 0
    if current.status is not IndexStatus.ACTIVE:
        for document_id, records in by_document.items():
            source = index.get_active_generation(baseline.index_id, document_id)
            if source is None or any(
                record.generation_id != source.generation_id for record in records
            ):
                raise RuntimeError(f"baseline document changed during snapshot read: {document_id}")
            generation_inputs = asdict(source)
            del generation_inputs["generation_id"]
            generation_inputs["index_id"] = target.index_id
            generation = DocumentGeneration.create(**generation_inputs)
            target_records = tuple(
                replace(
                    record,
                    index_id=target.index_id,
                    generation_id=generation.generation_id,
                    record_id="rec_"
                    + sha256(
                        f"{target.index_id}\0{generation.generation_id}\0{record.chunk_fingerprint}".encode()
                    ).hexdigest(),
                )
                for record in records
            )
            state = index.get_document_state(target.index_id, document_id)
            outcome = index.replace_document(
                generation, target_records, expected_revision=state.revision
            )
            if outcome.status is DocumentReplaceStatus.CONCURRENT_UPDATE:
                raise RuntimeError(f"candidate document changed during copy: {document_id}")
            if outcome.status is not DocumentReplaceStatus.ALREADY_INDEXED:
                written += outcome.record_count
        if current.status is IndexStatus.BUILDING:
            index.ensure_index(replace(target, status=IndexStatus.READY))
        index.ensure_index(replace(target, status=IndexStatus.ACTIVE))

    return replace(target, status=IndexStatus.ACTIVE), written


def rank_group(
    index: PostgresVectorIndex,
    manifest: VectorIndexManifest,
    dataset: RetrievalEvaluationDataset,
    embedder: Qwen3Embedder,
    prompt: str | None,
) -> tuple[dict, dict]:
    prefix = embedder.default_query_prompt() if prompt is None else prompt
    full_inputs = {case.query_id: prefix + case.query for case in dataset.cases}
    for query_id, model_input in full_inputs.items():
        if embedder.count_tokens(model_input) > manifest.max_input_tokens:
            raise ValueError(f"complete query exceeds token limit: {query_id}")

    timings = {case.query_id: [] for case in dataset.cases}
    ranking = {}
    for round_number in range(4):
        for case in dataset.cases:
            started = perf_counter()
            results = index.search(
                VectorSearchQuery(
                    index_id=manifest.index_id,
                    vector=tuple(embedder.embed_query(case.query, prompt=prompt)),
                    top_k=dataset.max_k,
                )
            )
            elapsed = (perf_counter() - started) * 1000
            if len(results) != dataset.max_k:
                raise RuntimeError(
                    f"wrong number of results for {case.query_id} in {manifest.index_id}"
                )
            if round_number:
                timings[case.query_id].append(elapsed)
            if round_number == 1:
                ranking[case.query_id] = results

    samples = [sample for query_values in timings.values() for sample in query_values]
    return ranking, {
        "query_prefix": prefix,
        "warmup_rounds": 1,
        "timed_rounds": 3,
        "retrieval_ms": {"median": median(samples), "min": min(samples), "max": max(samples)},
        "queries": {
            key: {
                "model_input": full_inputs[key],
                "samples_ms": values,
                "median_ms": median(values),
            }
            for key, values in timings.items()
        },
    }


def compare(
    baseline: RetrievalEvaluationReport, candidate: RetrievalEvaluationReport
) -> list[dict]:
    changes = []
    for old, new in zip(baseline.queries, candidate.queries, strict=True):
        if old.case != new.case:
            raise ValueError("question or approved labels changed between groups")
        old_hits = {hit.identity: hit.rank for hit in old.hits if hit.relevant}
        new_hits = {hit.identity: hit.rank for hit in new.hits if hit.relevant}
        changes.append(
            {
                "query_id": old.case.query_id,
                "formal": old.reciprocal_rank is not None,
                "first_relevant_rank": [old.first_relevant_rank, new.first_relevant_rank],
                "recall_at_k": {
                    k: [value, dict(new.recall_at_k)[k]] for k, value in old.recall_at_k
                },
                "evidence_ranks": [
                    dict(asdict(key), before=old_hits.get(key), after=new_hits.get(key))
                    for key in dict.fromkeys((*old_hits, *new_hits))
                ],
                "ranking_changed": [hit.identity for hit in old.hits]
                != [hit.identity for hit in new.hits],
            }
        )
    return changes


def check_control(actual: dict, reference: dict) -> None:
    for key in ("dataset", "corpus", "index", "metrics"):
        if actual[key] != reference[key]:
            raise RuntimeError(f"the default control changed the validated {key}")
    for original, repeated in zip(reference["queries"], actual["queries"], strict=True):
        query_id = original["query_id"]
        if any(original[key] != repeated[key] for key in ("query_id", "query", "case_type")):
            raise RuntimeError(f"the default control changed question {query_id}")
        if original["case_type"] != "answerable":
            continue
        if original["metrics"] != repeated["metrics"]:
            raise RuntimeError(f"the default control changed the metrics for {query_id}")
        for cutoff in (1, 3, 5):
            keys = (
                {
                    (hit["document_id"], hit["chunk_fingerprint"])
                    for hit in row["hits"][:cutoff]
                    if hit["relevant"]
                }
                for row in (original, repeated)
            )
            old, new = keys
            if old != new:
                raise RuntimeError(
                    f"the default control changed evidence for {query_id} at {cutoff}"
                )


def write_comparison(output: Path, groups: dict, differences: dict) -> None:
    write_json(output / "comparison.json", {"groups": groups, "differences": differences})
    lines = [
        "# 5.2 · Embedding 查询指令对照",
        "",
        "| 组别 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 片段数 | 向量字节数 | "
        "本轮写入 | 复制与启用 (s) | 检索中位数 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in groups.items():
        recall, mrr = group["metrics"]["recall_at_k"], group["metrics"]["mrr"]
        copy_seconds = group["copy_and_activation_seconds"]
        copy_label = "沿用索引" if copy_seconds is None else f"{copy_seconds:.3f}"
        lines.append(
            f"| {name} | {recall['1']:.4f} | {recall['3']:.4f} | {recall['5']:.4f} | {mrr:.4f} | "
            f"{group['chunk_count']} | {group['vector_bytes']} | {group['records_written']} | "
            f"{copy_label} | {group['retrieval_ms']['median']:.2f} |"
        )
    for name, changes in differences.items():
        lines.extend(("", f"## {name} 与默认组的差异", ""))
        lines.extend(
            f"- `{row['query_id']}` 排名有变化。首个相关结果 {row['first_relevant_rank']}，"
            f"前五条相关片段 {sum(hit['before'] is not None for hit in row['evidence_ranks'])} → "
            f"{sum(hit['after'] is not None for hit in row['evidence_ranks'])}。"
            for row in changes
            if row["ranking_changed"]
        )
        lines.extend(("", f"详见 [{name} 的逐题报告]({name}/retrieval-evaluation.md)。"))
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline-preview", "baseline-report", "dataset", "output-dir"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()

    dataset = load_retrieval_evaluation_dataset(args.dataset)
    preview = json.loads(args.baseline_preview.read_text(encoding="utf-8"))
    inventory, baseline, _ = evaluation_inputs_from_preview(preview)
    reference = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if dataset.dataset_version != "1.0.0" or inventory.snapshot != dataset.corpus:
        raise ValueError("the three groups require the validated v1.0.0 corpus and labels")

    settings = DatabaseSettings.from_env()
    if settings.url.database != EXPERIMENT_DATABASE:
        raise ValueError(f"use the isolated experiment database: {EXPERIMENT_DATABASE}")
    engine = settings.create_engine()
    try:
        with engine.connect() as connection:
            if connection.scalar(text("SELECT current_database()")) != EXPERIMENT_DATABASE:
                raise ValueError("database connection is outside the experiment database")
        index = PostgresVectorIndex(engine)
        if (
            index.get_manifest(baseline.index_id) != baseline
            or baseline.status is not IndexStatus.ACTIVE
        ):
            raise ValueError("stored default index does not match the validated baseline")
        source_records, source_bytes = index.read_active_records(baseline.index_id)
        source = records_by_chunk(source_records)
        if source.keys() != set(inventory.chunks):
            raise ValueError("the active baseline has different document chunks from the dataset")

        embedder = load_baseline_embedder(baseline)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        groups = {}
        differences = {}
        baseline_report = None
        for name, prompt in QUERY_PROFILES:
            output = args.output_dir / name
            output.mkdir()
            target, written, write_seconds = baseline, 0, None
            if prompt is not None:
                started = perf_counter()
                target, written = copy_index(index, baseline, source_records, name)
                write_seconds = perf_counter() - started
            current_records, vector_bytes = index.read_active_records(target.index_id)
            current = records_by_chunk(current_records)
            if current.keys() != source.keys() or vector_bytes != source_bytes:
                raise ValueError(f"the {name} index has different chunks or storage size")
            if any(
                current[key].embedding_input_hash != record.embedding_input_hash
                or current[key].vector != record.vector
                for key, record in source.items()
            ):
                raise ValueError(f"the {name} index has different document vectors")

            rankings, timings = rank_group(index, target, dataset, embedder, prompt)
            report = evaluate_retrieval(
                dataset=dataset,
                inventory=inventory,
                results_by_query=rankings,
                index_manifest=target,
            )
            report_payload = json.loads(report.to_json())
            if baseline_report is None:
                check_control(report_payload, reference)
                baseline_report = report
            else:
                differences[name] = compare(baseline_report, report)
            groups[name] = {
                "index_id": target.index_id,
                "index_identity": target.identity_payload,
                "chunk_set_fingerprint": inventory.snapshot.chunk_set_fingerprint,
                "chunk_count": len(current),
                "vector_bytes": vector_bytes,
                "records_written": written,
                "copy_and_activation_seconds": write_seconds,
                "metrics": report_payload["metrics"],
                **timings,
            }
            write_json(output / "measurements.json", groups[name])
            (output / "retrieval-evaluation.json").write_text(report.to_json(), encoding="utf-8")
            (output / "retrieval-evaluation.md").write_text(report.to_markdown(), encoding="utf-8")
            print(f"group={name} index={target.index_id} chunks={len(current)}", flush=True)
        write_comparison(args.output_dir, groups, differences)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

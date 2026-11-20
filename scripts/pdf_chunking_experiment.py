"""Check or compare the approved 600- and 1,000-character PDF configurations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from pdf_embedding_experiment import (
    EXPERIMENT_DATABASE,
    load_baseline_embedder,
    rank_group,
    records_by_chunk,
    write_json,
)
from pdf_indexing_preview import load_preview_document, preview_generation
from sqlalchemy import text

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    InMemoryVectorIndex,
    VectorIndexManifest,
)
from ai_learning_rag.embedding import Qwen3Embedder
from ai_learning_rag.pdf_chunking import PdfChunkingConfig, chunk_pdf_document
from ai_learning_rag.pdf_indexing import (
    ChunkRejectionCode,
    PdfChunkIndexer,
    PdfIndexingPolicy,
    PdfIndexingReport,
)
from ai_learning_rag.pdf_routing import PdfRoutingConfig
from ai_learning_rag.pdf_routing_cache import (
    PdfRoutingCache,
    RoutingCacheMode,
    RoutingCacheStatus,
)
from ai_learning_rag.pdf_schema import ExtractionProvenance, TraceablePdfChunk
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    CorpusInventory,
    RetrievalEvaluationDataset,
    ReviewStatus,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview


@dataclass(frozen=True)
class PreparedDocument:
    generation: DocumentGeneration
    chunks: tuple[TraceablePdfChunk, ...]
    registry: dict[str, ExtractionProvenance]


def uncovered_ranges(
    start: int, end: int, coverage: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find the parts of a source text span not covered by accepted candidate chunks."""
    gaps = []
    position = start
    for span_start, span_end in sorted(coverage):
        if span_end <= position:
            continue
        if span_start >= end:
            break
        if span_start > position:
            gaps.append((position, span_start))
        position = min(end, span_end)
    if position < end:
        gaps.append((position, end))
    return gaps


def accepted_chunks(
    chunks: tuple[TraceablePdfChunk, ...], report: PdfIndexingReport
) -> tuple[TraceablePdfChunk, ...]:
    accepted = {decision.chunk_fingerprint for decision in report.decisions if decision.accepted}
    return tuple(chunk for chunk in chunks if chunk.source_fingerprint in accepted)


def find_missing_spans(
    baseline: tuple[TraceablePdfChunk, ...], candidate: tuple[TraceablePdfChunk, ...]
) -> list[dict]:
    coverage = defaultdict(list)
    for chunk in candidate:
        for region in chunk.source_regions:
            coverage[chunk.document_id, region.block_id].append(
                (region.source_text_start, region.source_text_end)
            )

    missing = []
    for chunk in baseline:
        for region in chunk.source_regions:
            for start, end in uncovered_ranges(
                region.source_text_start,
                region.source_text_end,
                coverage[chunk.document_id, region.block_id],
            ):
                offset = region.chunk_content_start - region.source_text_start
                snippet = chunk.content[start + offset : end + offset]
                if snippet.strip():
                    missing.append(
                        {
                            "baseline_chunk_id": chunk.id,
                            "baseline_fingerprint": chunk.source_fingerprint,
                            "page": region.page_number,
                            "source_block_id": region.block_id,
                            "source_start": start,
                            "source_end": end,
                            "text": snippet,
                        }
                    )
    return missing


def write_report(output: Path, summary: dict) -> None:
    status = {
        "paused_token_limit": "候选有片段超过 token 上限，暂停正式评分。",
        "ready_for_annotation": "候选通过长度检查，片段标注待人工复核。",
        "approved_for_evaluation": "候选通过长度检查，片段集合与已确认的标注匹配。",
    }
    lines = ["# 5.2 · PDF 切分入库检查", "", status[summary["status"]], ""]
    lines.extend(
        (
            "| PDF | 600 字符通过 | 1,000 字符通过 | 1,000 字符组超限 | 候选最长输入 (token) |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for sample in summary["samples"]:
        lines.append(
            f"| {sample['path']} | {sample['baseline_accepted']} | "
            f"{sample['candidate_accepted']} | "
            f"{len(sample['over_token_limit'])} | {sample['max_candidate_tokens']} |"
        )
    for sample in summary["samples"]:
        if not sample["over_token_limit"] and not sample["missing_baseline_spans"]:
            continue
        lines.extend(("", f"## {sample['path']}", ""))
        lines.extend(
            f"- 候选 `{row['chunk_id']}` 位于 PDF 第 {row['page_start']}–{row['page_end']} 页，"
            f"输入为 **{row['token_count']} token**。原文块 {'、'.join(row['source_block_ids'])}。"
            for row in sample["over_token_limit"]
        )
        if sample["missing_baseline_spans"]:
            lines.extend(("", "下列已入库的基线原文没有被通过检查的候选片段完整覆盖。", ""))
            lines.extend(
                f"- PDF 第 {row['page']} 页，基线 `{row['baseline_chunk_id']}`，"
                f"原文块 `{row['source_block_id']}`，"
                f"字符 {row['source_start']}–{row['source_end']}。"
                f"关联标注题目 {'、'.join(row['related_queries']) or '无'}。"
                for row in sample["missing_baseline_spans"]
            )
    lines.extend(("", "逐片段文字、PDF 页码和区域在 `candidate-chunks.json` 中。"))
    (output / "gate-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "gate-comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_datasets(
    baseline: RetrievalEvaluationDataset,
    candidate: RetrievalEvaluationDataset,
    inventory: CorpusInventory,
) -> None:
    if candidate.corpus != inventory.snapshot:
        raise ValueError("the approved annotation snapshot differs from the prepared candidate")
    if (
        baseline.dataset_id != candidate.dataset_id
        or baseline.dataset_version == candidate.dataset_version
    ):
        raise ValueError(
            "the candidate requires its own version in the baseline evaluation dataset"
        )
    if (baseline.recall_ks, baseline.mrr_k) != (candidate.recall_ks, candidate.mrr_k):
        raise ValueError("cutoffs changed between the two evaluation datasets")
    before = [(case.query_id, case.query, case.case_type) for case in baseline.cases]
    after = [(case.query_id, case.query, case.case_type) for case in candidate.cases]
    if before != after or any(
        case.review_status is not ReviewStatus.APPROVED
        for case in (*baseline.cases, *candidate.cases)
    ):
        raise ValueError("both groups must use the same questions with approved labels")


def evidence_rank(evidence: TraceablePdfChunk, hits: list[TraceablePdfChunk | None]) -> int | None:
    for rank in range(1, len(hits) + 1):
        covered_by = tuple(hit for hit in hits[:rank] if hit is not None)
        if not find_missing_spans((evidence,), covered_by):
            return rank
    return None


def compare_reports(
    baseline: dict,
    candidate: dict,
    original_chunks: dict[ChunkIdentity, TraceablePdfChunk],
    candidate_chunks: dict[ChunkIdentity, TraceablePdfChunk],
) -> list[dict]:
    rows = []
    for old, new in zip(baseline["queries"], candidate["queries"], strict=True):
        if old["query_id"] != new["query_id"] or old["query"] != new["query"]:
            raise ValueError("report questions changed between the cutoffs")
        old_relevant = {
            ChunkIdentity(hit["document_id"], hit["chunk_fingerprint"]): hit["rank"]
            for hit in old["hits"]
            if hit["relevant"]
        }
        # Keep positions of irrelevant hits. An evidence span must be covered at its true rank.
        old_hits = [
            original_chunks[ChunkIdentity(hit["document_id"], hit["chunk_fingerprint"])]
            if hit["relevant"]
            else None
            for hit in old["hits"]
        ]
        new_hits = [
            candidate_chunks[ChunkIdentity(hit["document_id"], hit["chunk_fingerprint"])]
            if hit["relevant"]
            else None
            for hit in new["hits"]
        ]
        evidence = []
        for reference in old["relevant_chunks"]:
            identity = ChunkIdentity(reference["document_id"], reference["chunk_fingerprint"])
            chunk = original_chunks[identity]
            before = evidence_rank(chunk, old_hits)
            after = evidence_rank(chunk, new_hits)
            evidence.append(
                {
                    "chunk_id": chunk.id,
                    "sample_id": reference["sample_id"],
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "source_block_ids": list(chunk.source_block_ids),
                    "baseline_labeled_rank": old_relevant.get(identity),
                    "baseline_coverage_rank": before,
                    "candidate_coverage_rank": after,
                }
            )
        rows.append(
            {
                "query_id": old["query_id"],
                "case_type": old["case_type"],
                "relevant_total": [len(old["relevant_chunks"]), len(new["relevant_chunks"])],
                "hits_at_k": {
                    k: [sum(hit["relevant"] for hit in row["hits"][:k]) for row in (old, new)]
                    for k in (1, 3, 5)
                },
                "first_relevant_rank": [
                    next((hit["rank"] for hit in row["hits"] if hit["relevant"]), None)
                    for row in (old, new)
                ],
                "source_evidence": evidence,
            }
        )
    return rows


def write_comparison(output: Path, baseline: dict, candidate: dict, rows: list[dict]) -> None:
    write_json(
        output / "comparison.json", {"baseline": baseline, "candidate": candidate, "queries": rows}
    )
    lines = [
        "# 5.2 · PDF 切分对照",
        "",
        "600 字符组复用已验证的 5.1 基线和 Embedding 默认组计时。",
        "1,000 字符组按 v1.1.0 标注评分。各组 Recall 的分母来自各自的标注。",
        "",
        "| 组别 | 片段数 | 向量字节数 | 构建用时 (s) | 检索中位数 (ms) | "
        "Recall@1 | Recall@3 | Recall@5 | MRR@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, group in (("600 字符", baseline), ("1,000 字符", candidate)):
        metrics = group["metrics"]
        recall = metrics["recall_at_k"]
        elapsed = group["build_seconds"]
        build = "沿用已有索引" if elapsed is None else f"{elapsed:.3f}"
        lines.append(
            f"| {label} | {group['chunk_count']} | {group['vector_bytes']} | {build} | "
            f"{group['retrieval_ms']['median']:.2f} | {recall['1']:.4f} | {recall['3']:.4f} | "
            f"{recall['5']:.4f} | {metrics['mrr']:.4f} |"
        )
    lines.extend(
        (
            "",
            "## 逐题相关片段数",
            "",
            "单元格按 600 → 1,000 字符组显示。两组的具体 PDF 证据可在逐题报告中定位。",
            "",
            "| 题目 | 相关总数 | 前 1 条 | 前 3 条 | 前 5 条 |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for row in rows:
        if row["case_type"] != "answerable":
            continue
        totals = " → ".join(map(str, row["relevant_total"]))
        counts = " | ".join(" → ".join(map(str, count)) for count in row["hits_at_k"].values())
        lines.append(f"| `{row['query_id']}` | {totals} | {counts} |")
    lines.extend(("", "## 原文证据覆盖变化", ""))
    for row in rows:
        for source in row["source_evidence"]:
            before = source["baseline_coverage_rank"]
            after = source["candidate_coverage_rank"]
            if before != after:
                lines.append(
                    f"- `{row['query_id']}`，{source['sample_id']} 第 {source['page_start']} 页，"
                    f"原片段 `{source['chunk_id']}` 完整覆盖位置 {before or '前五之外'} "
                    f"→ {after or '前五之外'}。"
                )
    lines.extend(
        (
            "",
            "详细结果见 [600 字符组](600/retrieval-evaluation.md) 和 "
            "[1,000 字符组](1000/retrieval-evaluation.md)。源页区域和逐题耗时随 JSON 保存。",
        )
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_index(
    index: PostgresVectorIndex,
    target: VectorIndexManifest,
    documents: list[PreparedDocument],
    embedder: Qwen3Embedder,
    policy: PdfIndexingPolicy,
) -> tuple[VectorIndexManifest, int, float | None]:
    started = perf_counter()
    current = index.get_manifest(target.index_id)
    if current is None:
        index.ensure_index(target)
        current = target
    if current.status is IndexStatus.ACTIVE:
        return current, 0, None
    if current.status not in {IndexStatus.BUILDING, IndexStatus.READY}:
        raise RuntimeError(f"cannot use candidate index in status {current.status}")
    indexer = PdfChunkIndexer(
        index=index,
        embedder=embedder,
        token_counter=embedder,
        max_input_tokens=target.max_input_tokens,
        policy=policy,
    )
    written = 0
    for document in documents:
        result = indexer.index_document(
            generation=document.generation,
            chunks=document.chunks,
            provenance_registry=document.registry,
        )
        if (
            result.replacement is None
            or result.replacement.status is DocumentReplaceStatus.CONCURRENT_UPDATE
        ):
            raise RuntimeError(
                f"failed to write candidate generation {document.generation.generation_id}"
            )
        written += result.replacement.record_count or 0
    if current.status is IndexStatus.BUILDING:
        index.ensure_index(replace(target, status=IndexStatus.READY))
    active = replace(target, status=IndexStatus.ACTIVE)
    index.ensure_index(active)
    return active, written, perf_counter() - started


def run_evaluation(
    *,
    output: Path,
    control: Path,
    baseline: VectorIndexManifest,
    baseline_dataset: RetrievalEvaluationDataset,
    target: VectorIndexManifest,
    dataset: RetrievalEvaluationDataset,
    inventory: CorpusInventory,
    documents: list[PreparedDocument],
    original_chunks: dict[ChunkIdentity, TraceablePdfChunk],
    candidate_chunks: dict[ChunkIdentity, TraceablePdfChunk],
    embedder: Qwen3Embedder,
    policy: PdfIndexingPolicy,
) -> None:
    before = json.loads((control / "retrieval-evaluation.json").read_text(encoding="utf-8"))
    baseline_time = json.loads((control / "measurements.json").read_text(encoding="utf-8"))
    if (
        before["corpus"] != baseline_dataset.corpus.to_payload()
        or before["dataset"]["dataset_version"] != baseline_dataset.dataset_version
        or before["dataset"]["dataset_id"] != baseline_dataset.dataset_id
        or before["index"]["index_id"] != baseline.index_id
        or before["index"]["identity"] != baseline.identity_payload
        or baseline_time["index_id"] != baseline.index_id
        or baseline_time["chunk_set_fingerprint"] != baseline_dataset.corpus.chunk_set_fingerprint
        or baseline_time["metrics"] != before["metrics"]
        or baseline_time["query_prefix"] != embedder.default_query_prompt()
        or (baseline_time["warmup_rounds"], baseline_time["timed_rounds"]) != (1, 3)
    ):
        raise ValueError("the control reports do not match the validated default group")
    for case, query in zip(baseline_dataset.cases, before["queries"], strict=True):
        if (
            (query["query_id"], query["query"], query["case_type"])
            != (case.query_id, case.query, case.case_type.value)
            or query["relevant_chunks"] != [ref.to_payload() for ref in case.relevant_chunks]
            or len(query["hits"]) != baseline_dataset.max_k
            or len(baseline_time["queries"][case.query_id]["samples_ms"]) != 3
        ):
            raise ValueError(f"the control report changed approved content for {case.query_id}")

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
            raise ValueError("stored baseline index does not match the control reports")
        base_records, base_bytes = index.read_active_records(baseline.index_id)
        if records_by_chunk(base_records).keys() != original_chunks.keys():
            raise ValueError("stored baseline chunks differ from the control snapshot")
        if baseline_time["vector_bytes"] != base_bytes or baseline_time["chunk_count"] != len(
            base_records
        ):
            raise ValueError("baseline storage differs from the control measurements")

        active, written, elapsed = build_candidate_index(index, target, documents, embedder, policy)
        records, vector_bytes = index.read_active_records(active.index_id)
        if records_by_chunk(records).keys() != candidate_chunks.keys():
            raise ValueError("indexed candidate chunks differ from the approved snapshot")
        candidate_dir = output / "1000"
        candidate_dir.mkdir()
        storage = {
            "index_id": active.index_id,
            "chunk_set_fingerprint": inventory.snapshot.chunk_set_fingerprint,
            "chunk_count": len(records),
            "vector_bytes": vector_bytes,
            "records_written": written,
            "build_seconds": elapsed,
        }
        write_json(candidate_dir / "build.json", storage)
        rankings, times = rank_group(index, active, dataset, embedder, None)
        report = evaluate_retrieval(
            dataset=dataset, inventory=inventory, results_by_query=rankings, index_manifest=active
        )
        after = json.loads(report.to_json())
        candidate_time = storage | times | {"metrics": after["metrics"]}
        write_json(candidate_dir / "measurements.json", candidate_time)
        (candidate_dir / "retrieval-evaluation.json").write_text(report.to_json(), encoding="utf-8")
        (candidate_dir / "retrieval-evaluation.md").write_text(
            report.to_markdown(), encoding="utf-8"
        )
        baseline_dir = output / "600"
        baseline_dir.mkdir()
        for filename in (
            "retrieval-evaluation.json",
            "retrieval-evaluation.md",
            "measurements.json",
        ):
            copyfile(control / filename, baseline_dir / filename)
        rows = compare_reports(before, after, original_chunks, candidate_chunks)
        write_comparison(output, baseline_time | {"build_seconds": None}, candidate_time, rows)
        print(f"comparison complete: {output / 'comparison.md'}")
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "baseline-preview", "dataset", "output-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, help="approved 1,000-character labels")
    parser.add_argument(
        "--baseline-control", type=Path, help="default group reports and measurements"
    )
    args = parser.parse_args()
    if bool(args.candidate_dataset) != bool(args.baseline_control):
        parser.error("--candidate-dataset and --baseline-control must be supplied together")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preview = json.loads(args.baseline_preview.read_text(encoding="utf-8"))
    baseline, base_index, _ = evaluation_inputs_from_preview(preview)
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    if (
        dataset.corpus != baseline.snapshot
        or manifest["logical_index_name"] != base_index.logical_name
    ):
        raise ValueError("the supplied dataset and manifest do not match the confirmed baseline")
    base_config = PdfChunkingConfig(target_size=400, hard_max_size=600, overlap_percent=20)
    candidate_config = replace(base_config, hard_max_size=1000)
    if base_config.chunking_profile != baseline.snapshot.chunking_profile:
        raise ValueError("the 600-character configuration differs from the confirmed baseline")
    target_identity = base_index.identity_payload | {"logical_name": "stage52-chunking-1000-v1"}
    target_index = VectorIndexManifest.create(
        **target_identity,
        required_pgvector_version=base_index.required_pgvector_version,
        audit_metadata=base_index.audit_metadata,
    )

    embedder = load_baseline_embedder(base_index)
    policy = PdfIndexingPolicy()
    indexer = PdfChunkIndexer(
        index=InMemoryVectorIndex(),
        embedder=embedder,
        token_counter=embedder,
        max_input_tokens=base_index.max_input_tokens,
        policy=policy,
    )
    routing_config = PdfRoutingConfig.from_json(Path(manifest["routing_config"]))
    routing_cache = PdfRoutingCache(Path(manifest["routing_cache_dir"]), RoutingCacheMode.READ_ONLY)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_samples = {row["sample_id"]: row for row in preview["samples"]}
    if set(original_samples) != {sample["sample_id"] for sample in manifest["samples"]}:
        raise ValueError("the manifest specifies a different set of PDF samples")
    original_chunks = {}
    candidate_chunks = {}
    candidate_documents = []
    catalog = []
    sample_reports = []
    related = defaultdict(list)
    for case in dataset.cases:
        for reference in case.relevant_chunks:
            related[reference.identity].append(case.query_id)

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        routing, ocr = load_preview_document(sample, routing_config, routing_cache)
        if any(
            event.status in {RoutingCacheStatus.MISS, RoutingCacheStatus.CORRUPT}
            for event in routing.cache_events
        ):
            raise ValueError(f"{sample_id} has missing or corrupt frozen routing inputs")
        document = routing.document
        registry = {item.id: item for item in document.extraction_provenances}
        accepted_by_size = {}
        gates = {}
        for config, index in ((base_config, base_index), (candidate_config, target_index)):
            generation = preview_generation(
                document, index_id=index.index_id, ocr_fingerprint=ocr, config=config, policy=policy
            )
            chunks = chunk_pdf_document(document, config).chunks
            report = indexer.build_gate_report(
                generation=generation, chunks=chunks, provenance_registry=registry
            )
            hard_max = config.hard_max_size
            accepted_by_size[hard_max] = accepted_chunks(chunks, report)
            gates[hard_max] = report
            if hard_max == 1000:
                candidate_documents.append(PreparedDocument(generation, chunks, registry))
            (args.output_dir / f"{sample_id}-{hard_max}.gate.json").write_text(
                report.to_json(), encoding="utf-8"
            )
            if hard_max == 600:
                original = original_samples[sample_id]
                if (generation.generation_id, report.accepted_count, report.rejected_count) != (
                    original["generation_id"],
                    original["gate_accepted"],
                    original["gate_rejected"],
                ):
                    raise ValueError(
                        f"600-character inputs or gate decisions changed for {sample_id}"
                    )

        before, after = accepted_by_size[600], accepted_by_size[1000]
        original_chunks.update(
            (ChunkIdentity(document.document_id, c.source_fingerprint), c) for c in before
        )
        candidate_chunks.update(
            (ChunkIdentity(document.document_id, c.source_fingerprint), c) for c in after
        )
        report = gates[1000]
        chunk_by_id = {chunk.id: chunk for chunk in candidate_documents[-1].chunks}
        rejected = []
        for decision in report.decisions:
            chunk = chunk_by_id[decision.chunk_id]
            entry = asdict(chunk) | {
                "sample_id": sample_id,
                "path": sample["path"],
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "accepted": decision.accepted,
                "token_count": decision.token_count,
                "rejection_code": decision.rejection_code,
                "source_block_ids": list(chunk.source_block_ids),
                "chunk_id": chunk.id,
            }
            catalog.append(entry)
            if decision.rejection_code is ChunkRejectionCode.TOKEN_LIMIT_EXCEEDED:
                rejected.append(entry)
        missing = find_missing_spans(before, after)
        for row in missing:
            row["related_queries"] = related[
                ChunkIdentity(document.document_id, row["baseline_fingerprint"])
            ]
        sample_reports.append(
            {
                "path": sample["path"],
                "baseline_accepted": len(before),
                "candidate_accepted": len(after),
                "max_candidate_tokens": max(
                    (row.token_count or 0 for row in report.decisions), default=0
                ),
                "over_token_limit": rejected,
                "missing_baseline_spans": missing,
            }
        )
    if original_chunks.keys() != set(baseline.chunks):
        raise ValueError("the 600-character gate did not reproduce the confirmed chunk set")
    candidate = CorpusInventory.create(
        corpus_id=baseline.snapshot.corpus_id,
        samples=baseline.snapshot.samples,
        chunking_profile=candidate_config.chunking_profile,
        gate_profile=policy.profile,
        chunks=candidate_chunks,
    )
    (args.output_dir / "candidate-chunks.json").write_text(
        json.dumps(
            {"corpus_inventory": candidate.to_payload(), "chunks": catalog},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = (
        "paused_token_limit"
        if any(row["over_token_limit"] for row in sample_reports)
        else "ready_for_annotation"
    )
    approved = None
    if status == "ready_for_annotation" and args.candidate_dataset is not None:
        approved = load_retrieval_evaluation_dataset(args.candidate_dataset)
        validate_datasets(dataset, approved, candidate)
        status = "approved_for_evaluation"
    write_report(
        args.output_dir,
        {
            "status": status,
            "token_limit": base_index.max_input_tokens,
            "baseline_config": asdict(base_config),
            "candidate_config": asdict(candidate_config),
            "baseline_corpus": baseline.snapshot.to_payload(),
            "candidate_corpus": candidate.snapshot.to_payload(),
            "embedding_identity": base_index.identity_payload,
            "planned_candidate_index_id": target_index.index_id,
            "samples": sample_reports,
        },
    )
    print(f"{status}: {args.output_dir / 'gate-comparison.md'}")
    if approved is not None:
        run_evaluation(
            output=args.output_dir,
            control=args.baseline_control,
            baseline=base_index,
            baseline_dataset=dataset,
            target=target_index,
            dataset=approved,
            inventory=candidate,
            documents=candidate_documents,
            original_chunks=original_chunks,
            candidate_chunks=candidate_chunks,
            embedder=embedder,
            policy=policy,
        )


if __name__ == "__main__":
    main()

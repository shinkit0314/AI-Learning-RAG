"""Compare exact vector, PostgreSQL keyword, and RRF retrieval on PDF evaluation v1.0.0."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from time import perf_counter

from pdf_embedding_experiment import check_control, compare, load_baseline_embedder, write_json
from sqlalchemy import Engine, text

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.document_index import SearchResult
from ai_learning_rag.hybrid_ranking import (
    HYBRID_CANDIDATE_LIMIT,
    HYBRID_OUTPUT_LIMIT,
    RRF_RANK_CONSTANT,
    HybridSearchResult,
)
from ai_learning_rag.pdf_retrieval import PdfRetriever
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    QueryEvaluation,
    RetrievalEvaluationDataset,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview

METHODS = ("vector", "keyword", "hybrid")
METHOD_LABELS = {"vector": "精确向量", "keyword": "纯关键词", "hybrid": "混合"}
FIGURE_QUERY = "layout-figure-two-dimensions"
FIGURE_SAMPLE = "layout-001"
SYSTEM_DESIGN_QUERY = "layout-taxonomy-system-design"
KEYWORD_PROBES = {
    FIGURE_QUERY: "Fig. 2",
    "layout-agent-characteristics": "Fig. 4",
    "native-pomdp-observation-state": "POMDP",
    "scan-disk-failure-frequency": "10000",
}


def hit_trace(hits: Sequence[SearchResult]) -> list[dict]:
    return [
        {
            "rank": rank,
            "score": hit.score,
            "record_id": hit.record.record_id,
            "document_id": hit.record.document_id,
            "chunk_fingerprint": hit.record.chunk_fingerprint,
            "content_type": hit.record.content_type,
            "page_start": hit.record.page_start,
            "page_end": hit.record.page_end,
            "provenance": hit.record.provenance,
        }
        for rank, hit in enumerate(hits, 1)
    ]


def hybrid_trace(result: HybridSearchResult) -> dict:
    return {
        "vector": hit_trace(result.vector_candidates),
        "keyword": hit_trace(result.keyword_candidates),
        "fused": hit_trace(result.fused_candidates),
    }


def rank_of(hits: Sequence[SearchResult], identity: ChunkIdentity) -> int | None:
    return next(
        (
            rank
            for rank, hit in enumerate(hits, 1)
            if (hit.record.document_id, hit.record.chunk_fingerprint)
            == (identity.document_id, identity.chunk_fingerprint)
        ),
        None,
    )


def collect_runs(
    index: PostgresVectorIndex,
    retriever: PdfRetriever,
    dataset: RetrievalEvaluationDataset,
    index_id: str,
    document_id: str,
) -> tuple[dict, dict, dict, dict]:
    rankings: dict[str, dict[str, Sequence[SearchResult]]] = {name: {} for name in METHODS}
    hybrids: dict[str, HybridSearchResult] = {}
    scoped: dict[str, Sequence[SearchResult] | HybridSearchResult] = {}
    timings: dict[str, dict[str, list[float]]] = {
        name: {} for name in (*METHODS, "vector_scoped", "hybrid_scoped")
    }
    for run in range(4):
        for case in dataset.cases:
            query_id, query = case.query_id, case.query
            names = (
                (*METHODS, "vector_scoped", "hybrid_scoped")
                if query_id == FIGURE_QUERY
                else METHODS
            )
            for name in names:
                scope = document_id if name.endswith("_scoped") else None
                start = perf_counter()
                if name.startswith("hybrid"):
                    prepared = retriever.prepare_search(query, document_id=scope)
                    result = index.search_hybrid(prepared, query)
                    hits = result.hits
                elif name == "keyword":
                    hits = index.search_keywords(query, index_id=index_id)
                else:
                    hits = retriever.search(query, document_id=scope)
                elapsed = (perf_counter() - start) * 1000
                if run:
                    timings[name].setdefault(query_id, []).append(elapsed)
                if run == 1:
                    if name == "hybrid":
                        hybrids[query_id] = result
                    if name == "hybrid_scoped":
                        scoped[name] = result
                    elif name == "vector_scoped":
                        scoped[name] = hits
                    else:
                        rankings[name][query_id] = hits
    return rankings, hybrids, scoped, timings


def filter_comparison(
    rankings: dict,
    scoped: dict,
    document_id: str,
    gold: ChunkIdentity,
) -> dict:
    hybrid = scoped["hybrid_scoped"]
    if any(
        hit.record.document_id != document_id
        for hit in (*hybrid.vector_candidates, *hybrid.keyword_candidates)
    ):
        raise RuntimeError("a scoped hybrid candidate crossed the requested document boundary")
    comparisons = {}
    for name, after in (("vector", scoped["vector_scoped"]), ("hybrid", hybrid.hits)):
        before = rankings[name][FIGURE_QUERY]
        outside = [
            sum(h.record.document_id != document_id for h in hits) for hits in (before, after)
        ]
        ranks = [rank_of(hits, gold) for hits in (before, after)]
        if outside[1] or (name == "vector" and ranks[1] is None):
            raise RuntimeError(f"{name} failed the Fig. 2 document-scope check")
        comparisons[name] = {
            "outside_before_after": outside,
            "gold_rank_before_after": ranks,
            "unscoped": hit_trace(before),
            "scoped": hit_trace(after),
        }
    comparisons["scoped_candidates"] = hybrid_trace(hybrid)
    return comparisons


def system_design_diagnostics(
    index: PostgresVectorIndex,
    retriever: PdfRetriever,
    query_report: QueryEvaluation,
    hybrid: HybridSearchResult,
    chunk_count: int,
) -> list[dict]:
    rows = []
    references = {ref.identity: ref for ref in query_report.case.relevant_chunks}
    for identity in query_report.missing_relevant:
        reference = references[identity]
        rows.append(
            {
                **identity.to_payload(),
                "page_start": reference.page_start,
                "page_end": reference.page_end,
                "source_block_ids": reference.source_block_ids,
                "vector_candidate_rank": rank_of(hybrid.vector_candidates, identity),
                "keyword_candidate_rank": rank_of(hybrid.keyword_candidates, identity),
                "fused_rank": rank_of(hybrid.fused_candidates, identity),
            }
        )
    outside_window = [row for row in rows if row["fused_rank"] is None]
    if outside_window:
        question = query_report.case.query
        prepared = retriever.prepare_search(question, top_k=chunk_count)
        deep_vector = index.search(prepared)
        deep_keyword = index.search_keywords(
            question, index_id=prepared.index_id, top_k=chunk_count
        )
        for row in outside_window:
            identity = ChunkIdentity(row["document_id"], row["chunk_fingerprint"])
            row["deep_vector_rank"] = rank_of(deep_vector, identity)
            row["deep_keyword_rank"] = rank_of(deep_keyword, identity)
    return rows


def keyword_diagnostics(
    index: PostgresVectorIndex,
    engine: Engine,
    dataset: RetrievalEvaluationDataset,
    index_id: str,
    keyword_rankings: dict[str, Sequence[SearchResult]],
) -> dict:
    output = {}
    with engine.connect() as connection:
        for case in dataset.cases:
            if case.query_id not in KEYWORD_PROBES:
                continue
            probe = KEYWORD_PROBES[case.query_id]
            gold = case.relevant_chunks[0]
            tokens = connection.execute(
                text(
                    "SELECT token, lexemes FROM ts_debug('pg_catalog.simple', :content) "
                    "WHERE lexemes IS NOT NULL"
                ),
                {"content": gold.content},
            ).mappings()
            probe_hits = index.search_keywords(probe, index_id=index_id)
            output[case.query_id] = {
                "probe": probe,
                "probe_rank": rank_of(probe_hits, gold.identity),
                "full_question_rank": rank_of(keyword_rankings[case.query_id], gold.identity),
                "gold_tokens": [dict(token) for token in tokens],
                "probe_hits": hit_trace(probe_hits),
            }
    return output


def write_summary(
    output: Path,
    reports: dict,
    timings: dict,
    scope: dict,
    changes: dict,
    system_design: list[dict],
    probes: dict,
) -> None:
    lines = [
        "# 5.3 · PDF 检索对照",
        "",
        "| 检索 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 耗时中位数（毫秒） | 范围（毫秒） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in METHODS:
        metrics = reports[name].aggregate
        recall = " | ".join(f"{value:.4f}" for _, value in metrics.recall_at_k)
        ms = [ms for times in timings[name].values() for ms in times]
        lines.append(
            f"| [{METHOD_LABELS[name]}]({name}/retrieval-evaluation.md) "
            f"| {recall} | {metrics.mrr:.4f} | "
            f"{median(ms):.2f} | {min(ms):.2f}–{max(ms):.2f} |"
        )
    lines.extend(("", "## Fig. 2 指定文档对照", ""))
    for name in ("vector", "hybrid"):
        before, after = scope[name]["outside_before_after"]
        ranks = " → ".join(
            str(rank) if rank is not None else "未命中"
            for rank in scope[name]["gold_rank_before_after"]
        )
        lines.append(f"- {METHOD_LABELS[name]}越界 {before} → {after} 条，图注名次 {ranks}。")
    lines.extend(
        (
            "",
            "## 逐题变化",
            "",
            "| 题目 | 向量／关键词／混合的首个相关名次 | 向量／关键词／混合的前五命中数 |",
            "| --- | --- | --- |",
        )
    )
    for vector, keyword, hybrid in zip(*(reports[name].queries for name in METHODS), strict=True):
        query_id = vector.case.query_id
        cases = (vector, keyword, hybrid)
        if vector.reciprocal_rank is None:
            lines.append(f"| `{query_id}` | 诊断题 | 见各组报告 |")
            continue
        ranks = " / ".join(str(case.first_relevant_rank or "未命中") for case in cases)
        counts = " / ".join(str(sum(hit.relevant for hit in case.hits)) for case in cases)
        lines.append(f"| `{query_id}` | {ranks} | {counts} |")
    lines.extend(("", "## 证据定位", ""))
    for change in changes["hybrid"]:
        lost = sum(evidence["after"] is None for evidence in change["evidence_ranks"])
        gained = sum(evidence["before"] is None for evidence in change["evidence_ranks"])
        before_rank, after_rank = change["first_relevant_rank"]
        if lost or gained or before_rank != after_rank:
            lines.append(
                f"- `{change['query_id']}` 的首个相关结果 {before_rank} → "
                f"{after_rank or '未命中'}，混合前五条新增 {gained} 段、丢失 {lost} 段。"
            )
    for row in system_design:
        lines.append(
            f"- 系统设计题第 {row['page_start']} 页的基线遗漏 `{row['chunk_fingerprint']}`，"
            f"向量候选 {row['vector_candidate_rank'] or '未进候选'}，"
            f"关键词候选 {row['keyword_candidate_rank'] or '未进候选'}，"
            f"融合名次 {row['fused_rank'] or '未进候选'}。"
        )
        if "deep_vector_rank" in row:
            lines.append(
                f"  扩展诊断名次为向量 {row['deep_vector_rank'] or '未命中'}、"
                f"关键词 {row['deep_keyword_rank'] or '未命中'}。"
            )
    lines.extend(("", "## 关键词诊断", ""))
    for probe in probes.values():
        lines.append(
            f"- `{probe['probe']}` 的标注证据名次。"
            f"短词查询 {probe['probe_rank'] or '未命中'}，"
            f"完整原题 {probe['full_question_rank'] or '未命中'}。"
        )
    lines.extend(
        (
            "",
            "候选名次、逐题片段身份、来源区域和逐次计时见 [comparison.json](comparison.json)。",
            "",
        )
    )
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    preview = json.loads(args.preview.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    inventory, manifest, preview_rankings = evaluation_inputs_from_preview(preview)
    preview_report = evaluate_retrieval(
        dataset=dataset,
        inventory=inventory,
        index_manifest=manifest,
        results_by_query=preview_rankings,
    )
    check_control(preview_report.to_payload(), reference)
    selected = next(
        sample for sample in inventory.snapshot.samples if sample.sample_id == FIGURE_SAMPLE
    )
    figure_case = next(case for case in dataset.cases if case.query_id == FIGURE_QUERY)
    gold = figure_case.relevant_chunks[0].identity
    if gold.document_id != selected.document_id or dataset.max_k != HYBRID_OUTPUT_LIMIT:
        raise ValueError("the approved Fig. 2 scope or output limit does not match the dataset")

    engine = DatabaseSettings.from_env().create_engine()
    try:
        with engine.connect() as connection:
            runtime = {
                "postgresql": connection.execute(text("SHOW server_version")).scalar_one(),
                "pgvector": connection.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar_one(),
            }
        index = PostgresVectorIndex(engine)
        stored_manifest = index.get_manifest(manifest.index_id)
        if stored_manifest != manifest:
            raise ValueError("the active database manifest differs from the preview")
        records, _ = index.read_active_records(manifest.index_id)
        identities = [ChunkIdentity(r.document_id, r.chunk_fingerprint) for r in records]
        if len(identities) != len(inventory.chunks) or set(identities) != set(inventory.chunks):
            raise ValueError("the active database corpus differs from the preview")

        retriever = PdfRetriever(
            index=index, manifest=manifest, embedder=load_baseline_embedder(manifest)
        )
        rankings, hybrids, scoped, timings = collect_runs(
            index, retriever, dataset, manifest.index_id, selected.document_id
        )
        reports = {
            name: evaluate_retrieval(
                dataset=dataset,
                inventory=inventory,
                index_manifest=manifest,
                results_by_query=rankings[name],
            )
            for name in METHODS
        }
        check_control(reports["vector"].to_payload(), reference)
        scope = filter_comparison(rankings, scoped, selected.document_id, gold)
        changes = {
            name: compare(reports["vector"], reports[name]) for name in ("keyword", "hybrid")
        }
        system_design = next(
            row for row in reports["vector"].queries if row.case.query_id == SYSTEM_DESIGN_QUERY
        )
        missing_evidence = system_design_diagnostics(
            index,
            retriever,
            system_design,
            hybrids[SYSTEM_DESIGN_QUERY],
            inventory.snapshot.chunk_count,
        )
        probes = keyword_diagnostics(index, engine, dataset, manifest.index_id, rankings["keyword"])
    finally:
        engine.dispose()

    for name, report in reports.items():
        group = args.output_dir / name
        group.mkdir(parents=True, exist_ok=True)
        (group / "retrieval-evaluation.json").write_text(report.to_json(), encoding="utf-8")
        (group / "retrieval-evaluation.md").write_text(report.to_markdown(), encoding="utf-8")
    write_json(
        args.output_dir / "comparison.json",
        {
            "dataset": dataset.dataset_version,
            "corpus": inventory.snapshot.to_payload(),
            "index": reports["vector"].to_payload()["index"],
            "database_runtime": runtime,
            "settings": {
                "candidate_limit": HYBRID_CANDIDATE_LIMIT,
                "output_limit": HYBRID_OUTPUT_LIMIT,
                "rrf_constant": RRF_RANK_CONSTANT,
                "keyword_config": "pg_catalog.simple",
                "scope_query": FIGURE_QUERY,
                "scope_document_id": selected.document_id,
                "warmups": 1,
                "timed_runs": 3,
            },
            "metrics": {name: report.to_payload()["metrics"] for name, report in reports.items()},
            "elapsed_ms": timings,
            "differences": changes,
            "filter": scope,
            "hybrid_candidates": {
                query_id: hybrid_trace(result) for query_id, result in hybrids.items()
            },
            "system_design_missing_evidence": missing_evidence,
            "keyword_diagnostics": probes,
        },
    )
    write_summary(args.output_dir, reports, timings, scope, changes, missing_evidence, probes)
    print(f"dataset={dataset.dataset_id}@{dataset.dataset_version} -> {args.output_dir}")


if __name__ == "__main__":
    main()

"""Human-review worksheet rendering for proposed retrieval relevance labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    EvaluationCaseType,
    RelevantChunkReference,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
    ReviewStatus,
)
from ai_learning_rag.retrieval_evaluation_preview import evaluation_inputs_from_preview


def build_annotation_review_markdown(payload: Mapping[str, object]) -> str:
    """Render preview hits as candidates, never as approved relevance gold."""

    if payload.get("schema_version") != "pdf-indexing-retrieval-preview-v1":
        raise ValueError("unsupported retrieval preview payload")
    queries = _require_sequence(payload.get("queries"), label="queries")
    lines = [
        "# PDF retrieval annotation review",
        "",
        "> This worksheet contains AI-prepared candidates, not approved relevance gold.",
        "> Check each proposed question and every retrieved chunk before approval.",
        "",
        f"- Index: `{payload.get('index_id', '')}`",
        f"- Model: `{payload.get('model_id', '')}`",
        f"- Model revision: `{payload.get('model_revision', '')}`",
        "",
        "## Review instructions",
        "",
        "For each question:",
        "",
        "1. Confirm that the question is clear and answerable from the fixed PDF corpus.",
        "2. Mark every chunk containing direct answer evidence as relevant.",
        "3. Mark topic-only or keyword-only chunks as not relevant.",
        "4. Record any missing relevant chunk that is absent from Top-5.",
        "5. Do not copy fingerprints manually; approval will be applied to the formal JSON later.",
        "",
    ]
    for raw_query in queries:
        query = _require_mapping(raw_query, label="query")
        hits = _require_sequence(query.get("hits"), label="query.hits")
        lines.extend(
            [
                f"## {query.get('query_id', '')}",
                "",
                f"- Question: {query.get('query', '')}",
                f"- Case type: `{query.get('case_type', 'answerable')}`",
                *(
                    [
                        "- Expected review source: "
                        f"`{query.get('expected_sample_id')}` "
                        f"pages {query.get('expected_pages', [])}"
                    ]
                    if query.get("expected_sample_id") is not None
                    else ["- Expected review source: none — diagnostic case"]
                ),
                "- Question decision: [ ] approve  [ ] revise  [ ] reject",
                "- Missing relevant evidence outside Top-5: [ ] no  [ ] yes — notes:",
                "",
                "| Rank | Decision | Score | Sample | Pages | Type | Content preview |",
                "| ---: | --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for raw_hit in hits:
            hit = _require_mapping(raw_hit, label="hit")
            content = str(hit.get("content", ""))
            preview = " ".join(content.split())[:180].replace("|", "\\|")
            lines.append(
                f"| {hit.get('rank', '')} | [ ] relevant [ ] not relevant | "
                f"{float(hit.get('score', 0.0)):.6f} | `{hit.get('sample_id', '')}` | "
                f"{hit.get('page_start', '')}-{hit.get('page_end', '')} | "
                f"{hit.get('content_type', '')} | {preview} |"
            )
        lines.extend(["", "### Complete candidate chunks", ""])
        for raw_hit in hits:
            hit = _require_mapping(raw_hit, label="hit")
            section_path = " > ".join(str(value) for value in hit.get("section_path", []))
            lines.extend(
                [
                    f"#### Rank {hit.get('rank', '')}",
                    "",
                    "- Decision: [ ] relevant  [ ] not relevant",
                    f"- Score: `{float(hit.get('score', 0.0)):.6f}`",
                    f"- Sample: `{hit.get('sample_id', '')}`",
                    f"- Pages: {hit.get('page_start', '')}-{hit.get('page_end', '')}",
                    f"- Section: {section_path or '(none)'}",
                    f"- Document ID: `{hit.get('document_id', '')}`",
                    f"- Chunk fingerprint: `{hit.get('chunk_fingerprint', '')}`",
                    f"- Record ID: `{hit.get('record_id', '')}`",
                    "",
                    "##### Full content",
                    "",
                    *[
                        f"> {line}" if line else ">"
                        for line in str(hit.get("content", "")).split("\n")
                    ],
                    "",
                ]
            )
    return "\n".join(lines)


def build_approved_evaluation_dataset(
    *,
    preview: Mapping[str, object],
    decisions: Mapping[str, object],
) -> RetrievalEvaluationDataset:
    """Promote explicit human decisions using audit fields from the exact chunk catalog."""

    if decisions.get("schema_version") != "pdf-retrieval-annotation-decisions-v1":
        raise ValueError("unsupported retrieval annotation decisions")
    inventory, _manifest, _rankings = evaluation_inputs_from_preview(preview)
    queries = {
        str(query.get("query_id", "")): query
        for value in _require_sequence(preview.get("queries"), label="queries")
        for query in (_require_mapping(value, label="query"),)
    }
    catalog = {
        ChunkIdentity(
            str(chunk.get("document_id", "")),
            str(chunk.get("chunk_fingerprint", "")),
        ): chunk
        for value in _require_sequence(preview.get("chunk_catalog"), label="chunk_catalog")
        for chunk in (_require_mapping(value, label="catalog chunk"),)
    }
    if set(catalog) != set(inventory.chunks):
        raise ValueError("chunk catalog does not match the compatible corpus inventory")

    cases: list[RetrievalEvaluationCase] = []
    decision_values = _require_sequence(decisions.get("cases"), label="decisions.cases")
    for value in decision_values:
        decision = _require_mapping(value, label="decision")
        query_id = str(decision.get("query_id", ""))
        query = queries.get(query_id)
        if query is None:
            raise ValueError(f"decision references unknown preview query: {query_id}")
        reference_values = _require_sequence(
            decision.get("relevant_chunks"), label="decision.relevant_chunks"
        )
        references = tuple(
            _reference_from_catalog(
                catalog,
                _require_mapping(reference, label="relevant chunk decision"),
            )
            for reference in reference_values
        )
        cases.append(
            RetrievalEvaluationCase(
                query_id=query_id,
                query=str(query.get("query", "")),
                case_type=EvaluationCaseType(str(decision.get("case_type", ""))),
                review_status=ReviewStatus.APPROVED,
                relevant_chunks=references,
                notes=str(decision.get("notes", "")),
            )
        )
    if set(queries) != {case.query_id for case in cases}:
        raise ValueError("approved decisions must cover every preview query exactly once")

    return RetrievalEvaluationDataset(
        dataset_id=str(decisions.get("dataset_id", "")),
        dataset_version=str(decisions.get("dataset_version", "")),
        corpus=inventory.snapshot,
        recall_ks=tuple(
            _require_integer(value, label="recall_k")
            for value in _require_sequence(decisions.get("recall_ks"), label="recall_ks")
        ),
        mrr_k=_require_integer(decisions.get("mrr_k"), label="mrr_k"),
        cases=tuple(cases),
    )


def _reference_from_catalog(
    catalog: Mapping[ChunkIdentity, Mapping[str, object]],
    decision: Mapping[str, object],
) -> RelevantChunkReference:
    identity = ChunkIdentity(
        str(decision.get("document_id", "")),
        str(decision.get("chunk_fingerprint", "")),
    )
    chunk = catalog.get(identity)
    if chunk is None:
        raise ValueError("approved relevant chunk is absent from the exact chunk catalog")
    return RelevantChunkReference(
        identity=identity,
        sample_id=str(chunk.get("sample_id", "")),
        page_start=_require_integer(chunk.get("page_start"), label="page_start"),
        page_end=_require_integer(chunk.get("page_end"), label="page_end"),
        content_type=str(chunk.get("content_type", "")),
        section_path=tuple(
            str(value)
            for value in _require_sequence(chunk.get("section_path"), label="section_path")
        ),
        source_block_ids=tuple(
            str(value)
            for value in _require_sequence(chunk.get("source_block_ids"), label="source_block_ids")
        ),
        content=str(chunk.get("content", "")),
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value

"""Human-review payloads for the small real-PDF retrieval preview."""

from dataclasses import dataclass

from ai_learning_rag.document_index import SearchResult
from ai_learning_rag.pdf_schema import TraceablePdfChunk


@dataclass(frozen=True, slots=True)
class RetrievalPreviewCase:
    """One versioned query and its coarse expected source location."""

    query_id: str
    query: str
    expected_sample_id: str | None
    expected_pages: tuple[int, ...]
    top_k: int
    case_type: str = "answerable"

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be blank")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.case_type not in {"answerable", "unanswerable", "ambiguous"}:
            raise ValueError(f"unsupported case_type: {self.case_type}")
        if self.case_type == "answerable":
            if self.expected_sample_id is None or not self.expected_sample_id.strip():
                raise ValueError("answerable cases require expected_sample_id")
            if not self.expected_pages:
                raise ValueError("answerable cases require expected_pages")
        elif self.expected_sample_id is not None or self.expected_pages:
            raise ValueError("diagnostic cases must not declare an expected source")
        if any(page < 1 for page in self.expected_pages):
            raise ValueError("expected_pages must contain positive page numbers")
        if tuple(sorted(set(self.expected_pages))) != self.expected_pages:
            raise ValueError("expected_pages must be sorted and unique")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


def build_chunk_catalog_entry(
    *,
    sample_id: str,
    path: str,
    chunk: TraceablePdfChunk,
) -> dict[str, object]:
    """Preserve one accepted chunk for exhaustive annotation completeness checks."""

    if not sample_id.strip():
        raise ValueError("sample_id must not be blank")
    if not path.strip():
        raise ValueError("path must not be blank")
    return {
        "sample_id": sample_id,
        "path": path,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_fingerprint": chunk.source_fingerprint,
        "chunk_index": chunk.chunk_index,
        "content_type": chunk.content_type.value,
        "section_path": list(chunk.section_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "source_block_ids": list(chunk.source_block_ids),
        "content": chunk.content,
    }


def build_query_preview(
    case: RetrievalPreviewCase,
    results: list[SearchResult],
    document_samples: dict[str, str],
) -> dict[str, object]:
    """Map retrieval records to a stable, JSON-compatible review payload."""

    hits = []
    for rank, result in enumerate(results, start=1):
        record = result.record
        sample_id = document_samples.get(record.document_id, "unknown")
        expected_document = (
            None if case.expected_sample_id is None else sample_id == case.expected_sample_id
        )
        expected_page = any(
            record.page_start <= page <= record.page_end for page in case.expected_pages
        )
        hits.append(
            {
                "rank": rank,
                "score": result.score,
                "record_id": record.record_id,
                "index_id": record.index_id,
                "generation_id": record.generation_id,
                "sample_id": sample_id,
                "document_id": record.document_id,
                "chunk_fingerprint": record.chunk_fingerprint,
                "chunk_index": record.chunk_index,
                "content_type": record.content_type,
                "section_path": list(record.section_path),
                "page_start": record.page_start,
                "page_end": record.page_end,
                "embedding_input_hash": record.embedding_input_hash,
                "embedding_input_token_count": record.embedding_input_token_count,
                "expected_document": expected_document,
                "expected_page": (
                    None if expected_document is None else expected_document and expected_page
                ),
                "content": record.content,
                "provenance": record.provenance,
            }
        )
    return {
        "query_id": case.query_id,
        "query": case.query,
        "case_type": case.case_type,
        "expected_sample_id": case.expected_sample_id,
        "expected_pages": list(case.expected_pages),
        "top_k": case.top_k,
        "matched_expected_document": (
            None
            if case.expected_sample_id is None
            else any(hit["expected_document"] for hit in hits)
        ),
        "matched_expected_page": (
            None if case.expected_sample_id is None else any(hit["expected_page"] for hit in hits)
        ),
        "hits": hits,
    }


def build_retrieval_markdown(payload: dict[str, object]) -> str:
    """Render all query previews without presenting them as formal quality metrics."""

    lines = [
        "# Real PDF retrieval preview",
        "",
        "> This is a fixed human-review preview, not a Recall@K or MRR evaluation.",
        "",
        f"- Index: `{payload['index_id']}`",
        f"- Model: `{payload['model_id']}`",
        f"- Model revision: `{payload['model_revision']}`",
        "",
    ]
    for query in payload["queries"]:  # type: ignore[union-attr]
        lines.extend(
            [
                f"## {query['query_id']}",
                "",
                f"- Query: {query['query']}",
                f"- Case type: `{query['case_type']}`",
                *(
                    [
                        f"- Expected: `{query['expected_sample_id']}` "
                        f"pages {query['expected_pages']}",
                        f"- Expected document in Top-K: {query['matched_expected_document']}",
                        f"- Expected page in Top-K: {query['matched_expected_page']}",
                    ]
                    if query["expected_sample_id"] is not None
                    else ["- Expected source: none — diagnostic case"]
                ),
                "",
                "| Rank | Score | Sample | Pages | Type | Content preview |",
                "| ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for hit in query["hits"]:
            preview = " ".join(hit["content"].split())[:180].replace("|", "\\|")
            lines.append(
                f"| {hit['rank']} | {hit['score']:.6f} | `{hit['sample_id']}` | "
                f"{hit['page_start']}-{hit['page_end']} | {hit['content_type']} | {preview} |"
            )
        lines.append("")
        lines.extend(["### Complete retrieved content", ""])
        for hit in query["hits"]:
            section_path = " > ".join(hit["section_path"]) or "(none)"
            lines.extend(
                [
                    f"#### Rank {hit['rank']}",
                    "",
                    f"- Score: `{hit['score']:.6f}`",
                    f"- Sample: `{hit['sample_id']}`",
                    f"- Pages: {hit['page_start']}-{hit['page_end']}",
                    f"- Content type: `{hit['content_type']}`",
                    f"- Section: {section_path}",
                    f"- Record ID: `{hit['record_id']}`",
                    f"- Chunk fingerprint: `{hit['chunk_fingerprint']}`",
                    "",
                    "##### Full content",
                    "",
                    *_as_markdown_blockquote(hit["content"]),
                    "",
                ]
            )
    return "\n".join(lines)


def _as_markdown_blockquote(content: str) -> list[str]:
    """Preserve every retrieved content line in a readable Markdown quote."""

    return [f"> {line}" if line else ">" for line in content.split("\n")]

"""JSON-compatible and human-readable previews of grounded PDF answers."""

from dataclasses import asdict, dataclass

from ai_learning_rag.pdf_generation import PdfAnswerStatus, PdfGroundedAnswer


@dataclass(frozen=True, slots=True)
class PdfGenerationPreviewCase:
    """One fixed question and its coarse expected grounded-answer outcome."""

    query_id: str
    query: str
    top_k: int
    expected_status: PdfAnswerStatus
    expected_sample_id: str | None = None
    expected_pages: tuple[int, ...] = ()
    expected_min_citations: int = 0

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be blank")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.expected_min_citations < 0:
            raise ValueError("expected_min_citations must not be negative")
        if self.expected_sample_id is None and self.expected_pages:
            raise ValueError("expected pages require an expected sample")
        if self.expected_sample_id is not None:
            if not self.expected_sample_id.strip():
                raise ValueError("expected_sample_id must not be blank")
            if not self.expected_pages or any(page < 1 for page in self.expected_pages):
                raise ValueError("expected_pages must contain positive page numbers")
        if tuple(sorted(set(self.expected_pages))) != self.expected_pages:
            raise ValueError("expected_pages must be sorted and unique")


def build_pdf_generation_case_preview(
    case: PdfGenerationPreviewCase,
    response: PdfGroundedAnswer,
    *,
    document_samples: dict[str, str],
    sample_sources: dict[str, str],
) -> dict[str, object]:
    """Build one complete audit payload from a validated grounded answer."""

    evidence = [
        {
            "marker": item.marker,
            "rank": item.rank,
            "score": item.score,
            "sample_id": document_samples.get(item.document_id, "unknown"),
            "source": sample_sources.get(
                document_samples.get(item.document_id, ""),
                "unknown",
            ),
            "index_id": item.index_id,
            "document_id": item.document_id,
            "generation_id": item.generation_id,
            "record_id": item.record_id,
            "chunk_fingerprint": item.chunk_fingerprint,
            "content_type": item.content_type,
            "section_path": list(item.section_path),
            "pages": list(item.pages),
            "content": item.content,
            "regions": [asdict(region) for region in item.regions],
        }
        for item in response.evidence
    ]
    citations = [
        {
            "marker": item.marker,
            "score": item.score,
            "sample_id": document_samples.get(item.document_id, "unknown"),
            "source": sample_sources.get(
                document_samples.get(item.document_id, ""),
                "unknown",
            ),
            "index_id": item.index_id,
            "document_id": item.document_id,
            "generation_id": item.generation_id,
            "record_id": item.record_id,
            "chunk_fingerprint": item.chunk_fingerprint,
            "section_path": list(item.section_path),
            "pages": list(item.pages),
            "regions": [asdict(region) for region in item.regions],
        }
        for item in response.citations
    ]
    matched_expected_source = _matched_expected_source(case, citations)
    return {
        "query_id": case.query_id,
        "query": case.query,
        "top_k": case.top_k,
        "expected_status": case.expected_status.value,
        "actual_status": response.status.value,
        "status_matches": response.status is case.expected_status,
        "expected_min_citations": case.expected_min_citations,
        "citation_count_matches": len(citations) >= case.expected_min_citations,
        "expected_sample_id": case.expected_sample_id,
        "expected_pages": list(case.expected_pages),
        "matched_expected_source": matched_expected_source,
        "answer": response.answer,
        "raw_generation_output": response.raw_generation_output,
        "evidence": evidence,
        "excluded_evidence": [asdict(item) for item in response.excluded_evidence],
        "citations": citations,
    }


def build_pdf_generation_markdown(payload: dict[str, object]) -> str:
    """Render a complete human-review report without claiming formal metrics."""

    lines = [
        "# Real PDF grounded-generation preview",
        "",
        "> This is a fixed human-review preview, not a formal retrieval or answer metric.",
        "",
        f"- Index: `{payload['index_id']}`",
        f"- Embedding model: `{payload['embedding_model_id']}`",
        f"- Generation provider: `{payload['generation_provider']}`",
        f"- Generation model: `{payload['generation_model_id']}`",
        "",
        "| Query | Expected | Actual | Evidence | Citations | Source match |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for query in payload["queries"]:  # type: ignore[union-attr]
        lines.append(
            f"| `{query['query_id']}` | `{query['expected_status']}` | "
            f"`{query['actual_status']}` | {len(query['evidence'])} | "
            f"{len(query['citations'])} | {query['matched_expected_source']} |"
        )
    lines.append("")

    for query in payload["queries"]:  # type: ignore[union-attr]
        lines.extend(
            [
                f"## {query['query_id']}",
                "",
                f"- Query: {query['query']}",
                f"- Expected status: `{query['expected_status']}`",
                f"- Actual status: `{query['actual_status']}`",
                f"- Status matches: {query['status_matches']}",
                f"- Minimum citations: {query['expected_min_citations']}",
                f"- Citation count matches: {query['citation_count_matches']}",
                f"- Expected sample: `{query['expected_sample_id']}`",
                f"- Expected pages: {query['expected_pages']}",
                f"- Expected source matched: {query['matched_expected_source']}",
                "",
                "### Answer",
                "",
                query["answer"],
                "",
                "### Citations",
                "",
            ]
        )
        if not query["citations"]:
            lines.extend(["(none)", ""])
        for citation in query["citations"]:
            lines.extend(
                [
                    f"#### [{citation['marker']}] `{citation['sample_id']}` pages "
                    f"{citation['pages']}",
                    "",
                    f"- Source: `{citation['source']}`",
                    f"- Record: `{citation['record_id']}`",
                    f"- Generation: `{citation['generation_id']}`",
                    f"- Score: `{citation['score']:.6f}`",
                    "",
                    "| Page | Block | Bbox | Method | Chunk range |",
                    "| ---: | --- | --- | --- | --- |",
                ]
            )
            for region in citation["regions"]:
                lines.append(
                    f"| {region['page_number']} | `{region['block_id']}` | "
                    f"`{region['bbox']}` | `{region['extraction_method']}` | "
                    f"{region['chunk_content_start']}-{region['chunk_content_end']} |"
                )
            lines.append("")

        lines.extend(["### Complete selected evidence", ""])
        if not query["evidence"]:
            lines.extend(["(none)", ""])
        for item in query["evidence"]:
            section_path = " > ".join(item["section_path"]) or "(none)"
            lines.extend(
                [
                    f"#### Evidence [{item['marker']}] / Rank {item['rank']}",
                    "",
                    f"- Sample: `{item['sample_id']}`",
                    f"- Source: `{item['source']}`",
                    f"- Pages: {item['pages']}",
                    f"- Section: {section_path}",
                    f"- Record: `{item['record_id']}`",
                    f"- Score: `{item['score']:.6f}`",
                    "",
                    *_as_markdown_blockquote(item["content"]),
                    "",
                ]
            )

        lines.extend(["### Excluded evidence", ""])
        if not query["excluded_evidence"]:
            lines.extend(["(none)", ""])
        for item in query["excluded_evidence"]:
            lines.append(
                f"- Rank {item['rank']} `{item['record_id']}`: `{item['reason']}` "
                f"({item['content_characters']} characters)"
            )
        lines.append("")
    return "\n".join(lines)


def _matched_expected_source(
    case: PdfGenerationPreviewCase,
    citations: list[dict[str, object]],
) -> bool | None:
    if case.expected_sample_id is None:
        return None
    return any(
        citation["sample_id"] == case.expected_sample_id
        and any(page in citation["pages"] for page in case.expected_pages)  # type: ignore[operator]
        for citation in citations
    )


def _as_markdown_blockquote(content: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in content.split("\n")]

"""Tests for human-readable real-PDF retrieval preview payloads."""

from hashlib import sha256

import pytest

from ai_learning_rag.document_index import SearchResult, VectorRecord
from ai_learning_rag.pdf_retrieval_preview import (
    RetrievalPreviewCase,
    build_chunk_catalog_entry,
    build_query_preview,
    build_retrieval_markdown,
)
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ChunkCandidate,
    ExtractionMethod,
    ParsedBlock,
    build_traceable_pdf_chunk,
)


def make_result(
    *,
    document_id: str,
    page: int,
    score: float,
    content: str = "Expected evidence with a | table separator.",
) -> SearchResult:
    """Build one fully traceable result without a database or model."""

    digest = sha256(f"{document_id}-{page}".encode()).hexdigest()
    return SearchResult(
        record=VectorRecord(
            record_id=f"rec_{digest}",
            index_id=f"idx_{'1' * 64}",
            generation_id=f"gen_{'2' * 64}",
            document_id=document_id,
            chunk_fingerprint=digest,
            chunk_index=0,
            content=content,
            content_type="body_text",
            section_path=("Section",),
            page_start=page,
            page_end=page,
            embedding_input_hash=sha256(b"input").hexdigest(),
            embedding_input_token_count=5,
            vector=(1.0, 0.0),
            provenance={"schema_version": 1, "source_regions": []},
        ),
        score=score,
    )


def test_preview_distinguishes_document_and_page_matches() -> None:
    """A correct document on the wrong page must remain visible as a partial match."""

    case = RetrievalPreviewCase(
        query_id="query-1",
        query="Where is the expected evidence?",
        expected_sample_id="sample-a",
        expected_pages=(4,),
        top_k=2,
    )
    payload = build_query_preview(
        case,
        [
            make_result(document_id="document-a", page=3, score=0.9),
            make_result(document_id="document-a", page=4, score=0.8),
        ],
        {"document-a": "sample-a"},
    )

    assert payload["matched_expected_document"] is True
    assert payload["matched_expected_page"] is True
    assert payload["hits"][0]["expected_document"] is True  # type: ignore[index]
    assert payload["hits"][0]["expected_page"] is False  # type: ignore[index]


def test_markdown_labels_preview_as_non_metric_human_review() -> None:
    """The small sample must not be misrepresented as formal retrieval evaluation."""

    case = RetrievalPreviewCase(
        query_id="query-1",
        query="Expected evidence?",
        expected_sample_id="sample-a",
        expected_pages=(4,),
        top_k=1,
    )
    query = build_query_preview(
        case,
        [make_result(document_id="document-a", page=4, score=1.0)],
        {"document-a": "sample-a"},
    )
    markdown = build_retrieval_markdown(
        {
            "index_id": f"idx_{'1' * 64}",
            "model_id": "fixture-model",
            "model_revision": "revision-1",
            "queries": [query],
        }
    )

    assert "not a Recall@K or MRR evaluation" in markdown
    assert "sample-a" in markdown
    assert "\\| table separator" in markdown


def test_diagnostic_preview_has_no_fake_expected_source() -> None:
    """Unanswerable and ambiguous cases must not pretend to have source gold."""

    case = RetrievalPreviewCase(
        query_id="diagnostic-1",
        query="What is not covered?",
        expected_sample_id=None,
        expected_pages=(),
        top_k=1,
        case_type="unanswerable",
    )
    query = build_query_preview(
        case,
        [make_result(document_id="document-a", page=4, score=1.0)],
        {"document-a": "sample-a"},
    )

    assert query["matched_expected_document"] is None
    assert query["matched_expected_page"] is None
    assert query["hits"][0]["expected_document"] is None  # type: ignore[index]
    markdown = build_retrieval_markdown(
        {
            "index_id": f"idx_{'1' * 64}",
            "model_id": "fixture-model",
            "model_revision": "revision-1",
            "queries": [query],
        }
    )
    assert "Case type: `unanswerable`" in markdown
    assert "Expected source: none — diagnostic case" in markdown


def test_preview_case_rejects_sources_for_diagnostics() -> None:
    """Diagnostic cases cannot accidentally inherit answerable expectations."""

    with pytest.raises(ValueError, match="must not declare an expected source"):
        RetrievalPreviewCase(
            query_id="diagnostic-1",
            query="Ambiguous question",
            expected_sample_id="sample-a",
            expected_pages=(1,),
            top_k=5,
            case_type="ambiguous",
        )


def test_markdown_preserves_complete_retrieved_content_after_summary() -> None:
    """The compact table may truncate, but the audit section must retain every line."""

    content = f"First evidence line.\n{'x' * 220}\nTAIL_MARKER"
    case = RetrievalPreviewCase(
        query_id="query-full-content",
        query="Show all evidence",
        expected_sample_id="sample-a",
        expected_pages=(4,),
        top_k=1,
    )
    query = build_query_preview(
        case,
        [
            make_result(
                document_id="document-a",
                page=4,
                score=1.0,
                content=content,
            )
        ],
        {"document-a": "sample-a"},
    )

    markdown = build_retrieval_markdown(
        {
            "index_id": f"idx_{'1' * 64}",
            "model_id": "fixture-model",
            "model_revision": "revision-1",
            "queries": [query],
        }
    )

    assert "### Complete retrieved content" in markdown
    assert "> First evidence line." in markdown
    assert f"> {'x' * 220}" in markdown
    assert "> TAIL_MARKER" in markdown


def test_chunk_catalog_preserves_every_field_needed_for_exhaustive_review() -> None:
    """Gold completeness checks need accepted chunks beyond the retrieved Top-K."""

    block = ParsedBlock(
        id="block-a",
        page_number=2,
        block_type=BlockType.PARAGRAPH,
        text="Complete catalog evidence",
        bbox=BoundingBox(1, 2, 3, 4),
        reading_order=0,
        extraction_method=ExtractionMethod.NATIVE,
    )
    chunk = build_traceable_pdf_chunk(
        ChunkCandidate.from_blocks(
            document_id="document-a",
            blocks=(block,),
            section_path=("Section",),
        ),
        chunk_index=3,
        chunking_profile="fixture-chunking-v1",
    )

    payload = build_chunk_catalog_entry(
        sample_id="sample-a",
        path="data/PDF/a.pdf",
        chunk=chunk,
    )

    assert payload == {
        "sample_id": "sample-a",
        "path": "data/PDF/a.pdf",
        "chunk_id": chunk.id,
        "document_id": "document-a",
        "chunk_fingerprint": chunk.source_fingerprint,
        "chunk_index": 3,
        "content_type": "body_text",
        "section_path": ["Section"],
        "page_start": 2,
        "page_end": 2,
        "source_block_ids": ["block-a"],
        "content": "Complete catalog evidence",
    }

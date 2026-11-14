"""Tests for auditable JSON and Markdown grounded-generation previews."""

from ai_learning_rag.pdf_generation import (
    PdfAnswerStatus,
    PdfCitation,
    PdfCitationRegion,
    PdfEvidence,
    PdfGroundedAnswer,
)
from ai_learning_rag.pdf_generation_preview import (
    PdfGenerationPreviewCase,
    build_pdf_generation_case_preview,
    build_pdf_generation_markdown,
)


def make_response() -> PdfGroundedAnswer:
    region = PdfCitationRegion(
        page_number=4,
        block_id="block-1",
        bbox=(10.0, 20.0, 110.0, 40.0),
        block_type="paragraph",
        extraction_method="rerun_ocr",
        reading_order=1,
        chunk_content_start=0,
        chunk_content_end=13,
        source_text_start=0,
        source_text_end=13,
        parent_picture_id=None,
        confidence=0.99,
        warnings=(),
        extraction_provenance_id="prov-1",
    )
    evidence = PdfEvidence(
        marker=1,
        rank=1,
        score=0.91,
        index_id=f"idx_{'1' * 64}",
        generation_id=f"gen_{'2' * 64}",
        document_id="pdf-document",
        record_id=f"rec_{'3' * 64}",
        chunk_fingerprint="4" * 64,
        content="Complete | evidence\nTAIL",
        content_type="body_text",
        section_path=("Section",),
        page_start=4,
        page_end=4,
        regions=(region,),
    )
    citation = PdfCitation(
        marker=1,
        score=evidence.score,
        index_id=evidence.index_id,
        generation_id=evidence.generation_id,
        document_id=evidence.document_id,
        record_id=evidence.record_id,
        chunk_fingerprint=evidence.chunk_fingerprint,
        section_path=evidence.section_path,
        pages=(4,),
        regions=(region,),
    )
    return PdfGroundedAnswer(
        query="Question",
        status=PdfAnswerStatus.ANSWERED,
        answer="Grounded answer [1].",
        citations=(citation,),
        evidence=(evidence,),
        excluded_evidence=(),
        raw_generation_output='{"status":"answered","answer":"Grounded answer [1]."}',
    )


def test_preview_payload_preserves_answer_evidence_and_page_regions() -> None:
    case = PdfGenerationPreviewCase(
        query_id="scan-case",
        query="Question",
        top_k=5,
        expected_status=PdfAnswerStatus.ANSWERED,
        expected_sample_id="scan-001",
        expected_pages=(4,),
    )

    payload = build_pdf_generation_case_preview(
        case,
        make_response(),
        document_samples={"pdf-document": "scan-001"},
        sample_sources={"scan-001": "data/PDF/scan.pdf"},
    )

    assert payload["status_matches"] is True
    assert payload["matched_expected_source"] is True
    assert payload["citations"][0]["regions"][0]["bbox"] == (  # type: ignore[index]
        10.0,
        20.0,
        110.0,
        40.0,
    )
    assert payload["evidence"][0]["content"] == "Complete | evidence\nTAIL"  # type: ignore[index]


def test_markdown_keeps_full_evidence_and_exact_citation_location() -> None:
    case = PdfGenerationPreviewCase(
        query_id="scan-case",
        query="Question",
        top_k=5,
        expected_status=PdfAnswerStatus.ANSWERED,
        expected_sample_id="scan-001",
        expected_pages=(4,),
    )
    query_payload = build_pdf_generation_case_preview(
        case,
        make_response(),
        document_samples={"pdf-document": "scan-001"},
        sample_sources={"scan-001": "data/PDF/scan.pdf"},
    )

    markdown = build_pdf_generation_markdown(
        {
            "index_id": f"idx_{'1' * 64}",
            "embedding_model_id": "fixture-embedder",
            "generation_provider": "fixture-provider",
            "generation_model_id": "fixture-generator",
            "queries": [query_payload],
        }
    )

    assert "not a formal retrieval or answer metric" in markdown
    assert "Generation provider: `fixture-provider`" in markdown
    assert "`block-1`" in markdown
    assert "`(10.0, 20.0, 110.0, 40.0)`" in markdown
    assert "> Complete | evidence" in markdown
    assert "> TAIL" in markdown

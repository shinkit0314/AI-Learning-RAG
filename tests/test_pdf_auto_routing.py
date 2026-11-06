"""Tests for page-range parser orchestration and fail-closed routing."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_learning_rag.pdf_auto_routing import (
    CachingPageRangeBackend,
    OcrReviewChoice,
    OcrReviewDecision,
    OcrReviewDecisions,
    ParseAttemptStatus,
    ParsedPageBatch,
    build_extraction_provenance,
    route_pdf_document,
)
from ai_learning_rag.pdf_chunking import PdfChunkingConfig, chunk_pdf_document
from ai_learning_rag.pdf_routing import (
    PageRoutingSignals,
    PdfPageRoute,
    PdfRoutingConfig,
)
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheStatus
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParseStatus,
)


def make_block(
    page_number: int,
    text: str,
    *,
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE,
) -> ParsedBlock:
    return ParsedBlock(
        id=f"block-{page_number}",
        page_number=page_number,
        block_type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(50, 50, 500, 100),
        reading_order=0,
        extraction_method=extraction_method,
    )


def make_page(
    page_number: int,
    text: str = "baseline text",
    *,
    status: ParseStatus = ParseStatus.PARSED,
    kind: PageKind = PageKind.NATIVE_TEXT,
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE,
) -> ParsedPage:
    blocks = (make_block(page_number, text, extraction_method=extraction_method),) if text else ()
    return ParsedPage(
        page_number=page_number,
        width=600,
        height=800,
        kind=kind,
        status=status,
        blocks=blocks,
        text_character_count=len(text.replace(" ", "")),
    )


def make_signals(page_number: int, **changes) -> PageRoutingSignals:
    values = {
        "page_number": page_number,
        "width": 600,
        "height": 800,
        "text_character_count": 100,
        "text_block_count": 5,
        "image_count": 0,
        "image_union_coverage": 0.0,
        "largest_image_coverage": 0.0,
        "drawing_count": 0,
        "unicode_anomaly_ratio": 0.0,
        "private_use_ratio": 0.0,
        "invalid_bbox_ratio": 0.0,
        "duplicate_glyph_ratio": 0.0,
        "latin_fragment_ratio": 0.0,
        "mixed_script_token_ratio": 0.0,
        "dominant_script": "latin",
        "column_count_candidate": 1,
        "column_left_block_count": 0,
        "column_right_block_count": 0,
        "column_gutter_ratio": 0.0,
        "column_vertical_overlap": 0.0,
    }
    values.update(changes)
    return PageRoutingSignals(**values)


def make_baseline(*pages: ParsedPage) -> ParsedDocument:
    return ParsedDocument(
        document_id="pdf-test",
        source="file:///test.pdf",
        file_sha256="a" * 64,
        parser_name="pymupdf",
        parser_version="1.28.2",
        pages=pages,
    )


@dataclass
class FakeBackend:
    texts: dict[int, str]
    fail_routes: tuple[PdfPageRoute, ...] = ()

    def __post_init__(self) -> None:
        self.calls: list[tuple[PdfPageRoute, tuple[int, int]]] = []

    def parse_range(
        self,
        source: Path,
        *,
        page_range: tuple[int, int],
        route: PdfPageRoute,
        file_sha256: str,
        config: PdfRoutingConfig,
    ) -> ParsedPageBatch:
        self.calls.append((route, page_range))
        if route in self.fail_routes:
            raise RuntimeError("controlled parser failure")
        method = (
            ExtractionMethod.RERUN_OCR
            if route in {PdfPageRoute.OCR, PdfPageRoute.OCR_VERIFICATION}
            else ExtractionMethod.LAYOUT
        )
        provenance = build_extraction_provenance(
            parser_name="fake-docling",
            parser_version="1.0",
            extraction_method=method,
            configuration_profile=f"route={route.value}",
            ocr_engine_name="fake-ocr" if method is ExtractionMethod.RERUN_OCR else None,
            ocr_engine_version="1.0" if method is ExtractionMethod.RERUN_OCR else None,
        )
        pages = tuple(
            make_page(
                page_number,
                self.texts[page_number],
                kind=(
                    PageKind.SCANNED_WITH_OCR
                    if method is ExtractionMethod.RERUN_OCR
                    else PageKind.COMPLEX_LAYOUT_TEXT
                ),
                extraction_method=method,
            )
            for page_number in range(page_range[0], page_range[1] + 1)
        )
        return ParsedPageBatch(pages=pages, provenance=provenance)

    def provenance_for_route(
        self,
        route: PdfPageRoute,
        config: PdfRoutingConfig,
    ):
        method = (
            ExtractionMethod.RERUN_OCR
            if route in {PdfPageRoute.OCR, PdfPageRoute.OCR_VERIFICATION}
            else ExtractionMethod.LAYOUT
        )
        return build_extraction_provenance(
            parser_name="fake-docling",
            parser_version="1.0",
            extraction_method=method,
            configuration_profile=f"route={route.value}",
            ocr_engine_name="fake-ocr" if method is ExtractionMethod.RERUN_OCR else None,
            ocr_engine_version="1.0" if method is ExtractionMethod.RERUN_OCR else None,
        )


def test_routes_contiguous_expensive_pages_and_merges_one_final_page_each() -> None:
    baseline = make_baseline(
        make_page(1, "simple native text"),
        make_page(2, "complex page two"),
        make_page(3, "complex page three"),
        make_page(
            4,
            "verified hidden text",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        ),
        make_page(5, "", status=ParseStatus.NEEDS_OCR, kind=PageKind.SCANNED_IMAGE),
    )
    signals = (
        make_signals(1),
        make_signals(
            2,
            column_count_candidate=2,
            column_left_block_count=3,
            column_right_block_count=3,
            column_gutter_ratio=0.15,
            column_vertical_overlap=0.8,
        ),
        make_signals(
            3,
            column_count_candidate=2,
            column_left_block_count=3,
            column_right_block_count=3,
            column_gutter_ratio=0.15,
            column_vertical_overlap=0.8,
        ),
        make_signals(
            4,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
        make_signals(
            5,
            text_character_count=0,
            text_block_count=0,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    backend = FakeBackend(
        texts={
            2: "docling page two",
            3: "docling page three",
            4: "verified hidden text",
            5: "ocr page five",
        }
    )

    result = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=backend,
    )

    assert backend.calls == [
        (PdfPageRoute.DOCLING, (2, 3)),
        (PdfPageRoute.OCR_VERIFICATION, (4, 4)),
        (PdfPageRoute.OCR, (5, 5)),
    ]
    assert [page.page_number for page in result.document.pages] == [1, 2, 3, 4, 5]
    assert all(page.status is ParseStatus.PARSED for page in result.document.pages)
    assert len(result.document.extraction_provenances) == 4
    assert result.document.pages[3].blocks[0].extraction_method is ExtractionMethod.EMBEDDED_OCR
    assert "EMBEDDED_OCR_VERIFIED" in result.document.pages[3].warnings


def test_hidden_ocr_conflict_is_failed_instead_of_guessing_a_winner() -> None:
    baseline = make_baseline(
        make_page(
            1,
            "属于数据系统",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    backend = FakeBackend(texts={1: "属千数据系统"})

    result = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=backend,
    )

    page = result.document.pages[0]
    assert page.status is ParseStatus.FAILED
    assert page.blocks == ()
    assert "OCR_RESULT_CONFLICT_REVIEW_REQUIRED" in page.warnings
    assert result.page_diagnostics[0].final_provenance_id is None
    assert result.page_diagnostics[0].ocr_conflict is not None


@pytest.mark.parametrize(
    ("choice", "expected_status", "expected_text", "expected_warning"),
    [
        (
            OcrReviewChoice.ACCEPT_EMBEDDED_OCR,
            ParseStatus.PARSED,
            "属于数据系统",
            "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR",
        ),
        (
            OcrReviewChoice.ACCEPT_RERUN_OCR,
            ParseStatus.PARSED,
            "属千数据系统",
            "MANUAL_REVIEW_ACCEPTED_RERUN_OCR",
        ),
        (
            OcrReviewChoice.REJECT_PAGE,
            ParseStatus.FAILED,
            None,
            "MANUAL_REVIEW_REJECTED_PAGE",
        ),
    ],
)
def test_human_review_resolves_each_conflict_explicitly(
    choice: OcrReviewChoice,
    expected_status: ParseStatus,
    expected_text: str | None,
    expected_warning: str,
) -> None:
    baseline = make_baseline(
        make_page(
            1,
            "属于数据系统",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    config = PdfRoutingConfig()
    backend = FakeBackend(texts={1: "属千数据系统"})
    initial = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=backend,
    )
    conflict = initial.page_diagnostics[0].ocr_conflict
    assert conflict is not None
    decisions = OcrReviewDecisions(
        document_sha256=baseline.file_sha256,
        routing_profile=config.routing_profile,
        decisions=(
            OcrReviewDecision(
                page_number=1,
                conflict_id=conflict.conflict_id,
                decision=choice,
                review_note="human compared the rendered source page",
            ),
        ),
    )

    reviewed = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=backend,
        ocr_review_decisions=decisions,
    )

    page = reviewed.document.pages[0]
    diagnostic = reviewed.page_diagnostics[0]
    assert page.status is expected_status
    assert expected_warning in page.warnings
    assert diagnostic.review_decision_id is not None
    if expected_text is None:
        assert page.blocks == ()
        assert diagnostic.final_provenance_id is None
    else:
        assert page.blocks[0].text == expected_text
        assert diagnostic.final_provenance_id == page.extraction_provenance_id
        provenance = reviewed.document.extraction_provenances[0]
        assert diagnostic.review_decision_id in provenance.configuration_profile
        chunks = chunk_pdf_document(reviewed.document, PdfChunkingConfig()).chunks
        assert chunks
        assert chunks[0].source_regions[0].extraction_provenance_id == provenance.id


def test_review_note_does_not_change_content_identity() -> None:
    baseline = make_baseline(
        make_page(
            1,
            "embedded candidate",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    config = PdfRoutingConfig()
    backend = FakeBackend(texts={1: "rerun candidate"})
    initial = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=backend,
    )
    conflict = initial.page_diagnostics[0].ocr_conflict
    assert conflict is not None

    reviewed_results = []
    for note in ("first human note", "corrected spelling in the human note"):
        reviewed_results.append(
            route_pdf_document(
                "test.pdf",
                baseline,
                signals,
                config,
                backend=backend,
                ocr_review_decisions=OcrReviewDecisions(
                    document_sha256=baseline.file_sha256,
                    routing_profile=config.routing_profile,
                    decisions=(
                        OcrReviewDecision(
                            page_number=1,
                            conflict_id=conflict.conflict_id,
                            decision=OcrReviewChoice.ACCEPT_RERUN_OCR,
                            review_note=note,
                        ),
                    ),
                ),
            )
        )

    assert reviewed_results[0].document == reviewed_results[1].document
    assert (
        reviewed_results[0].page_diagnostics[0].review_decision_id
        == reviewed_results[1].page_diagnostics[0].review_decision_id
    )


def test_review_decision_rejects_stale_conflict_id() -> None:
    baseline = make_baseline(
        make_page(
            1,
            "embedded",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    config = PdfRoutingConfig()
    decisions = OcrReviewDecisions(
        document_sha256=baseline.file_sha256,
        routing_profile=config.routing_profile,
        decisions=(
            OcrReviewDecision(
                page_number=1,
                conflict_id="ocr-conflict-stale",
                decision=OcrReviewChoice.ACCEPT_RERUN_OCR,
                review_note="stale decision",
            ),
        ),
    )

    with pytest.raises(ValueError, match="conflict_id does not match"):
        route_pdf_document(
            "test.pdf",
            baseline,
            signals,
            config,
            backend=FakeBackend(texts={1: "rerun"}),
            ocr_review_decisions=decisions,
        )


@pytest.mark.parametrize(
    ("document_sha256", "routing_profile", "error_match"),
    [
        ("b" * 64, PdfRoutingConfig().routing_profile, "document_sha256"),
        ("a" * 64, "stale-routing-profile", "routing_profile"),
    ],
)
def test_review_file_rejects_a_different_document_or_routing_profile(
    document_sha256: str,
    routing_profile: str,
    error_match: str,
) -> None:
    baseline = make_baseline(make_page(1, "simple page"))
    decisions = OcrReviewDecisions(
        document_sha256=document_sha256,
        routing_profile=routing_profile,
        decisions=(),
    )

    with pytest.raises(ValueError, match=error_match):
        route_pdf_document(
            "test.pdf",
            baseline,
            (make_signals(1),),
            PdfRoutingConfig(),
            backend=FakeBackend(texts={}),
            ocr_review_decisions=decisions,
        )


def test_docling_failure_does_not_silently_downgrade_complex_page() -> None:
    baseline = make_baseline(make_page(1, "column text from pymupdf"))
    signals = (
        make_signals(
            1,
            column_count_candidate=2,
            column_left_block_count=3,
            column_right_block_count=3,
            column_gutter_ratio=0.15,
            column_vertical_overlap=0.8,
        ),
    )
    backend = FakeBackend(texts={}, fail_routes=(PdfPageRoute.DOCLING,))

    result = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=backend,
    )

    page = result.document.pages[0]
    assert page.status is ParseStatus.FAILED
    assert page.blocks == ()
    assert result.page_diagnostics[0].attempts[0].status is ParseAttemptStatus.FAILED
    assert all(
        attempt.route is not PdfPageRoute.PYMUPDF for attempt in result.page_diagnostics[0].attempts
    )


def test_broken_hidden_text_layer_can_be_replaced_by_usable_rerun_ocr() -> None:
    baseline = make_baseline(
        make_page(
            1,
            "����",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
            unicode_anomaly_ratio=1.0,
        ),
    )
    backend = FakeBackend(texts={1: "可靠的重新识别文字"})

    result = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=backend,
    )

    page = result.document.pages[0]
    assert page.status is ParseStatus.PARSED
    assert page.blocks[0].text == "可靠的重新识别文字"
    assert page.blocks[0].extraction_method is ExtractionMethod.RERUN_OCR
    assert "RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER" in page.warnings


def test_page_cache_hit_preserves_output_without_recalling_backend(tmp_path) -> None:
    baseline = make_baseline(make_page(1, "complex baseline"))
    signals = (
        make_signals(
            1,
            column_count_candidate=2,
            column_left_block_count=3,
            column_right_block_count=3,
            column_gutter_ratio=0.15,
            column_vertical_overlap=0.8,
        ),
    )
    delegate = FakeBackend(texts={1: "cached docling result"})
    cache = PdfRoutingCache(tmp_path)
    backend = CachingPageRangeBackend(delegate, cache)

    first = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=backend,
    )
    second_backend = CachingPageRangeBackend(delegate, cache)
    second = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=second_backend,
    )

    assert len(delegate.calls) == 1
    assert first.document == second.document
    assert any(event.status is RoutingCacheStatus.HIT for event in second_backend.events)


def test_reviewed_rerun_ocr_reuses_the_cached_candidate(tmp_path) -> None:
    baseline = make_baseline(
        make_page(
            1,
            "embedded candidate",
            kind=PageKind.SCANNED_WITH_OCR,
            extraction_method=ExtractionMethod.EMBEDDED_OCR,
        )
    )
    signals = (
        make_signals(
            1,
            image_count=1,
            image_union_coverage=0.95,
            largest_image_coverage=0.95,
        ),
    )
    config = PdfRoutingConfig()
    delegate = FakeBackend(texts={1: "rerun candidate"})
    cache = PdfRoutingCache(tmp_path)
    initial_backend = CachingPageRangeBackend(delegate, cache)
    initial = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=initial_backend,
    )
    conflict = initial.page_diagnostics[0].ocr_conflict
    assert conflict is not None
    decisions = OcrReviewDecisions(
        document_sha256=baseline.file_sha256,
        routing_profile=config.routing_profile,
        decisions=(
            OcrReviewDecision(
                page_number=1,
                conflict_id=conflict.conflict_id,
                decision=OcrReviewChoice.ACCEPT_RERUN_OCR,
                review_note="rerun matches the rendered source",
            ),
        ),
    )

    reviewed_backend = CachingPageRangeBackend(delegate, cache)
    first_reviewed = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=reviewed_backend,
        ocr_review_decisions=decisions,
    )
    repeated_backend = CachingPageRangeBackend(delegate, cache)
    repeated = route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        config,
        backend=repeated_backend,
        ocr_review_decisions=decisions,
    )

    assert len(delegate.calls) == 1
    assert first_reviewed.document == repeated.document
    assert first_reviewed.page_diagnostics == repeated.page_diagnostics
    assert any(event.status is RoutingCacheStatus.HIT for event in reviewed_backend.events)

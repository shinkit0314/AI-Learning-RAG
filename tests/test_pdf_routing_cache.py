"""Tests for content-addressed routing cache behavior."""

from dataclasses import replace

from ai_learning_rag.pdf_auto_routing import build_extraction_provenance
from ai_learning_rag.pdf_routing import (
    PageRoutingSignals,
    PdfPageRoute,
    PdfRoutingConfig,
    build_route_plan,
)
from ai_learning_rag.pdf_routing_cache import (
    PdfRoutingCache,
    RoutingCacheStatus,
)
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedPage,
    ParseStatus,
)


def make_signals(page_number: int) -> PageRoutingSignals:
    return PageRoutingSignals(
        page_number=page_number,
        width=600,
        height=800,
        text_character_count=100,
        text_block_count=5,
        image_count=0,
        image_union_coverage=0.0,
        largest_image_coverage=0.0,
        drawing_count=0,
        unicode_anomaly_ratio=0.0,
        private_use_ratio=0.0,
        invalid_bbox_ratio=0.0,
        duplicate_glyph_ratio=0.0,
        latin_fragment_ratio=0.0,
        mixed_script_token_ratio=0.0,
        dominant_script="latin",
        column_count_candidate=1,
        column_left_block_count=0,
        column_right_block_count=0,
        column_gutter_ratio=0.0,
        column_vertical_overlap=0.0,
    )


def make_page(page_number: int, text: str) -> ParsedPage:
    block = ParsedBlock(
        id=f"block-{page_number}",
        page_number=page_number,
        block_type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(50, 50, 500, 100),
        reading_order=0,
        extraction_method=ExtractionMethod.LAYOUT,
    )
    return ParsedPage(
        page_number=page_number,
        width=600,
        height=800,
        kind=PageKind.COMPLEX_LAYOUT_TEXT,
        status=ParseStatus.PARSED,
        blocks=(block,),
        text_character_count=len(text.replace(" ", "")),
    )


def test_signal_and_decision_cache_round_trip_and_profile_invalidation(tmp_path) -> None:
    cache = PdfRoutingCache(tmp_path)
    signals = (make_signals(1),)
    config = PdfRoutingConfig()
    decisions = build_route_plan(signals, config)

    assert (
        cache.save_signals(
            signals,
            file_sha256="a" * 64,
            extractor_version=config.signal_extractor_version,
            pymupdf_version="1.28.2",
        ).status
        is RoutingCacheStatus.WRITTEN
    )
    loaded_signals, signal_event = cache.load_signals(
        file_sha256="a" * 64,
        extractor_version=config.signal_extractor_version,
        pymupdf_version="1.28.2",
    )
    assert signal_event.status is RoutingCacheStatus.HIT
    assert loaded_signals == signals

    cache.save_decisions(decisions, signals, routing_profile=config.routing_profile)
    loaded_decisions, decision_event = cache.load_decisions(
        signals,
        routing_profile=config.routing_profile,
    )
    assert decision_event.status is RoutingCacheStatus.HIT
    assert loaded_decisions == decisions

    changed = PdfRoutingConfig(min_useful_characters=30)
    missing, changed_event = cache.load_decisions(
        signals,
        routing_profile=changed.routing_profile,
    )
    assert missing is None
    assert changed_event.status is RoutingCacheStatus.MISS


def test_only_accepted_pages_are_cached_and_corruption_is_visible(tmp_path) -> None:
    cache = PdfRoutingCache(tmp_path)
    provenance = build_extraction_provenance(
        parser_name="fake",
        parser_version="1.0",
        extraction_method=ExtractionMethod.LAYOUT,
        configuration_profile="layout-v1",
    )
    page = make_page(1, "cached page text")
    page = replace(
        page,
        extraction_provenance_id=provenance.id,
        blocks=tuple(
            replace(block, extraction_provenance_id=provenance.id) for block in page.blocks
        ),
    )
    event = cache.save_page(
        page,
        file_sha256="a" * 64,
        route=PdfPageRoute.DOCLING,
        routing_profile="route-v1",
        provenance=provenance,
    )
    assert event.status is RoutingCacheStatus.WRITTEN

    loaded, loaded_event = cache.load_page(
        file_sha256="a" * 64,
        page_number=1,
        route=PdfPageRoute.DOCLING,
        routing_profile="route-v1",
        provenance=provenance,
    )
    assert loaded_event.status is RoutingCacheStatus.HIT
    assert loaded == page

    cache_path = tmp_path / "pages" / f"{event.key}.json"
    cache_path.write_text("not-json", encoding="utf-8")
    corrupted, corrupt_event = cache.load_page(
        file_sha256="a" * 64,
        page_number=1,
        route=PdfPageRoute.DOCLING,
        routing_profile="route-v1",
        provenance=provenance,
    )
    assert corrupted is None
    assert corrupt_event.status is RoutingCacheStatus.CORRUPT

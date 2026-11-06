"""Tests for deterministic PDF page signals and routing policy."""

from dataclasses import replace

import pytest

from ai_learning_rag.pdf_routing import (
    LayoutComplexity,
    PageRoutingSignals,
    PdfPageRoute,
    PdfRoutingConfig,
    TextQualityStatus,
    build_route_plan,
)
from ai_learning_rag.pdf_schema import PageKind


def make_signals(**changes) -> PageRoutingSignals:
    base = PageRoutingSignals(
        page_number=1,
        width=600,
        height=800,
        text_character_count=200,
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
    return replace(base, **changes)


def test_routing_profile_is_stable_and_changes_with_behavior() -> None:
    config = PdfRoutingConfig()

    assert config.routing_profile == PdfRoutingConfig().routing_profile
    assert replace(config, min_useful_characters=21).routing_profile != config.routing_profile
    assert (
        replace(config, mixed_script_affects_text_quality=True).routing_profile
        != config.routing_profile
    )


def test_routing_config_can_be_recalibrated_from_json(tmp_path) -> None:
    config_path = tmp_path / "candidate.json"
    config_path.write_text('{"min_useful_characters": 35}', encoding="utf-8")

    config = PdfRoutingConfig.from_json(config_path)

    assert config.min_useful_characters == 35
    assert config.strategy_version == "pdf-page-routing-v1"

    config_path.write_text('{"unknown_threshold": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown routing configuration fields"):
        PdfRoutingConfig.from_json(config_path)


@pytest.mark.parametrize(
    ("character_count", "expected_quality"),
    [
        (20, TextQualityStatus.USABLE),
        (19, TextQualityStatus.SUSPICIOUS),
        (0, TextQualityStatus.UNUSABLE),
    ],
)
def test_text_character_threshold_has_explicit_boundary(
    character_count: int,
    expected_quality: TextQualityStatus,
) -> None:
    decision = build_route_plan((make_signals(text_character_count=character_count),))[0]

    assert decision.text_quality is expected_quality


def test_simple_native_page_uses_pymupdf_fast_path() -> None:
    decision = build_route_plan((make_signals(),))[0]

    assert decision.source_kind is PageKind.NATIVE_TEXT
    assert decision.layout_complexity is LayoutComplexity.SIMPLE
    assert decision.selected_route is PdfPageRoute.PYMUPDF
    assert decision.reason_codes == ("NATIVE_TEXT_USABLE_SIMPLE_LAYOUT",)


def test_full_page_image_routes_empty_and_hidden_text_differently() -> None:
    empty_scan = make_signals(
        text_character_count=0,
        text_block_count=0,
        image_count=1,
        image_union_coverage=0.95,
        largest_image_coverage=0.95,
    )
    hidden_ocr = replace(empty_scan, page_number=2, text_character_count=200, text_block_count=4)

    decisions = build_route_plan((empty_scan, hidden_ocr))

    assert decisions[0].source_kind is PageKind.SCANNED_IMAGE
    assert decisions[0].selected_route is PdfPageRoute.OCR
    assert decisions[1].source_kind is PageKind.SCANNED_WITH_OCR
    assert decisions[1].selected_route is PdfPageRoute.OCR_VERIFICATION


def test_damaged_text_layer_routes_to_ocr() -> None:
    decision = build_route_plan((make_signals(unicode_anomaly_ratio=0.10),))[0]

    assert decision.text_quality is TextQualityStatus.UNUSABLE
    assert decision.source_kind is PageKind.DAMAGED_TEXT_LAYER
    assert decision.selected_route is PdfPageRoute.OCR


def test_mixed_script_ratio_is_diagnostic_only_by_default() -> None:
    signals = make_signals(mixed_script_token_ratio=0.50)

    decision = build_route_plan((signals,))[0]
    mixed_script = next(
        signal for signal in decision.signal_results if signal.name == "mixed_script_token_ratio"
    )

    assert mixed_script.status.value == "fail"
    assert not mixed_script.affects_text_quality
    assert decision.text_quality is TextQualityStatus.USABLE
    assert decision.source_kind is PageKind.NATIVE_TEXT
    assert decision.selected_route is PdfPageRoute.PYMUPDF


def test_mixed_script_quality_gate_can_be_enabled_by_configuration() -> None:
    signals = make_signals(mixed_script_token_ratio=0.50)
    config = PdfRoutingConfig(mixed_script_affects_text_quality=True)

    decision = build_route_plan((signals,), config)[0]

    assert decision.text_quality is TextQualityStatus.UNUSABLE
    assert decision.source_kind is PageKind.DAMAGED_TEXT_LAYER
    assert decision.selected_route is PdfPageRoute.OCR


def test_multicolumn_and_weak_signals_upgrade_to_docling() -> None:
    strong = make_signals(
        column_count_candidate=2,
        column_left_block_count=4,
        column_right_block_count=4,
        column_gutter_ratio=0.15,
        column_vertical_overlap=0.8,
    )
    weak = replace(
        make_signals(),
        page_number=2,
        text_block_count=25,
        drawing_count=15,
    )

    decisions = build_route_plan((strong, weak))

    assert all(decision.layout_complexity is LayoutComplexity.COMPLEX for decision in decisions)
    assert all(decision.selected_route is PdfPageRoute.DOCLING for decision in decisions)


def test_one_weak_layout_signal_is_uncertain_and_still_uses_docling() -> None:
    decision = build_route_plan((make_signals(text_block_count=25),))[0]

    assert decision.layout_complexity is LayoutComplexity.UNCERTAIN
    assert decision.selected_route is PdfPageRoute.DOCLING


def test_suspicious_non_scan_text_uses_docling_without_unnecessary_ocr() -> None:
    decision = build_route_plan((make_signals(text_character_count=19),))[0]

    assert decision.source_kind is PageKind.MIXED
    assert decision.selected_route is PdfPageRoute.DOCLING
    assert decision.reason_codes == ("MIXED_PAGE_REQUIRES_DOCLING",)


def test_blank_page_needs_no_parser_route() -> None:
    decision = build_route_plan((make_signals(text_character_count=0, text_block_count=0),))[0]

    assert decision.source_kind is PageKind.BLANK
    assert decision.selected_route is PdfPageRoute.NONE


def test_route_plan_requires_contiguous_page_signals() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_route_plan((make_signals(page_number=2),))


def test_obviously_unsupported_ocr_script_fails_before_model_inference() -> None:
    decision = build_route_plan(
        (
            make_signals(
                image_count=1,
                image_union_coverage=0.95,
                largest_image_coverage=0.95,
                dominant_script="kana",
            ),
        ),
        PdfRoutingConfig(ocr_language="ch"),
    )[0]

    assert decision.selected_route is PdfPageRoute.FAILED
    assert decision.reason_codes == ("OCR_LANGUAGE_UNSUPPORTED",)

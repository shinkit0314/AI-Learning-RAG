"""Tests for readable routing previews without model inference."""

from test_pdf_auto_routing import make_baseline, make_page, make_signals

from ai_learning_rag.pdf_routing import PdfRoutingConfig, build_route_plan
from ai_learning_rag.pdf_routing_preview import build_plan_markdown, build_plan_payload


def test_diagnostics_preview_exposes_route_reasons_and_raw_signals() -> None:
    baseline = make_baseline(make_page(1, "simple text"))
    signals = (make_signals(1),)
    config = PdfRoutingConfig()
    payload = build_plan_payload(
        "sample.pdf",
        baseline,
        signals,
        build_route_plan(signals, config),
        config,
    )

    markdown = build_plan_markdown(payload)

    assert payload["mode"] == "diagnostics_only"
    assert "NATIVE_TEXT_USABLE_SIMPLE_LAYOUT" in markdown
    assert "text_character_count" in markdown
    assert config.routing_profile in markdown

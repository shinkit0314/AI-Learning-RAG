"""Tests for serializable human OCR review evidence and decisions."""

import json

import pytest
from test_pdf_auto_routing import FakeBackend, make_baseline, make_page, make_signals

from ai_learning_rag.pdf_auto_routing import route_pdf_document
from ai_learning_rag.pdf_ocr_review import (
    DECISIONS_SCHEMA,
    build_ocr_review_decisions_template,
    build_ocr_review_evidence_payload,
    build_ocr_review_markdown,
    load_ocr_review_decisions,
)
from ai_learning_rag.pdf_routing import PdfRoutingConfig
from ai_learning_rag.pdf_schema import ExtractionMethod, PageKind


def make_conflict_result():
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
    return route_pdf_document(
        "test.pdf",
        baseline,
        signals,
        PdfRoutingConfig(),
        backend=FakeBackend(texts={1: "属千数据系统"}),
    )


def test_review_artifacts_separate_immutable_evidence_and_editable_decisions() -> None:
    result = make_conflict_result()

    evidence = build_ocr_review_evidence_payload(result)
    template = build_ocr_review_decisions_template(result)
    markdown = build_ocr_review_markdown(evidence)

    assert evidence["conflicts"][0]["embedded_ocr"]["normalized_text"] == "属于数据系统"
    assert evidence["conflicts"][0]["rerun_ocr"]["normalized_text"] == "属千数据系统"
    assert template["schema"] == DECISIONS_SCHEMA
    assert template["decisions"][0]["decision"] is None
    assert template["decisions"][0]["conflict_id"] == evidence["conflicts"][0]["conflict_id"]
    assert "PDF 原有隐藏 OCR" in markdown
    assert "独立重跑 OCR" in markdown


def test_decisions_loader_ignores_unresolved_pages_and_loads_explicit_choice(tmp_path) -> None:
    result = make_conflict_result()
    payload = build_ocr_review_decisions_template(result)
    payload["decisions"].append(
        {
            "page_number": 2,
            "conflict_id": "ocr-conflict-unresolved",
            "decision": None,
            "review_note": "",
        }
    )
    payload["decisions"][0]["decision"] = "accept_rerun_ocr"
    payload["decisions"][0]["review_note"] = "new OCR matches the rendered page"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    decisions = load_ocr_review_decisions(path)

    assert len(decisions.decisions) == 1
    assert decisions.decisions[0].decision.value == "accept_rerun_ocr"


def test_decisions_loader_rejects_invalid_choice_and_blank_note(tmp_path) -> None:
    result = make_conflict_result()
    payload = build_ocr_review_decisions_template(result)
    path = tmp_path / "review.json"

    payload["decisions"][0]["decision"] = "accept_everything"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be one of"):
        load_ocr_review_decisions(path)

    payload["decisions"][0]["decision"] = "accept_embedded_ocr"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-blank review_note"):
        load_ocr_review_decisions(path)

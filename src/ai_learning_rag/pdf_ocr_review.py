"""Serializable human review contract for hidden-OCR conflicts."""

from __future__ import annotations

import json
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ai_learning_rag.pdf_auto_routing import (
    OcrReviewChoice,
    OcrReviewDecision,
    OcrReviewDecisions,
    PdfAutoRoutingResult,
)

EVIDENCE_SCHEMA = "pdf-ocr-review-evidence-v1"
DECISIONS_SCHEMA = "pdf-ocr-review-decisions-v1"


def build_ocr_review_evidence_payload(result: PdfAutoRoutingResult) -> dict[str, Any]:
    """Build immutable evidence for every unresolved or manually reviewed conflict."""
    conflicts = []
    for diagnostic in result.page_diagnostics:
        conflict = diagnostic.ocr_conflict
        if conflict is None:
            continue
        embedded = asdict(conflict.embedded_ocr)
        rerun = asdict(conflict.rerun_ocr)
        conflicts.append(
            {
                "page_number": conflict.page_number,
                "conflict_id": conflict.conflict_id,
                "embedded_ocr": embedded,
                "rerun_ocr": rerun,
                "ordered_similarity": round(
                    SequenceMatcher(
                        None,
                        conflict.embedded_ocr.normalized_text,
                        conflict.rerun_ocr.normalized_text,
                    ).ratio(),
                    6,
                ),
                "final_status": diagnostic.final_status.value,
                "review_decision_id": diagnostic.review_decision_id,
            }
        )
    return {
        "schema": EVIDENCE_SCHEMA,
        "document": {
            "document_id": result.document.document_id,
            "source": result.document.source,
            "file_sha256": result.document.file_sha256,
        },
        "routing_profile": result.routing_profile,
        "conflicts": conflicts,
    }


def build_ocr_review_decisions_template(result: PdfAutoRoutingResult) -> dict[str, Any]:
    """Build the only review artifact intended for manual editing."""
    return {
        "schema": DECISIONS_SCHEMA,
        "document_sha256": result.document.file_sha256,
        "routing_profile": result.routing_profile,
        "decisions": [
            {
                "page_number": diagnostic.ocr_conflict.page_number,
                "conflict_id": diagnostic.ocr_conflict.conflict_id,
                "decision": None,
                "review_note": "",
            }
            for diagnostic in result.page_diagnostics
            if diagnostic.ocr_conflict is not None
        ],
    }


def load_ocr_review_decisions(path: str | Path) -> OcrReviewDecisions:
    """Load completed decisions while treating null choices as unresolved pages."""
    review_path = Path(path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OCR review decisions must be a JSON object")
    if payload.get("schema") != DECISIONS_SCHEMA:
        raise ValueError(f"OCR review decisions schema must be {DECISIONS_SCHEMA}")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("OCR review decisions must contain a decisions array")

    decisions = []
    for item in raw_decisions:
        if not isinstance(item, dict):
            raise ValueError("each OCR review decision must be a JSON object")
        raw_choice = item.get("decision")
        if raw_choice is None or raw_choice == "":
            continue
        try:
            choice = OcrReviewChoice(raw_choice)
        except ValueError as error:
            allowed = ", ".join(choice.value for choice in OcrReviewChoice)
            raise ValueError(f"OCR review decision must be one of: {allowed}") from error
        decisions.append(
            OcrReviewDecision(
                page_number=int(item["page_number"]),
                conflict_id=str(item["conflict_id"]),
                decision=choice,
                review_note=str(item.get("review_note", "")),
            )
        )
    return OcrReviewDecisions(
        document_sha256=str(payload.get("document_sha256", "")),
        routing_profile=str(payload.get("routing_profile", "")),
        decisions=tuple(decisions),
    )


def build_ocr_review_markdown(payload: dict[str, Any]) -> str:
    """Render full candidate text and hashes without suggesting an automatic winner."""
    document = payload["document"]
    lines = [
        "# PDF OCR 人工复核",
        "",
        f"- Document: `{document['document_id']}`",
        f"- File SHA-256: `{document['file_sha256']}`",
        f"- Routing profile: `{payload['routing_profile']}`",
        f"- Conflict pages: `{len(payload['conflicts'])}`",
        "",
        "只修改同目录下的 `.ocr-decisions.json`。不要修改本证据文件或候选文字哈希。",
    ]
    for conflict in payload["conflicts"]:
        embedded = conflict["embedded_ocr"]
        rerun = conflict["rerun_ocr"]
        lines.extend(
            [
                "",
                f"## Page {conflict['page_number']}",
                "",
                f"- Conflict ID: `{conflict['conflict_id']}`",
                f"- Ordered similarity: `{conflict['ordered_similarity']}`",
                f"- Final status: `{conflict['final_status']}`",
                f"- Review decision ID: `{conflict['review_decision_id'] or 'unresolved'}`",
                "",
                "### PDF 原有隐藏 OCR",
                "",
                f"- Text SHA-256: `{embedded['text_sha256']}`",
                f"- Parser: `{embedded['provenance']['parser_name']} "
                f"{embedded['provenance']['parser_version']}`",
                "",
                "~~~text",
                embedded["normalized_text"],
                "~~~",
                "",
                "### 独立重跑 OCR",
                "",
                f"- Text SHA-256: `{rerun['text_sha256']}`",
                f"- Parser: `{rerun['provenance']['parser_name']} "
                f"{rerun['provenance']['parser_version']}`",
                "",
                "~~~text",
                rerun["normalized_text"],
                "~~~",
            ]
        )
    return "\n".join(lines) + "\n"

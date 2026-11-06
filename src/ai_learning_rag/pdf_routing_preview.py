"""Human-readable JSON and Markdown views of PDF page routing evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_learning_rag.pdf_auto_routing import PdfAutoRoutingResult
from ai_learning_rag.pdf_routing import (
    PageRouteDecision,
    PageRoutingSignals,
    PdfRoutingConfig,
)
from ai_learning_rag.pdf_schema import ParsedDocument


def build_plan_payload(
    source: str | Path,
    baseline: ParsedDocument,
    signals: tuple[PageRoutingSignals, ...],
    decisions: tuple[PageRouteDecision, ...],
    config: PdfRoutingConfig,
) -> dict[str, Any]:
    """Build a diagnostics-only payload that never invokes Docling or OCR."""
    return {
        "mode": "diagnostics_only",
        "source": str(Path(source)),
        "document": {
            "document_id": baseline.document_id,
            "file_sha256": baseline.file_sha256,
            "page_count": baseline.page_count,
        },
        "routing_profile": config.routing_profile,
        "routing_config": config.profile_payload,
        "pages": [
            {
                "signals": asdict(page_signals),
                "decision": asdict(decision),
            }
            for page_signals, decision in zip(signals, decisions, strict=True)
        ],
    }


def build_result_payload(result: PdfAutoRoutingResult) -> dict[str, Any]:
    """Serialize the complete routed result without parser-private objects."""
    return {
        "mode": "full_routing",
        "routing_profile": result.routing_profile,
        "document": asdict(result.document),
        "pages": [
            {
                "signals": asdict(signals),
                "routing": asdict(diagnostic),
            }
            for signals, diagnostic in zip(
                result.page_signals,
                result.page_diagnostics,
                strict=True,
            )
        ],
        "cache_events": [asdict(event) for event in result.cache_events],
    }


def build_plan_markdown(payload: dict[str, Any]) -> str:
    """Render a diagnostics-only routing plan for manual threshold review."""
    lines = [
        "# PDF Page Routing Preview",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Source: `{payload['source']}`",
        f"- File SHA-256: `{payload['document']['file_sha256']}`",
        f"- Routing profile: `{payload['routing_profile']}`",
        "",
        "| Page | Source kind | Text quality | Layout | Route | Reasons |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for page in payload["pages"]:
        decision = page["decision"]
        lines.append(
            "| {page} | {kind} | {quality} | {layout} | {route} | {reasons} |".format(
                page=decision["page_number"],
                kind=decision["source_kind"],
                quality=decision["text_quality"],
                layout=decision["layout_complexity"],
                route=decision["selected_route"],
                reasons=", ".join(decision["reason_codes"]),
            )
        )
    for page in payload["pages"]:
        page_number = page["decision"]["page_number"]
        lines.extend(["", f"## Page {page_number}", "", "### Raw signals", ""])
        for name, value in page["signals"].items():
            if name not in {"page_number", "width", "height"}:
                lines.append(f"- `{name}`: `{value}`")
        lines.extend(["", "### Evaluated quality signals", ""])
        for signal in page["decision"]["signal_results"]:
            role = "quality gate" if signal["affects_text_quality"] else "diagnostic only"
            lines.append(
                f"- `{signal['name']}` = `{signal['value']}` → "
                f"**{signal['status']}** ({role}; {signal['reason']})"
            )
    return "\n".join(lines) + "\n"


def build_result_markdown(payload: dict[str, Any]) -> str:
    """Render final routes, parser attempts, provenance, warnings, and cache events."""
    document = payload["document"]
    lines = [
        "# PDF Automatic Routing Result",
        "",
        f"- Document: `{document['document_id']}`",
        f"- File SHA-256: `{document['file_sha256']}`",
        f"- Routing profile: `{payload['routing_profile']}`",
        "",
        "| Page | Route | Final status | Final provenance | Reasons |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for page in payload["pages"]:
        routing = page["routing"]
        decision = routing["decision"]
        lines.append(
            "| {page} | {route} | {status} | {provenance} | {reasons} |".format(
                page=decision["page_number"],
                route=decision["selected_route"],
                status=routing["final_status"],
                provenance=routing["final_provenance_id"] or "—",
                reasons=", ".join(decision["reason_codes"]),
            )
        )
    pages_by_number = {page["page_number"]: page for page in document["pages"]}
    for page in payload["pages"]:
        routing = page["routing"]
        decision = routing["decision"]
        page_number = decision["page_number"]
        final_page = pages_by_number[page_number]
        lines.extend(
            [
                "",
                f"## Page {page_number}",
                "",
                f"- Source classification: `{decision['source_kind']}`",
                f"- Text quality: `{decision['text_quality']}`",
                f"- Layout complexity: `{decision['layout_complexity']}`",
                f"- Final status: `{routing['final_status']}`",
                f"- Warnings: `{', '.join(final_page['warnings']) or 'none'}`",
                "",
                "### Attempts",
                "",
            ]
        )
        conflict = routing.get("ocr_conflict")
        if conflict is not None:
            lines.extend(
                [
                    f"- OCR conflict: `{conflict['conflict_id']}`",
                    f"- Review decision: `{routing['review_decision_id'] or 'unresolved'}`",
                    "",
                ]
            )
        if not routing["attempts"]:
            lines.append("- No parser required.")
        for attempt in routing["attempts"]:
            provenance = attempt["provenance"]
            parser = provenance["parser_name"] if provenance else "unavailable"
            parser_version = provenance["parser_version"] if provenance else "unavailable"
            lines.append(
                f"- `{attempt['route']}` → **{attempt['status']}**, "
                f"parser `{parser} {parser_version}`, "
                f"failure `{attempt['failure_code'] or 'none'}`"
            )
        lines.extend(["", "### Signal decisions", ""])
        for signal in decision["signal_results"]:
            role = "quality gate" if signal["affects_text_quality"] else "diagnostic only"
            lines.append(
                f"- `{signal['name']}` = `{signal['value']}` → **{signal['status']}** ({role})"
            )
    if payload["cache_events"]:
        lines.extend(["", "## Cache events", ""])
        for event in payload["cache_events"]:
            lines.append(
                f"- `{event['stage']}` `{event['status']}` key `{event['key'][:12]}` "
                f"{event['message']}"
            )
    return "\n".join(lines) + "\n"

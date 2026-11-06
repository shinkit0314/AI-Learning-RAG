"""Generate or apply auditable per-page OCR review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.pdf_auto_routing import parse_pdf_auto
from ai_learning_rag.pdf_ocr_review import (
    build_ocr_review_decisions_template,
    build_ocr_review_evidence_payload,
    build_ocr_review_markdown,
    load_ocr_review_decisions,
)
from ai_learning_rag.pdf_routing import PdfRoutingConfig
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheMode
from ai_learning_rag.pdf_routing_preview import (
    build_result_markdown,
    build_result_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "apply"))
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pdf-routing-v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-routing-preview"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/pdf-routing-cache"),
    )
    parser.add_argument(
        "--cache-mode",
        choices=[mode.value for mode in RoutingCacheMode],
        default=RoutingCacheMode.READ_WRITE.value,
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        help="editable decisions JSON; defaults beside the generated evidence",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.decisions or args.output_dir / f"{args.pdf.stem}.ocr-decisions.json"
    if args.action == "apply" and not decisions_path.is_file():
        parser.error(f"decisions file does not exist: {decisions_path}")
    review_decisions = load_ocr_review_decisions(decisions_path) if args.action == "apply" else None
    config = PdfRoutingConfig.from_json(args.config)
    cache = PdfRoutingCache(args.cache_dir, RoutingCacheMode(args.cache_mode))
    result = parse_pdf_auto(
        args.pdf,
        config,
        cache=cache,
        ocr_review_decisions=review_decisions,
    )

    routing_payload = build_result_payload(result)
    routing_json_path = args.output_dir / f"{args.pdf.stem}.routing.json"
    routing_markdown_path = args.output_dir / f"{args.pdf.stem}.routing.md"
    routing_json_path.write_text(
        json.dumps(routing_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    routing_markdown_path.write_text(
        build_result_markdown(routing_payload),
        encoding="utf-8",
    )

    evidence = build_ocr_review_evidence_payload(result)
    evidence_json_path = args.output_dir / f"{args.pdf.stem}.ocr-review.json"
    evidence_markdown_path = args.output_dir / f"{args.pdf.stem}.ocr-review.md"
    evidence_json_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evidence_markdown_path.write_text(
        build_ocr_review_markdown(evidence),
        encoding="utf-8",
    )

    if args.action == "generate":
        if decisions_path.exists():
            print(f"preserved existing editable decisions: {decisions_path}")
        else:
            decisions_path.write_text(
                json.dumps(
                    build_ocr_review_decisions_template(result),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"created editable decisions: {decisions_path}")

    unresolved = sum(
        diagnostic.ocr_conflict is not None and diagnostic.review_decision_id is None
        for diagnostic in result.page_diagnostics
    )
    reviewed = sum(
        diagnostic.review_decision_id is not None for diagnostic in result.page_diagnostics
    )
    print(
        f"{args.pdf}: conflicts={len(evidence['conflicts'])} reviewed={reviewed} "
        f"unresolved={unresolved} -> {evidence_markdown_path}"
    )


if __name__ == "__main__":
    main()

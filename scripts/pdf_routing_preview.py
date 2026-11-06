"""Generate deterministic JSON and Markdown previews for PDF page routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.pdf_auto_routing import parse_pdf_auto
from ai_learning_rag.pdf_extraction import parse_pdf
from ai_learning_rag.pdf_routing import PdfRoutingConfig, analyze_pdf_pages, build_route_plan
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheMode
from ai_learning_rag.pdf_routing_preview import (
    build_plan_markdown,
    build_plan_payload,
    build_result_markdown,
    build_result_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
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
        "--run-parsers",
        action="store_true",
        help="run Docling/OCR routes; without this flag only cheap diagnostics are generated",
    )
    args = parser.parse_args()
    config = PdfRoutingConfig.from_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in args.pdfs:
        if args.run_parsers:
            cache = PdfRoutingCache(
                args.cache_dir,
                RoutingCacheMode(args.cache_mode),
            )
            result = parse_pdf_auto(pdf_path, config, cache=cache)
            payload = build_result_payload(result)
            markdown = build_result_markdown(payload)
        else:
            baseline = parse_pdf(pdf_path)
            signals = analyze_pdf_pages(pdf_path)
            decisions = build_route_plan(signals, config)
            payload = build_plan_payload(pdf_path, baseline, signals, decisions, config)
            markdown = build_plan_markdown(payload)

        json_path = args.output_dir / f"{pdf_path.stem}.routing.json"
        markdown_path = args.output_dir / f"{pdf_path.stem}.routing.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(markdown, encoding="utf-8")
        print(f"{pdf_path}: mode={payload['mode']} -> {markdown_path}")


if __name__ == "__main__":
    main()

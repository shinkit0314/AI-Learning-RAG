"""Generate JSON and Markdown previews for structure-aware PDF chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_learning_rag.pdf_auto_routing import parse_pdf_auto
from ai_learning_rag.pdf_chunking import (
    PdfChunkingConfig,
    chunk_pdf_document,
)
from ai_learning_rag.pdf_chunking_preview import build_markdown_preview
from ai_learning_rag.pdf_extraction import parse_pdf
from ai_learning_rag.pdf_ocr_review import load_ocr_review_decisions
from ai_learning_rag.pdf_routing import PdfRoutingConfig
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheMode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to parse and chunk")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-chunking-preview"),
        help="directory for JSON and Markdown previews",
    )
    parser.add_argument("--target-size", type=int, default=400)
    parser.add_argument("--hard-max-size", type=int, default=600)
    parser.add_argument("--overlap-percent", type=int, default=20)
    parser.add_argument("--preview-characters", type=int, default=1200)
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help="route pages through PyMuPDF, Docling, and OCR before chunking",
    )
    parser.add_argument(
        "--routing-config",
        type=Path,
        default=Path("configs/pdf-routing-v1.json"),
    )
    parser.add_argument(
        "--routing-cache-dir",
        type=Path,
        default=Path("output/pdf-routing-cache"),
    )
    parser.add_argument(
        "--routing-cache-mode",
        choices=[mode.value for mode in RoutingCacheMode],
        default=RoutingCacheMode.READ_WRITE.value,
    )
    parser.add_argument(
        "--ocr-review-decisions",
        type=Path,
        help="apply document-bound manual OCR decisions before chunking",
    )
    args = parser.parse_args()
    if args.ocr_review_decisions is not None and not args.auto_route:
        parser.error("--ocr-review-decisions requires --auto-route")
    if args.ocr_review_decisions is not None and len(args.pdfs) != 1:
        parser.error("--ocr-review-decisions requires exactly one PDF")
    config = PdfChunkingConfig(
        target_size=args.target_size,
        hard_max_size=args.hard_max_size,
        overlap_percent=args.overlap_percent,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_config = PdfRoutingConfig.from_json(args.routing_config) if args.auto_route else None
    routing_cache = (
        PdfRoutingCache(
            args.routing_cache_dir,
            RoutingCacheMode(args.routing_cache_mode),
        )
        if args.auto_route
        else None
    )
    ocr_review_decisions = (
        load_ocr_review_decisions(args.ocr_review_decisions)
        if args.ocr_review_decisions is not None
        else None
    )

    for pdf_path in args.pdfs:
        if routing_config is not None:
            routing_result = parse_pdf_auto(
                pdf_path,
                routing_config,
                cache=routing_cache,
                ocr_review_decisions=ocr_review_decisions,
            )
            document = routing_result.document
        else:
            document = parse_pdf(pdf_path)
        result = chunk_pdf_document(document, config)
        payload = {
            "document": {
                "document_id": document.document_id,
                "source": document.source,
                "file_sha256": document.file_sha256,
                "parser_name": document.parser_name,
                "parser_version": document.parser_version,
                "page_count": document.page_count,
                "routing_profile": document.routing_profile,
            },
            "chunking_profile": config.chunking_profile,
            "status": result.status.value,
            "chunks": [asdict(chunk) for chunk in result.chunks],
            "diagnostics": [asdict(diagnostic) for diagnostic in result.diagnostics],
        }
        json_path = args.output_dir / f"{pdf_path.stem}.chunks.json"
        markdown_path = args.output_dir / f"{pdf_path.stem}.chunks.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            build_markdown_preview(
                document,
                result,
                preview_characters=args.preview_characters,
            ),
            encoding="utf-8",
        )
        print(
            f"{pdf_path}: status={result.status.value} chunks={len(result.chunks)} "
            f"diagnostics={len(result.diagnostics)} -> {markdown_path}"
        )


if __name__ == "__main__":
    main()

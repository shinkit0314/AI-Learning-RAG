"""Run Docling and save output in the project's parser-independent schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ai_learning_rag.docling_extraction import (
    create_docling_converter,
    parse_pdf_with_docling,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to parse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-docling-baseline"),
        help="directory for parser-independent Docling JSON files",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="enable Docling OCR instead of evaluating the existing text layer only",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help="ignore the PDF text layer and OCR the rendered full page (requires --ocr)",
    )
    parser.add_argument(
        "--ocr-language",
        default="ch",
        help="RapidOCR language code (default: ch for simplified Chinese)",
    )
    args = parser.parse_args()
    if args.force_full_page_ocr and not args.ocr:
        parser.error("--force-full-page-ocr requires --ocr")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converter = create_docling_converter(
        do_ocr=args.ocr,
        force_full_page_ocr=args.force_full_page_ocr,
        ocr_language=args.ocr_language,
    )
    for pdf_path in args.pdfs:
        document = parse_pdf_with_docling(
            pdf_path,
            do_ocr=args.ocr,
            force_full_page_ocr=args.force_full_page_ocr,
            ocr_language=args.ocr_language,
            converter=converter,
        )
        output_path = args.output_dir / f"{pdf_path.stem}.docling.json"
        output_path.write_text(
            json.dumps(asdict(document), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        block_types = Counter(block.block_type for page in document.pages for block in page.blocks)
        summary = ", ".join(
            f"{block_type}={count}" for block_type, count in sorted(block_types.items())
        )
        print(
            f"{pdf_path}: pages={document.page_count} warnings={len(document.warnings)} "
            f"{summary} -> {output_path}"
        )


if __name__ == "__main__":
    main()

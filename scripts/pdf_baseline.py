"""Run the PyMuPDF baseline and save traceable JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_learning_rag.pdf_extraction import parse_pdf, parsed_document_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF files to parse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-baseline"),
        help="directory for parser JSON files",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in args.pdfs:
        document = parse_pdf(pdf_path)
        output_path = args.output_dir / f"{pdf_path.stem}.pymupdf.json"
        output_path.write_text(
            json.dumps(parsed_document_to_dict(document), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        parsed_pages = sum(page.status == "parsed" for page in document.pages)
        print(
            f"{pdf_path}: pages={document.page_count} parsed={parsed_pages} "
            f"warnings={len(document.warnings)} -> {output_path}"
        )


if __name__ == "__main__":
    main()

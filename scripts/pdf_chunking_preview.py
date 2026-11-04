"""Generate JSON and Markdown previews for structure-aware PDF chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ai_learning_rag.pdf_chunking import (
    PdfChunkingConfig,
    chunk_pdf_document,
)
from ai_learning_rag.pdf_chunking_preview import build_markdown_preview
from ai_learning_rag.pdf_extraction import parse_pdf


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
    args = parser.parse_args()
    config = PdfChunkingConfig(
        target_size=args.target_size,
        hard_max_size=args.hard_max_size,
        overlap_percent=args.overlap_percent,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in args.pdfs:
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

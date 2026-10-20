"""PyMuPDF baseline extraction with page, bbox, and scan diagnostics."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Any
from unicodedata import normalize

import pymupdf

from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParseStatus,
)

FULL_PAGE_IMAGE_THRESHOLD = 0.8
MIN_USEFUL_TEXT_CHARACTERS = 20
MARGIN_BAND_RATIO = 0.1
MAX_MARGIN_BLOCK_HEIGHT_RATIO = 0.04
PAGE_NUMBER_PATTERN = re.compile(
    r"(?:page\s*)?(?:\d+|[ivxlcdm]+)(?:\s*(?:of|/)\s*\d+)?",
    re.IGNORECASE,
)


def parse_pdf(source: str | Path) -> ParsedDocument:
    """Extract text blocks and diagnostics without silently accepting empty scans."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF file")

    file_sha256 = _file_sha256(source_path)
    document_warnings: list[str] = []
    pages: list[ParsedPage] = []

    with pymupdf.open(source_path) as document:
        if document.needs_pass:
            raise ValueError("password-protected PDFs are not supported by this baseline")

        for page_index, page in enumerate(document):
            parsed_page = _parse_page(page, page_index + 1, file_sha256)
            pages.append(parsed_page)
            document_warnings.extend(
                f"PAGE_{parsed_page.page_number}_{warning}" for warning in parsed_page.warnings
            )
            if parsed_page.status is ParseStatus.NEEDS_OCR:
                document_warnings.append(f"PAGE_{parsed_page.page_number}_NEEDS_OCR")
            elif parsed_page.status is ParseStatus.FAILED:
                document_warnings.append(f"PAGE_{parsed_page.page_number}_PARSE_FAILED")

    pages = list(_classify_margin_blocks(tuple(pages)))

    return ParsedDocument(
        document_id=f"pdf-{file_sha256[:16]}",
        source=source_path.as_uri(),
        file_sha256=file_sha256,
        parser_name="pymupdf",
        parser_version=pymupdf.VersionBind,
        pages=tuple(pages),
        warnings=tuple(document_warnings),
    )


def parsed_document_to_dict(document: ParsedDocument) -> dict[str, Any]:
    """Convert the parser-independent dataclasses to JSON-compatible data."""
    return asdict(document)


def _parse_page(
    page: pymupdf.Page,
    page_number: int,
    file_sha256: str,
) -> ParsedPage:
    image_info = page.get_image_info()
    image_coverage_ratio = _image_coverage_ratio(page.rect, image_info)
    raw_text_blocks = [
        block
        for block in page.get_text("blocks", sort=True)
        if len(block) >= 7 and block[6] == 0 and block[4].strip()
    ]
    text_character_count = sum(len("".join(block[4].split())) for block in raw_text_blocks)

    kind, status, extraction_method, warnings = _classify_page(
        text_character_count=text_character_count,
        image_coverage_ratio=image_coverage_ratio,
    )

    blocks = tuple(
        ParsedBlock(
            id=f"{file_sha256[:12]}-p{page_number:04d}-b{reading_order:04d}",
            page_number=page_number,
            block_type=BlockType.PARAGRAPH,
            text=block[4].strip(),
            bbox=BoundingBox(*map(float, block[:4])),
            reading_order=reading_order,
            extraction_method=extraction_method,
        )
        for reading_order, block in enumerate(raw_text_blocks)
    )

    return ParsedPage(
        page_number=page_number,
        width=float(page.rect.width),
        height=float(page.rect.height),
        kind=kind,
        status=status,
        blocks=blocks,
        warnings=warnings,
        text_character_count=text_character_count,
        image_count=len(image_info),
        image_coverage_ratio=image_coverage_ratio,
    )


def _classify_page(
    *,
    text_character_count: int,
    image_coverage_ratio: float,
) -> tuple[PageKind, ParseStatus, ExtractionMethod, tuple[str, ...]]:
    has_useful_text = text_character_count >= MIN_USEFUL_TEXT_CHARACTERS
    has_full_page_image = image_coverage_ratio >= FULL_PAGE_IMAGE_THRESHOLD

    if has_full_page_image and has_useful_text:
        return (
            PageKind.SCANNED_WITH_OCR,
            ParseStatus.PARSED,
            ExtractionMethod.EMBEDDED_OCR,
            ("EMBEDDED_OCR_TEXT_LAYER",),
        )
    if has_full_page_image:
        return (
            PageKind.SCANNED_IMAGE,
            ParseStatus.NEEDS_OCR,
            ExtractionMethod.RERUN_OCR,
            ("OCR_REQUIRED",),
        )
    if has_useful_text:
        return PageKind.NATIVE_TEXT, ParseStatus.PARSED, ExtractionMethod.NATIVE, ()
    return (
        PageKind.NATIVE_TEXT,
        ParseStatus.FAILED,
        ExtractionMethod.NATIVE,
        ("NO_EXTRACTABLE_TEXT",),
    )


def _classify_margin_blocks(pages: tuple[ParsedPage, ...]) -> tuple[ParsedPage, ...]:
    """Label high-confidence repeated headers and footers without dropping text."""
    if not pages:
        return ()

    required_page_count = max(2, ceil(len(pages) / 2))
    pages_by_signature: dict[BlockType, dict[str, set[int]]] = {
        BlockType.HEADER: defaultdict(set),
        BlockType.FOOTER: defaultdict(set),
    }

    for page in pages:
        for block in page.blocks:
            margin_type = _margin_type(block, page)
            if margin_type is not None and _is_small_margin_block(block, page):
                pages_by_signature[margin_type][_margin_signature(block.text)].add(page.page_number)

    repeated_signatures = {
        margin_type: {
            signature
            for signature, page_numbers in signatures.items()
            if len(page_numbers) >= required_page_count
        }
        for margin_type, signatures in pages_by_signature.items()
    }

    classified_pages: list[ParsedPage] = []
    for page in pages:
        classified_blocks: list[ParsedBlock] = []
        for block in page.blocks:
            margin_type = _margin_type(block, page)
            block_type = block.block_type
            if margin_type is BlockType.FOOTER and _is_page_number(block.text):
                block_type = BlockType.FOOTER
            elif (
                margin_type is not None
                and _is_small_margin_block(block, page)
                and _margin_signature(block.text) in repeated_signatures[margin_type]
            ):
                block_type = margin_type
            classified_blocks.append(replace(block, block_type=block_type))
        classified_pages.append(replace(page, blocks=tuple(classified_blocks)))

    return tuple(classified_pages)


def _margin_type(block: ParsedBlock, page: ParsedPage) -> BlockType | None:
    if block.bbox.y1 <= page.height * MARGIN_BAND_RATIO:
        return BlockType.HEADER
    if block.bbox.y0 >= page.height * (1 - MARGIN_BAND_RATIO):
        return BlockType.FOOTER
    return None


def _is_small_margin_block(block: ParsedBlock, page: ParsedPage) -> bool:
    return (block.bbox.y1 - block.bbox.y0) <= page.height * MAX_MARGIN_BLOCK_HEIGHT_RATIO


def _margin_signature(text: str) -> str:
    canonical_text = normalize("NFKC", text).casefold()
    canonical_text = re.sub(r"\d+", "#", canonical_text)
    return re.sub(r"\s+", "", canonical_text)


def _is_page_number(text: str) -> bool:
    return PAGE_NUMBER_PATTERN.fullmatch(normalize("NFKC", text).strip()) is not None


def _image_coverage_ratio(
    page_rect: pymupdf.Rect,
    image_info: list[dict[str, Any]],
) -> float:
    if page_rect.is_empty:
        return 0.0

    visible_area = 0.0
    for image in image_info:
        image_rect = pymupdf.Rect(image["bbox"]) & page_rect
        if not image_rect.is_empty and not image_rect.is_infinite:
            visible_area += image_rect.get_area()

    return min(1.0, visible_area / page_rect.get_area())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

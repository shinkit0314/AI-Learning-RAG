"""Tests for the traceable PyMuPDF extraction baseline."""

import json
from hashlib import sha256
from pathlib import Path
from unicodedata import normalize

import pymupdf
import pytest

from ai_learning_rag.pdf_extraction import parse_pdf
from ai_learning_rag.pdf_schema import BlockType, ExtractionMethod, PageKind, ParseStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DATA_DIR = PROJECT_ROOT / "data" / "PDF"
SAMPLE_PDF_PATHS = tuple(
    PDF_DATA_DIR / filename for filename in ("native.pdf", "layout.pdf", "scan.pdf")
)
requires_local_pdf_samples = pytest.mark.skipif(
    not all(path.is_file() for path in SAMPLE_PDF_PATHS),
    reason="local PDF research samples are not stored in the public repository",
)


def normalized_text(text: str) -> str:
    """Normalize presentation differences without changing the expected wording."""
    return "".join(normalize("NFKC", text).split())


def test_manifest_describes_three_reviewable_sample_types() -> None:
    manifest = json.loads((PDF_DATA_DIR / "sample_manifest.json").read_text(encoding="utf-8"))

    assert [sample["expected_type"] for sample in manifest["samples"]] == [
        "native_text",
        "complex_layout_text",
        "scanned_with_ocr",
    ]
    assert all(len(sample["manual_checks"]) >= 3 for sample in manifest["samples"])


@requires_local_pdf_samples
def test_manifest_keeps_samples_fixed_and_reviewable() -> None:
    manifest = json.loads((PDF_DATA_DIR / "sample_manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["samples"]) == 3
    for sample in manifest["samples"]:
        pdf_path = PROJECT_ROOT / sample["path"]
        assert pdf_path.is_file()
        assert sha256(pdf_path.read_bytes()).hexdigest() == sample["sha256"]
        assert len(sample["manual_checks"]) >= 3
        with pymupdf.open(pdf_path) as document:
            assert document.page_count == sample["expected_page_count"]


@requires_local_pdf_samples
def test_manifest_text_checks_match_baseline_output() -> None:
    manifest = json.loads((PDF_DATA_DIR / "sample_manifest.json").read_text(encoding="utf-8"))

    for sample in manifest["samples"]:
        document = parse_pdf(PROJECT_ROOT / sample["path"])
        for check in sample["manual_checks"]:
            assert check["kind"] == "text_contains"
            page = document.pages[check["page_number"] - 1]
            actual_text = "\n".join(block.text for block in page.blocks)
            assert normalized_text(check["expected"]) in normalized_text(actual_text)


@requires_local_pdf_samples
def test_native_sample_returns_traceable_blocks() -> None:
    document = parse_pdf(PDF_DATA_DIR / "native.pdf")

    assert document.parser_name == "pymupdf"
    assert document.file_sha256 == (
        "cdb8c8d333c4c3e9ac0f2547b67ebebe646a33ef6be83b587cb6602cb6ab40d9"
    )
    assert document.page_count == 6
    assert all(page.status is ParseStatus.PARSED for page in document.pages)
    assert all(page.blocks for page in document.pages)
    assert all(
        block.page_number == page.page_number
        and block.bbox.x1 > block.bbox.x0
        and block.bbox.y1 > block.bbox.y0
        for page in document.pages
        for block in page.blocks
    )


@requires_local_pdf_samples
def test_repeated_margin_text_and_page_numbers_are_classified_without_deletion() -> None:
    document = parse_pdf(PDF_DATA_DIR / "layout.pdf")

    for page in document.pages[1:]:
        block_types = {block.text: block.block_type for block in page.blocks}
        assert block_types["R. Sapkota et al."] is BlockType.HEADER
        assert block_types["Information Fusion 126 (2026) 103599"] is BlockType.HEADER
        assert block_types[str(page.page_number)] is BlockType.FOOTER

    first_body_block = next(
        block
        for block in document.pages[1].blocks
        if block.text.startswith("configuration; and CLIPS")
    )
    assert first_body_block.block_type is BlockType.PARAGRAPH


@requires_local_pdf_samples
def test_large_repeated_slide_title_is_not_misclassified_as_a_header() -> None:
    document = parse_pdf(PDF_DATA_DIR / "native.pdf")

    slide_titles = [
        block
        for page in document.pages
        for block in page.blocks
        if block.text == "強化学習の定式化" and block.bbox.y0 < 10
    ]
    assert len(slide_titles) == 2
    assert all(block.block_type is BlockType.PARAGRAPH for block in slide_titles)
    assert all(
        any(
            block.text == "©︎MATSUO LAB, THE UNIVERSITY OF TOKYO"
            and block.block_type is BlockType.FOOTER
            for block in page.blocks
        )
        for page in document.pages
    )


@requires_local_pdf_samples
def test_scanned_sample_records_embedded_ocr_instead_of_treating_it_as_native() -> None:
    document = parse_pdf(PDF_DATA_DIR / "scan.pdf")

    assert all(page.kind is PageKind.SCANNED_WITH_OCR for page in document.pages)
    assert all(page.image_coverage_ratio >= 0.8 for page in document.pages)
    assert all(
        block.extraction_method is ExtractionMethod.EMBEDDED_OCR
        for page in document.pages
        for block in page.blocks
    )
    assert document.warnings == tuple(
        f"PAGE_{page_number}_EMBEDDED_OCR_TEXT_LAYER" for page_number in range(1, 6)
    )


def test_image_only_scan_is_explicitly_routed_to_ocr(tmp_path: Path) -> None:
    text_pdf = pymupdf.open()
    source_page = text_pdf.new_page(width=300, height=400)
    source_page.insert_text((40, 80), "Rendered into an image")
    pixmap = source_page.get_pixmap()

    scan_pdf = pymupdf.open()
    scan_page = scan_pdf.new_page(width=300, height=400)
    scan_page.insert_image(scan_page.rect, stream=pixmap.tobytes("png"))
    scan_path = tmp_path / "image-only-scan.pdf"
    scan_pdf.save(scan_path)
    scan_pdf.close()
    text_pdf.close()

    document = parse_pdf(scan_path)
    page = document.pages[0]

    assert page.kind is PageKind.SCANNED_IMAGE
    assert page.status is ParseStatus.NEEDS_OCR
    assert not page.blocks
    assert "OCR_REQUIRED" in page.warnings
    assert document.warnings == ("PAGE_1_OCR_REQUIRED", "PAGE_1_NEEDS_OCR")

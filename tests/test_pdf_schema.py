"""Unit tests for parser-independent PDF data contracts."""

import pytest

from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ChunkCandidate,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParsedPicture,
    ParseStatus,
)


def build_block(
    *,
    block_id: str = "block_1",
    page_number: int = 1,
    reading_order: int = 0,
    text: str = "Traceable PDF text",
) -> ParsedBlock:
    """Build a small valid block for focused schema tests."""
    return ParsedBlock(
        id=block_id,
        page_number=page_number,
        block_type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(72.0, 100.0, 500.0, 140.0),
        reading_order=reading_order,
        extraction_method=ExtractionMethod.NATIVE,
        confidence=1.0,
    )


def test_parsed_document_keeps_page_and_bbox_traceability() -> None:
    """Every block should remain linked to a one-based page and source region."""
    block = build_block()
    page = ParsedPage(
        page_number=1,
        width=612.0,
        height=792.0,
        kind=PageKind.NATIVE_TEXT,
        status=ParseStatus.PARSED,
        blocks=(block,),
    )
    document = ParsedDocument(
        document_id="doc_pdf",
        source="file:///data/PDF/native.pdf",
        file_sha256="a" * 64,
        parser_name="baseline",
        parser_version="0.1",
        pages=(page,),
    )

    assert document.page_count == 1
    assert document.pages[0].blocks[0].page_number == 1
    assert document.pages[0].blocks[0].bbox.as_tuple() == (72.0, 100.0, 500.0, 140.0)


def test_chunk_candidate_preserves_source_blocks_pages_and_regions() -> None:
    """Chunk input should expose the exact blocks needed for later citations."""
    first = build_block(block_id="block_1", reading_order=0, text="First paragraph")
    second = build_block(block_id="block_2", reading_order=1, text="Second paragraph")

    candidate = ChunkCandidate(
        document_id="doc_pdf",
        section_path=("3 Method", "3.1 Architecture"),
        blocks=(first, second),
    )

    assert candidate.content == "First paragraph\nSecond paragraph"
    assert candidate.page_start == 1
    assert candidate.page_end == 1
    assert candidate.source_block_ids == ("block_1", "block_2")
    assert candidate.bboxes == (
        (72.0, 100.0, 500.0, 140.0),
        (72.0, 100.0, 500.0, 140.0),
    )


@pytest.mark.parametrize(
    "bbox",
    [
        (1.0, 1.0, 1.0, 2.0),
        (1.0, 2.0, 2.0, 1.0),
        (float("nan"), 1.0, 2.0, 2.0),
    ],
)
def test_bounding_box_rejects_invalid_rectangles(
    bbox: tuple[float, float, float, float],
) -> None:
    """Invalid coordinates must fail before they reach citation metadata."""
    with pytest.raises(ValueError):
        BoundingBox(*bbox)


def test_page_rejects_a_block_from_another_page() -> None:
    """A page must never silently contain a block with a different page number."""
    with pytest.raises(ValueError, match="containing page"):
        ParsedPage(
            page_number=1,
            width=612.0,
            height=792.0,
            kind=PageKind.COMPLEX_LAYOUT_TEXT,
            status=ParseStatus.PARSED,
            blocks=(build_block(page_number=2),),
        )


def test_page_rejects_invalid_diagnostic_counts() -> None:
    """Page diagnostics must remain safe to compare across parser runs."""
    with pytest.raises(ValueError, match="image_coverage_ratio"):
        ParsedPage(
            page_number=1,
            width=612.0,
            height=792.0,
            kind=PageKind.NATIVE_TEXT,
            status=ParseStatus.PARSED,
            image_coverage_ratio=1.1,
        )


def test_document_rejects_noncontiguous_page_numbers() -> None:
    """Missing pages must be represented explicitly instead of disappearing."""
    page = ParsedPage(
        page_number=2,
        width=612.0,
        height=792.0,
        kind=PageKind.SCANNED_IMAGE,
        status=ParseStatus.NEEDS_OCR,
    )

    with pytest.raises(ValueError, match="contiguous"):
        ParsedDocument(
            document_id="doc_pdf",
            source="file:///data/PDF/scan.pdf",
            file_sha256="b" * 64,
            parser_name="baseline",
            parser_version="0.1",
            pages=(page,),
        )


def test_chunk_candidate_rejects_reversed_reading_order() -> None:
    """Downstream chunking must not silently scramble source reading order."""
    first = build_block(block_id="block_1", reading_order=1)
    second = build_block(block_id="block_2", reading_order=0)

    with pytest.raises(ValueError, match="reading order"):
        ChunkCandidate(document_id="doc_pdf", blocks=(first, second))


def test_picture_text_is_traceable_but_rejected_from_ordinary_chunks() -> None:
    picture_id = "picture_1"
    picture_text = ParsedBlock(
        id="picture_text_1",
        page_number=1,
        block_type=BlockType.PICTURE_TEXT,
        text="diagram node",
        bbox=BoundingBox(100, 100, 180, 130),
        reading_order=0,
        extraction_method=ExtractionMethod.RERUN_OCR,
        parent_picture_id=picture_id,
    )
    picture = ParsedPicture(
        id=picture_id,
        page_number=1,
        bbox=BoundingBox(80, 80, 500, 400),
        child_block_ids=(picture_text.id,),
    )
    page = ParsedPage(
        page_number=1,
        width=600,
        height=800,
        kind=PageKind.SCANNED_WITH_OCR,
        status=ParseStatus.PARSED,
        blocks=(picture_text,),
        pictures=(picture,),
    )

    assert page.pictures[0].child_block_ids == (picture_text.id,)
    with pytest.raises(ValueError, match="picture text"):
        ChunkCandidate(document_id="doc_pdf", blocks=(picture_text,))

"""Unit tests for parser-independent PDF data contracts."""

import pytest

from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ChunkBlockSlice,
    ChunkCandidate,
    ChunkContentType,
    ChunkSourceRegion,
    ChunkWarning,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParsedPicture,
    ParseStatus,
    TraceablePdfChunk,
    WarningSeverity,
    build_traceable_pdf_chunk,
)


def build_block(
    *,
    block_id: str = "block_1",
    page_number: int = 1,
    reading_order: int = 0,
    text: str = "Traceable PDF text",
    block_type: BlockType = BlockType.PARAGRAPH,
) -> ParsedBlock:
    """Build a small valid block for focused schema tests."""
    return ParsedBlock(
        id=block_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
        bbox=BoundingBox(72.0, 100.0, 500.0, 140.0),
        reading_order=reading_order,
        extraction_method=ExtractionMethod.NATIVE,
        confidence=1.0,
    )


def candidate_from_blocks(
    *blocks: ParsedBlock,
    section_path: tuple[str, ...] = (),
) -> ChunkCandidate:
    """Create a whole-block candidate for tests not focused on partial slices."""
    return ChunkCandidate.from_blocks(
        document_id="doc_pdf",
        blocks=blocks,
        section_path=section_path,
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

    candidate = ChunkCandidate.from_blocks(
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
        ChunkCandidate.from_blocks(document_id="doc_pdf", blocks=(first, second))


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
        ChunkCandidate.from_blocks(document_id="doc_pdf", blocks=(picture_text,))


def test_finished_chunk_maps_content_spans_back_to_pdf_regions() -> None:
    """Chunk text must retain the exact page and bbox for every source block."""
    first = build_block(block_id="block_1", reading_order=0, text="First paragraph")
    second = build_block(
        block_id="block_2",
        page_number=2,
        reading_order=0,
        text="Second paragraph",
    )
    candidate = ChunkCandidate.from_blocks(
        document_id="doc_pdf",
        section_path=("3 Method", "3.1 Architecture"),
        blocks=(first, second),
    )

    chunk = build_traceable_pdf_chunk(
        candidate,
        chunk_index=4,
        chunking_profile="pdf-structure-v1",
    )

    assert chunk.id.startswith("pdf_chunk_")
    assert len(chunk.id.removeprefix("pdf_chunk_")) == 8
    assert len(chunk.source_fingerprint) == 64
    assert chunk.chunking_profile == "pdf-structure-v1"
    assert chunk.document_id == "doc_pdf"
    assert chunk.chunk_index == 4
    assert chunk.content == "First paragraph\nSecond paragraph"
    assert chunk.content_type is ChunkContentType.BODY_TEXT
    assert chunk.section_path == ("3 Method", "3.1 Architecture")
    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert chunk.source_block_ids == ("block_1", "block_2")
    assert chunk.bboxes == (
        (72.0, 100.0, 500.0, 140.0),
        (72.0, 100.0, 500.0, 140.0),
    )
    assert [
        chunk.content[region.chunk_content_start : region.chunk_content_end]
        for region in chunk.source_regions
    ] == ["First paragraph", "Second paragraph"]
    assert [
        (region.source_text_start, region.source_text_end) for region in chunk.source_regions
    ] == [
        (0, len("First paragraph")),
        (0, len("Second paragraph")),
    ]


def test_partial_block_slice_maps_chunk_text_back_to_source_offsets() -> None:
    """One chunk may safely contain only a contiguous part of a source block."""
    block = build_block(text="0123456789")
    candidate = ChunkCandidate(
        document_id="doc_pdf",
        block_slices=(ChunkBlockSlice(block, source_text_start=2, source_text_end=7),),
    )

    chunk = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )

    assert chunk.content == "23456"
    region = chunk.source_regions[0]
    assert (region.chunk_content_start, region.chunk_content_end) == (0, 5)
    assert (region.source_text_start, region.source_text_end) == (2, 7)


def test_equal_text_from_different_source_ranges_has_different_fingerprint() -> None:
    """Repeated text inside one block must remain distinguishable by source offsets."""
    block = build_block(text="abcabc")
    first_candidate = ChunkCandidate(
        document_id="doc_pdf",
        block_slices=(ChunkBlockSlice(block, 0, 3),),
    )
    second_candidate = ChunkCandidate(
        document_id="doc_pdf",
        block_slices=(ChunkBlockSlice(block, 3, 6),),
    )

    first = build_traceable_pdf_chunk(
        first_candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )
    second = build_traceable_pdf_chunk(
        second_candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )

    assert first.content == second.content == "abc"
    assert first.source_fingerprint != second.source_fingerprint
    assert first.id != second.id


def test_finished_chunk_id_does_not_depend_on_display_order() -> None:
    """Reordering other chunks must not change this chunk's identity."""
    candidate = candidate_from_blocks(build_block())

    first_position = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )
    later_position = build_traceable_pdf_chunk(
        candidate,
        chunk_index=9,
        chunking_profile="pdf-structure-v1",
    )

    assert first_position.id == later_position.id


def test_chunking_profile_changes_id_but_not_source_fingerprint() -> None:
    """Processing lineage should change identity without hiding equal source content."""
    candidate = candidate_from_blocks(build_block())

    first = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )
    second = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v2",
    )

    assert first.source_fingerprint == second.source_fingerprint
    assert first.id != second.id


def test_finished_chunk_id_changes_with_source_identity() -> None:
    """Equal-looking text from different source blocks must remain distinguishable."""
    first_candidate = candidate_from_blocks(build_block(block_id="block_1"))
    second_candidate = candidate_from_blocks(build_block(block_id="block_2"))

    first = build_traceable_pdf_chunk(
        first_candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )
    second = build_traceable_pdf_chunk(
        second_candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
    )

    assert first.id != second.id


def test_code_block_can_become_a_first_class_code_chunk() -> None:
    """Code must retain indentation and avoid falling through the OTHER type."""
    code_text = "def answer():\n    return 42"
    candidate = candidate_from_blocks(build_block(text=code_text, block_type=BlockType.CODE))

    chunk = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
        content_type=ChunkContentType.CODE,
    )

    assert chunk.content_type is ChunkContentType.CODE
    assert chunk.content == code_text
    assert chunk.source_regions[0].block_type is BlockType.CODE


def test_other_content_is_preserved_with_a_structured_warning() -> None:
    """Unknown content must be quarantined without being discarded."""
    block = build_block(block_type=BlockType.OTHER)
    candidate = candidate_from_blocks(block)
    warning = ChunkWarning(
        code="UNSUPPORTED_CONTENT_TYPE",
        severity=WarningSeverity.WARNING,
        message="The source block type has no dedicated chunking strategy.",
        source_block_ids=(block.id,),
    )

    chunk = build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
        content_type=ChunkContentType.OTHER,
        warnings=(warning,),
    )

    assert chunk.content_type is ChunkContentType.OTHER
    assert chunk.warnings == (warning,)


def test_finished_chunk_rejects_a_source_range_outside_content() -> None:
    """Invalid offsets must fail before citations can point at unrelated text."""
    region = ChunkSourceRegion(
        block_id="block_1",
        page_number=1,
        bbox=BoundingBox(72, 100, 500, 140),
        block_type=BlockType.PARAGRAPH,
        extraction_method=ExtractionMethod.NATIVE,
        reading_order=0,
        chunk_content_start=0,
        chunk_content_end=20,
        source_text_start=0,
        source_text_end=20,
    )

    with pytest.raises(ValueError, match="inside chunk content"):
        TraceablePdfChunk(
            id="pdf_chunk_invalid",
            source_fingerprint="a" * 64,
            document_id="doc_pdf",
            chunk_index=0,
            chunking_profile="pdf-structure-v1",
            content="short text",
            content_type=ChunkContentType.BODY_TEXT,
            source_regions=(region,),
        )


def test_finished_chunk_rejects_overlapping_source_ranges() -> None:
    """One content span must not ambiguously claim two source blocks."""
    first = ChunkSourceRegion(
        block_id="block_1",
        page_number=1,
        bbox=BoundingBox(72, 100, 500, 140),
        block_type=BlockType.PARAGRAPH,
        extraction_method=ExtractionMethod.NATIVE,
        reading_order=0,
        chunk_content_start=0,
        chunk_content_end=6,
        source_text_start=0,
        source_text_end=6,
    )
    second = ChunkSourceRegion(
        block_id="block_2",
        page_number=1,
        bbox=BoundingBox(72, 150, 500, 190),
        block_type=BlockType.PARAGRAPH,
        extraction_method=ExtractionMethod.NATIVE,
        reading_order=1,
        chunk_content_start=5,
        chunk_content_end=10,
        source_text_start=0,
        source_text_end=5,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        TraceablePdfChunk(
            id="pdf_chunk_invalid",
            source_fingerprint="a" * 64,
            document_id="doc_pdf",
            chunk_index=0,
            chunking_profile="pdf-structure-v1",
            content="0123456789",
            content_type=ChunkContentType.BODY_TEXT,
            source_regions=(first, second),
        )


def test_finished_chunk_rejects_unmapped_non_whitespace_content() -> None:
    """Every meaningful character sent to Embedding must have a PDF source."""
    region = ChunkSourceRegion(
        block_id="block_1",
        page_number=1,
        bbox=BoundingBox(72, 100, 500, 140),
        block_type=BlockType.PARAGRAPH,
        extraction_method=ExtractionMethod.NATIVE,
        reading_order=0,
        chunk_content_start=7,
        chunk_content_end=17,
        source_text_start=0,
        source_text_end=10,
    )

    with pytest.raises(ValueError, match="unmapped source text"):
        TraceablePdfChunk(
            id="pdf_chunk_invalid",
            source_fingerprint="a" * 64,
            document_id="doc_pdf",
            chunk_index=0,
            chunking_profile="pdf-structure-v1",
            content="prefix mapped text",
            content_type=ChunkContentType.BODY_TEXT,
            source_regions=(region,),
        )

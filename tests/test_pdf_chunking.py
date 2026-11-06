from dataclasses import replace

import pytest

from ai_learning_rag.pdf_chunking import (
    ChunkingDiagnostic,
    ChunkingDiagnosticCode,
    PdfChunkingConfig,
    PdfChunkingResult,
    PdfChunkingStatus,
    PdfChunkPreparationResult,
    chunk_pdf_document,
    compose_cross_page_body_units,
    compose_same_page_body_units,
    prepare_pdf_document,
    preserve_atomic_figure_caption_units,
    preserve_atomic_formula_units,
    preserve_isolated_other_units,
    split_oversized_body_units,
    split_oversized_code_units,
    split_oversized_table_units,
)
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ChunkCandidate,
    ChunkContentType,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParsedPicture,
    ParseStatus,
    WarningSeverity,
    build_traceable_pdf_chunk,
)


def build_chunk(*, chunk_index: int = 0):
    block = ParsedBlock(
        id=f"block_{chunk_index}",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text=f"Chunk {chunk_index}",
        bbox=BoundingBox(72, 100 + chunk_index * 50, 500, 140 + chunk_index * 50),
        reading_order=chunk_index,
        extraction_method=ExtractionMethod.NATIVE,
    )
    candidate = ChunkCandidate.from_blocks(document_id="doc_pdf", blocks=(block,))
    return build_traceable_pdf_chunk(
        candidate,
        chunk_index=chunk_index,
        chunking_profile=PdfChunkingConfig().chunking_profile,
    )


def test_default_config_has_a_stable_readable_profile() -> None:
    config = PdfChunkingConfig()

    assert config.overlap_max == 80
    assert config.chunking_profile == (
        "pdf-structure-v1;length=characters;target=400;hard_max=600;overlap_percent=20"
    )
    assert replace(config, target_size=500).chunking_profile != config.chunking_profile


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_size": 0}, "target_size"),
        ({"target_size": 700}, "hard_max_size"),
        ({"overlap_percent": 100}, "overlap_percent"),
        ({"length_unit": "token count"}, "length_unit"),
        ({"strategy_version": "PDF v1"}, "strategy_version"),
    ],
)
def test_config_rejects_values_that_cannot_form_a_safe_profile(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        replace(PdfChunkingConfig(), **changes)


def test_diagnostic_requires_stable_page_and_block_references() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        ChunkingDiagnostic(
            code=ChunkingDiagnosticCode.UNKNOWN_PAGE_WARNING,
            severity=WarningSeverity.WARNING,
            message="Pages must be canonical.",
            page_numbers=(2, 1),
        )

    with pytest.raises(ValueError, match="block ids must be unique"):
        ChunkingDiagnostic(
            code=ChunkingDiagnosticCode.UNKNOWN_BLOCK_WARNING,
            severity=WarningSeverity.WARNING,
            message="Blocks must be canonical.",
            source_block_ids=("block_1", "block_1"),
        )


def test_result_status_is_derived_from_chunks_and_error_diagnostics() -> None:
    chunk = build_chunk()
    error = ChunkingDiagnostic(
        code=ChunkingDiagnosticCode.PAGE_PARSE_FAILED_SKIPPED,
        severity=WarningSeverity.ERROR,
        message="One page was not parsed.",
        page_numbers=(2,),
    )

    assert PdfChunkingResult(chunks=(chunk,)).status is PdfChunkingStatus.COMPLETED
    assert PdfChunkingResult(chunks=(chunk,), diagnostics=(error,)).status is (
        PdfChunkingStatus.PARTIAL
    )


def test_empty_result_adds_a_terminal_error_diagnostic() -> None:
    result = PdfChunkingResult()

    assert result.status is PdfChunkingStatus.FAILED
    assert result.diagnostics[-1].code is ChunkingDiagnosticCode.NO_CHUNKS_PRODUCED
    assert result.diagnostics[-1].severity is WarningSeverity.ERROR


def test_result_requires_contiguous_chunk_indexes() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        PdfChunkingResult(chunks=(build_chunk(chunk_index=1),))


def make_block(
    block_id: str,
    reading_order: int,
    block_type: BlockType,
    text: str,
    *,
    page_number: int = 1,
    heading_level: int | None = None,
    parent_picture_id: str | None = None,
    warnings: tuple[str, ...] = (),
    extraction_method: ExtractionMethod = ExtractionMethod.LAYOUT,
) -> ParsedBlock:
    return ParsedBlock(
        id=block_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
        bbox=BoundingBox(72, 50 + reading_order * 20, 500, 65 + reading_order * 20),
        reading_order=reading_order,
        extraction_method=extraction_method,
        heading_level=heading_level,
        parent_picture_id=parent_picture_id,
        warnings=warnings,
    )


def make_page(
    page_number: int,
    *blocks: ParsedBlock,
    status: ParseStatus = ParseStatus.PARSED,
    warnings: tuple[str, ...] = (),
    pictures: tuple[ParsedPicture, ...] = (),
) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        width=595,
        height=842,
        kind=PageKind.COMPLEX_LAYOUT_TEXT,
        status=status,
        blocks=blocks,
        pictures=pictures,
        warnings=warnings,
        text_character_count=sum(len(block.text) for block in blocks),
    )


def make_document(
    *pages: ParsedPage,
    warnings: tuple[str, ...] = (),
) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-preparation",
        source="file:///sample.pdf",
        file_sha256="a" * 64,
        parser_name="test-parser",
        parser_version="1.0",
        pages=pages,
        warnings=warnings,
    )


def diagnostic_codes(result: PdfChunkPreparationResult) -> set[ChunkingDiagnosticCode]:
    return {diagnostic.code for diagnostic in result.diagnostics}


def test_diagnostic_rejects_unregistered_string_code() -> None:
    with pytest.raises(ValueError, match="ChunkingDiagnosticCode"):
        ChunkingDiagnostic(  # type: ignore[arg-type]
            code="AD_HOC_CODE",
            severity=WarningSeverity.WARNING,
            message="Arbitrary codes must not enter the contract.",
        )


def test_preparation_filters_fixed_noise_and_keeps_pending_headings_once() -> None:
    title = make_block("title", 0, BlockType.TITLE, "RAG Handbook")
    header = make_block("header", 1, BlockType.HEADER, "Company confidential")
    heading = make_block("heading", 2, BlockType.HEADING, "1 Chunking")
    page_number = make_block("page-number", 3, BlockType.PRINTED_PAGE_NUMBER, "1")
    first_body = make_block("body-1", 4, BlockType.PARAGRAPH, "First paragraph.")
    second_body = make_block("body-2", 5, BlockType.LIST_ITEM, "Second body unit.")

    result = prepare_pdf_document(
        make_document(
            make_page(
                1,
                title,
                header,
                heading,
                page_number,
                first_body,
                second_body,
            )
        )
    )

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("title", "heading", "body-1"),
        ("body-2",),
    ]
    assert all(
        unit.candidate.section_path == ("RAG Handbook", "1 Chunking") for unit in result.units
    )
    assert all(unit.content_type is ChunkContentType.BODY_TEXT for unit in result.units)
    filtered_ids = {
        block_id
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.BLOCK_FILTERED
        for block_id in diagnostic.source_block_ids
    }
    assert filtered_ids == {"header", "page-number"}


def test_preparation_excludes_picture_text_but_keeps_figure_caption() -> None:
    picture_text = make_block(
        "picture-text",
        0,
        BlockType.PICTURE_TEXT,
        "Node A",
        parent_picture_id="picture-1",
        warnings=("PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS",),
    )
    caption = make_block(
        "caption",
        1,
        BlockType.FIGURE_CAPTION,
        "Figure 1. Chunk pipeline",
        parent_picture_id="picture-1",
    )
    picture = ParsedPicture(
        id="picture-1",
        page_number=1,
        bbox=BoundingBox(50, 40, 520, 300),
        child_block_ids=("picture-text", "caption"),
    )

    result = prepare_pdf_document(
        make_document(make_page(1, picture_text, caption, pictures=(picture,)))
    )

    assert [unit.candidate.source_block_ids for unit in result.units] == [("caption",)]
    assert result.units[0].content_type is ChunkContentType.FIGURE_CAPTION
    assert ChunkingDiagnosticCode.BLOCK_FILTERED in diagnostic_codes(result)


def test_unknown_page_warning_blocks_page_but_unknown_block_warning_is_local() -> None:
    blocked_page_body = make_block("page-blocked", 0, BlockType.PARAGRAPH, "Unsafe page")
    blocked_block = make_block(
        "block-blocked",
        0,
        BlockType.PARAGRAPH,
        "Unsafe block",
        page_number=2,
        warnings=("NEW_PARSER_WARNING",),
    )
    accepted_block = make_block(
        "accepted",
        1,
        BlockType.PARAGRAPH,
        "Safe block",
        page_number=2,
    )

    result = prepare_pdf_document(
        make_document(
            make_page(1, blocked_page_body, warnings=("NEW_PAGE_WARNING",)),
            make_page(2, blocked_block, accepted_block),
        )
    )

    assert [unit.candidate.source_block_ids for unit in result.units] == [("accepted",)]
    assert ChunkingDiagnosticCode.UNKNOWN_PAGE_WARNING in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.UNKNOWN_BLOCK_WARNING in diagnostic_codes(result)


def test_known_warnings_are_accepted_and_bound_to_source_blocks() -> None:
    other = make_block(
        "other",
        0,
        BlockType.OTHER,
        "Custom parser content",
        warnings=("DOCLING_UNMAPPED_LABEL_CUSTOM",),
    )
    document = make_document(
        make_page(1, other, warnings=("EMBEDDED_OCR_TEXT_LAYER",)),
        warnings=("PAGE_1_EMBEDDED_OCR_TEXT_LAYER",),
    )

    result = prepare_pdf_document(document)

    assert len(result.units) == 1
    assert result.units[0].content_type is ChunkContentType.OTHER
    assert {warning.code for warning in result.units[0].warnings} == {
        "EMBEDDED_OCR_TEXT_LAYER",
        "DOCLING_UNMAPPED_LABEL_CUSTOM",
    }
    assert all(warning.source_block_ids == ("other",) for warning in result.units[0].warnings)
    assert ChunkingDiagnosticCode.SOURCE_DOCUMENT_WARNING_RECORDED not in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.UNSUPPORTED_CONTENT_TYPE in diagnostic_codes(result)


@pytest.mark.parametrize(
    "warning",
    [
        "EMBEDDED_OCR_VERIFIED",
        "RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER",
        "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR",
        "MANUAL_REVIEW_ACCEPTED_RERUN_OCR",
    ],
)
def test_accepted_auto_routing_warning_can_continue_to_chunking(warning: str) -> None:
    body = make_block("body", 0, BlockType.PARAGRAPH, "Reviewed OCR text")

    result = prepare_pdf_document(make_document(make_page(1, body, warnings=(warning,))))

    assert len(result.units) == 1
    assert result.units[0].warnings[0].code == warning
    assert ChunkingDiagnosticCode.UNKNOWN_PAGE_WARNING not in diagnostic_codes(result)


def test_page_warnings_remain_bound_to_blocks_from_their_own_page() -> None:
    heading = make_block("heading", 0, BlockType.HEADING, "1 Introduction")
    body = make_block(
        "body",
        0,
        BlockType.PARAGRAPH,
        "Body on the next page.",
        page_number=2,
    )

    result = prepare_pdf_document(
        make_document(
            make_page(1, heading, warnings=("EMBEDDED_OCR_TEXT_LAYER",)),
            make_page(2, body, warnings=("DOCLING_INVALID_BBOX",)),
        )
    )

    warnings_by_code = {
        warning.code: warning.source_block_ids for warning in result.units[0].warnings
    }
    assert warnings_by_code == {
        "EMBEDDED_OCR_TEXT_LAYER": ("heading",),
        "DOCLING_INVALID_BBOX": ("body",),
    }


def test_decimal_heading_inference_is_limited_to_heading_blocks() -> None:
    title = make_block("title", 0, BlockType.TITLE, "RAG Handbook")
    chapter = make_block("chapter", 1, BlockType.HEADING, "1 Chunking")
    section = make_block("section", 2, BlockType.HEADING, "1.1 Parameters")
    body = make_block("body", 3, BlockType.PARAGRAPH, "Body text.")
    citation = make_block(
        "citation",
        4,
        BlockType.PARAGRAPH,
        "See 3.2.1 Chunk parameters for details.",
    )

    result = prepare_pdf_document(
        make_document(make_page(1, title, chapter, section, body, citation))
    )

    assert result.units[0].candidate.source_block_ids == (
        "title",
        "chapter",
        "section",
        "body",
    )
    assert result.units[1].candidate.source_block_ids == ("citation",)
    assert result.units[1].candidate.section_path == (
        "RAG Handbook",
        "1 Chunking",
        "1.1 Parameters",
    )
    inferred = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.HEADING_LEVEL_INFERRED
    )
    assert inferred.source_block_ids == ("chapter", "section")


def test_parser_heading_level_wins_conflict_and_sparse_level_is_preserved() -> None:
    heading = make_block(
        "heading",
        0,
        BlockType.HEADING,
        "1 Overview",
        heading_level=3,
    )
    body = make_block("body", 1, BlockType.PARAGRAPH, "Body text.")

    result = prepare_pdf_document(make_document(make_page(1, heading, body)))

    assert result.units[0].candidate.section_path == ("1 Overview",)
    assert ChunkingDiagnosticCode.HEADING_LEVEL_CONFLICT in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.HEADING_LEVEL_GAP in diagnostic_codes(result)


def test_unknown_heading_level_defaults_to_level_one() -> None:
    heading = make_block("heading", 0, BlockType.HEADING, "Appendix")
    body = make_block("body", 1, BlockType.PARAGRAPH, "Appendix body.")

    result = prepare_pdf_document(make_document(make_page(1, heading, body)))

    assert result.units[0].candidate.section_path == ("Appendix",)
    assert ChunkingDiagnosticCode.HEADING_LEVEL_UNKNOWN in diagnostic_codes(result)


def test_additional_title_is_treated_as_level_one_heading() -> None:
    root = make_block("root", 0, BlockType.TITLE, "Document title")
    first_body = make_block("first-body", 1, BlockType.PARAGRAPH, "Opening body.")
    second_title = make_block("second-title", 2, BlockType.TITLE, "Part two")
    second_body = make_block("second-body", 3, BlockType.PARAGRAPH, "Part two body.")

    result = prepare_pdf_document(
        make_document(make_page(1, root, first_body, second_title, second_body))
    )

    assert result.units[1].candidate.source_block_ids == ("second-title", "second-body")
    assert result.units[1].candidate.section_path == ("Document title", "Part two")
    assert ChunkingDiagnosticCode.ADDITIONAL_TITLE_AS_HEADING in diagnostic_codes(result)


def test_closed_heading_without_body_is_diagnosed_and_not_attached() -> None:
    first = make_block("first", 0, BlockType.HEADING, "First", heading_level=1)
    second = make_block("second", 1, BlockType.HEADING, "Second", heading_level=1)
    body = make_block("body", 2, BlockType.PARAGRAPH, "Body text.")

    result = prepare_pdf_document(make_document(make_page(1, first, second, body)))

    assert result.units[0].candidate.source_block_ids == ("second", "body")
    heading_without_body = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.HEADING_WITHOUT_BODY
    )
    assert heading_without_body.source_block_ids == ("first",)


def test_unusable_page_resets_section_context_before_later_body() -> None:
    heading = make_block("heading", 0, BlockType.HEADING, "1 Before gap")
    later_body = make_block(
        "later-body",
        0,
        BlockType.PARAGRAPH,
        "Body after unavailable page.",
        page_number=3,
    )

    result = prepare_pdf_document(
        make_document(
            make_page(1, heading),
            make_page(
                2,
                status=ParseStatus.NEEDS_OCR,
                warnings=("OCR_REQUIRED",),
            ),
            make_page(3, later_body),
        )
    )

    assert result.units[0].candidate.source_block_ids == ("later-body",)
    assert result.units[0].candidate.section_path == ()
    assert ChunkingDiagnosticCode.PAGE_NEEDS_OCR_SKIPPED in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.SECTION_CONTEXT_RESET in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.HEADING_WITHOUT_BODY in diagnostic_codes(result)


def test_failed_page_is_excluded_with_a_distinct_diagnostic() -> None:
    result = prepare_pdf_document(
        make_document(
            make_page(
                1,
                status=ParseStatus.FAILED,
                warnings=("NO_EXTRACTABLE_TEXT",),
            )
        )
    )

    assert result.units == ()
    assert diagnostic_codes(result) == {ChunkingDiagnosticCode.PAGE_PARSE_FAILED_SKIPPED}


def test_filter_diagnostics_are_aggregated_in_source_order() -> None:
    header_1 = make_block("header-1", 0, BlockType.HEADER, "Repeated header")
    body_1 = make_block("body-1", 1, BlockType.PARAGRAPH, "Page one body")
    header_2 = make_block(
        "header-2",
        0,
        BlockType.HEADER,
        "Repeated header",
        page_number=2,
    )
    body_2 = make_block(
        "body-2",
        1,
        BlockType.PARAGRAPH,
        "Page two body",
        page_number=2,
    )

    result = prepare_pdf_document(
        make_document(make_page(1, header_1, body_1), make_page(2, header_2, body_2))
    )

    filtered = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.BLOCK_FILTERED
    )
    assert filtered.page_numbers == (1, 2)
    assert filtered.source_block_ids == ("header-1", "header-2")


def test_oversized_body_block_splits_with_exact_contiguous_offsets() -> None:
    heading = make_block("heading", 0, BlockType.HEADING, "1 简介")
    text = "第一句内容较短。第二句内容继续扩展。第三句仍然需要保留。最后一句结束。"
    body = make_block("body", 1, BlockType.PARAGRAPH, text)
    prepared = prepare_pdf_document(make_document(make_page(1, heading, body)))

    result = split_oversized_body_units(
        prepared,
        PdfChunkingConfig(target_size=16, hard_max_size=24),
    )

    body_slices = [unit.candidate.block_slices[-1] for unit in result.units]
    assert "".join(block_slice.text for block_slice in body_slices) == text
    assert [block_slice.source_text_start for block_slice in body_slices] == [
        0,
        *[block_slice.source_text_end for block_slice in body_slices[:-1]],
    ]
    assert body_slices[-1].source_text_end == len(text)
    assert result.units[0].candidate.source_block_ids == ("heading", "body")
    assert all(unit.candidate.source_block_ids == ("body",) for unit in result.units[1:])
    assert all(len(unit.candidate.content) <= 24 for unit in result.units)
    assert ChunkingDiagnosticCode.BODY_BLOCK_SPLIT in diagnostic_codes(result)


@pytest.mark.parametrize(
    ("text", "expected_first_end"),
    [
        ("abcdefghi\njklmnopqr\nstuvwxyz", 10),
        ("abcdef ghi mnopqr stuvwx", 11),
        ("x" * 35, 12),
    ],
)
def test_body_split_falls_back_from_newline_to_whitespace_then_hard_cut(
    text: str,
    expected_first_end: int,
) -> None:
    body = make_block("body", 0, BlockType.PARAGRAPH, text)
    prepared = prepare_pdf_document(make_document(make_page(1, body)))

    result = split_oversized_body_units(
        prepared,
        PdfChunkingConfig(target_size=10, hard_max_size=12),
    )

    assert result.units[0].candidate.block_slices[0].source_text_end == expected_first_end
    assert "".join(unit.candidate.content for unit in result.units) == text
    assert all(len(unit.candidate.content) <= 12 for unit in result.units)


def test_oversized_special_content_is_left_for_its_own_strategy() -> None:
    table = make_block("table", 0, BlockType.TABLE, "cell data\n" * 20)
    prepared = prepare_pdf_document(make_document(make_page(1, table)))

    result = split_oversized_body_units(
        prepared,
        PdfChunkingConfig(target_size=20, hard_max_size=30),
    )

    assert result.units == prepared.units
    assert ChunkingDiagnosticCode.BODY_BLOCK_SPLIT not in diagnostic_codes(result)


def test_heading_prefix_over_hard_max_is_visible_but_still_attaches_once() -> None:
    heading = make_block(
        "heading",
        0,
        BlockType.HEADING,
        "A heading that is longer than the configured hard maximum",
        heading_level=1,
    )
    body = make_block("body", 1, BlockType.PARAGRAPH, "Body text")
    prepared = prepare_pdf_document(make_document(make_page(1, heading, body)))

    result = split_oversized_body_units(
        prepared,
        PdfChunkingConfig(target_size=15, hard_max_size=20),
    )

    assert result.units[0].candidate.source_block_ids == ("heading", "body")
    assert all(unit.candidate.source_block_ids == ("body",) for unit in result.units[1:])
    assert ChunkingDiagnosticCode.HEADING_PREFIX_EXCEEDS_HARD_MAX in diagnostic_codes(result)


def test_same_page_body_units_combine_until_target_boundary() -> None:
    blocks = tuple(
        make_block(f"body-{index}", index, BlockType.PARAGRAPH, str(index) * 8)
        for index in range(1, 5)
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=0)
    prepared = split_oversized_body_units(
        prepare_pdf_document(make_document(make_page(1, *blocks))),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("body-1", "body-2", "body-3"),
        ("body-4",),
    ]


def test_hard_max_flushes_body_before_target_can_be_reached() -> None:
    first = make_block("first", 0, BlockType.PARAGRAPH, "a" * 15)
    second = make_block("second", 1, BlockType.PARAGRAPH, "b" * 15)
    config = PdfChunkingConfig(target_size=20, hard_max_size=25, overlap_percent=0)
    prepared = split_oversized_body_units(
        prepare_pdf_document(make_document(make_page(1, first, second))),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("first",),
        ("second",),
    ]


def test_length_boundary_repeats_at_most_one_complete_body_unit() -> None:
    blocks = tuple(
        make_block(f"body-{index}", index, BlockType.PARAGRAPH, str(index) * 9)
        for index in range(1, 5)
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=50)
    prepared = split_oversized_body_units(
        prepare_pdf_document(make_document(make_page(1, *blocks))),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("body-1", "body-2", "body-3"),
        ("body-3", "body-4"),
    ]
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED in diagnostic_codes(result)


def test_page_section_and_content_type_boundaries_do_not_create_overlap() -> None:
    page_one_body = make_block("page-1-body", 0, BlockType.PARAGRAPH, "page one")
    table = make_block("table", 1, BlockType.TABLE, "a | b")
    heading = make_block("heading", 2, BlockType.HEADING, "2 Next", heading_level=1)
    section_body = make_block("section-body", 3, BlockType.PARAGRAPH, "new section")
    page_two_body = make_block(
        "page-2-body",
        0,
        BlockType.PARAGRAPH,
        "page two",
        page_number=2,
    )
    config = PdfChunkingConfig(target_size=30, hard_max_size=40, overlap_percent=50)
    prepared = split_oversized_body_units(
        prepare_pdf_document(
            make_document(
                make_page(1, page_one_body, table, heading, section_body),
                make_page(2, page_two_body),
            )
        ),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("page-1-body",),
        ("table",),
        ("heading", "section-body"),
        ("page-2-body",),
    ]
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_slices_from_one_oversized_block_remain_atomic_during_composition() -> None:
    body = make_block("body", 0, BlockType.PARAGRAPH, "x" * 35)
    config = PdfChunkingConfig(target_size=10, hard_max_size=12, overlap_percent=50)
    prepared = split_oversized_body_units(
        prepare_pdf_document(make_document(make_page(1, body))),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [
        (
            unit.candidate.block_slices[0].source_text_start,
            unit.candidate.block_slices[0].source_text_end,
        )
        for unit in result.units
    ] == [(0, 12), (12, 24), (24, 35)]
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_repeated_heading_text_is_still_a_same_page_hard_boundary() -> None:
    first_heading = make_block("heading-1", 0, BlockType.HEADING, "Repeated", heading_level=1)
    first_body = make_block("body-1", 1, BlockType.PARAGRAPH, "first body")
    second_heading = make_block("heading-2", 2, BlockType.HEADING, "Repeated", heading_level=1)
    second_body = make_block("body-2", 3, BlockType.PARAGRAPH, "second body")
    config = PdfChunkingConfig(target_size=50, hard_max_size=80, overlap_percent=0)
    prepared = split_oversized_body_units(
        prepare_pdf_document(
            make_document(make_page(1, first_heading, first_body, second_heading, second_body))
        ),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("heading-1", "body-1"),
        ("heading-2", "body-2"),
    ]
    assert result.units[0].candidate.section_path == result.units[1].candidate.section_path


def test_small_body_units_can_combine_across_multiple_adjacent_pages() -> None:
    pages = tuple(
        make_page(
            page_number,
            make_block(
                f"body-{page_number}",
                0,
                BlockType.PARAGRAPH,
                str(page_number) * 5,
                page_number=page_number,
            ),
        )
        for page_number in range(1, 4)
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=0)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(make_document(*pages)),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("body-1", "body-2", "body-3")
    ]
    assert result.units[0].candidate.page_start == 1
    assert result.units[0].candidate.page_end == 3
    assert ChunkingDiagnosticCode.CROSS_PAGE_BODY_MERGED in diagnostic_codes(result)


def test_repeated_heading_text_is_still_a_cross_page_hard_boundary() -> None:
    first_heading = make_block("heading-1", 0, BlockType.HEADING, "Repeated", heading_level=1)
    first_body = make_block("body-1", 1, BlockType.PARAGRAPH, "first body")
    second_heading = make_block(
        "heading-2",
        0,
        BlockType.HEADING,
        "Repeated",
        page_number=2,
        heading_level=1,
    )
    second_body = make_block(
        "body-2",
        1,
        BlockType.PARAGRAPH,
        "second body",
        page_number=2,
    )
    config = PdfChunkingConfig(target_size=50, hard_max_size=80, overlap_percent=0)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(
                make_document(
                    make_page(1, first_heading, first_body),
                    make_page(2, second_heading, second_body),
                )
            ),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("heading-1", "body-1"),
        ("heading-2", "body-2"),
    ]
    assert result.units[0].candidate.section_path == result.units[1].candidate.section_path


def test_page_boundary_stays_closed_after_target_is_reached() -> None:
    page_one = make_block("page-1", 0, BlockType.PARAGRAPH, "a" * 20)
    page_two = make_block(
        "page-2",
        0,
        BlockType.PARAGRAPH,
        "b" * 5,
        page_number=2,
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=50)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(make_document(make_page(1, page_one), make_page(2, page_two))),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("page-1",),
        ("page-2",),
    ]
    assert ChunkingDiagnosticCode.CROSS_PAGE_BODY_MERGED not in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_empty_physical_page_prevents_cross_page_combination() -> None:
    page_one = make_block("page-1", 0, BlockType.PARAGRAPH, "page one")
    page_three = make_block(
        "page-3",
        0,
        BlockType.PARAGRAPH,
        "page three",
        page_number=3,
    )
    config = PdfChunkingConfig(target_size=30, hard_max_size=40, overlap_percent=0)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(
                make_document(
                    make_page(1, page_one),
                    make_page(2),
                    make_page(3, page_three),
                )
            ),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("page-1",),
        ("page-3",),
    ]
    assert ChunkingDiagnosticCode.CROSS_PAGE_BODY_MERGED not in diagnostic_codes(result)


def test_cross_page_text_preserves_hyphen_and_reports_extraction_change() -> None:
    left = make_block(
        "left",
        0,
        BlockType.PARAGRAPH,
        "multi-",
        extraction_method=ExtractionMethod.NATIVE,
    )
    right = make_block(
        "right",
        0,
        BlockType.PARAGRAPH,
        "modal",
        page_number=2,
        extraction_method=ExtractionMethod.RERUN_OCR,
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=0)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(make_document(make_page(1, left), make_page(2, right))),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert result.units[0].candidate.content == "multi-\nmodal"
    assert ChunkingDiagnosticCode.CROSS_PAGE_HYPHEN_PRESERVED in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.CROSS_PAGE_EXTRACTION_METHOD_CHANGED in diagnostic_codes(result)


def test_cross_page_length_boundary_can_overlap_one_complete_body_unit() -> None:
    first = make_block("first", 0, BlockType.PARAGRAPH, "a" * 8)
    overlap = make_block("overlap", 1, BlockType.LIST_ITEM, "b" * 9)
    following = make_block(
        "following",
        0,
        BlockType.PARAGRAPH,
        "c" * 15,
        page_number=2,
    )
    config = PdfChunkingConfig(target_size=20, hard_max_size=25, overlap_percent=50)
    prepared = compose_same_page_body_units(
        split_oversized_body_units(
            prepare_pdf_document(
                make_document(make_page(1, first, overlap), make_page(2, following))
            ),
            config,
        ),
        config,
    )

    result = compose_cross_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("first", "overlap"),
        ("overlap", "following"),
    ]
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.CROSS_PAGE_BODY_MERGED not in diagnostic_codes(result)


def test_table_at_or_below_hard_max_remains_whole() -> None:
    table = make_block("table", 0, BlockType.TABLE, "name | value\nalpha | 1")
    prepared = prepare_pdf_document(make_document(make_page(1, table)))

    result = split_oversized_table_units(
        prepared,
        PdfChunkingConfig(target_size=20, hard_max_size=30),
    )

    assert result.units == prepared.units
    assert ChunkingDiagnosticCode.TABLE_BLOCK_SPLIT not in diagnostic_codes(result)


def test_oversized_table_splits_on_rows_without_repeating_header() -> None:
    text = "name | value\nalpha | 1\nbeta | 2\ngamma | 3\n"
    table = make_block("table", 0, BlockType.TABLE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, table)))

    result = split_oversized_table_units(
        prepared,
        PdfChunkingConfig(target_size=20, hard_max_size=25),
    )

    assert "".join(unit.candidate.content for unit in result.units) == text
    assert all(len(unit.candidate.content) <= 25 for unit in result.units)
    assert sum(unit.candidate.content.count("name | value") for unit in result.units) == 1
    assert result.units[0].candidate.content == "name | value\nalpha | 1\n"
    assert all(unit.content_type is ChunkContentType.TABLE for unit in result.units)
    assert ChunkingDiagnosticCode.TABLE_BLOCK_SPLIT in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.TABLE_ROW_SPLIT not in diagnostic_codes(result)


def test_oversized_table_row_falls_back_to_whitespace() -> None:
    text = "header\naa bb cc dd ee ff"
    table = make_block("table", 0, BlockType.TABLE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, table)))

    result = split_oversized_table_units(
        prepared,
        PdfChunkingConfig(target_size=10, hard_max_size=12),
    )

    assert "".join(unit.candidate.content for unit in result.units) == text
    assert result.units[0].candidate.content == "header\n"
    assert all(len(unit.candidate.content) <= 12 for unit in result.units)
    assert ChunkingDiagnosticCode.TABLE_ROW_SPLIT in diagnostic_codes(result)


def test_oversized_table_row_without_whitespace_uses_hard_cut() -> None:
    text = "x" * 25
    table = make_block("table", 0, BlockType.TABLE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, table)))

    result = split_oversized_table_units(
        prepared,
        PdfChunkingConfig(target_size=10, hard_max_size=12),
    )

    assert [len(unit.candidate.content) for unit in result.units] == [12, 12, 1]
    assert "".join(unit.candidate.content for unit in result.units) == text
    assert ChunkingDiagnosticCode.TABLE_ROW_SPLIT in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_formula_at_or_below_hard_max_remains_atomic_without_diagnostic() -> None:
    formula = make_block("formula", 0, BlockType.FORMULA, "E = mc^2")
    prepared = prepare_pdf_document(make_document(make_page(1, formula)))

    result = preserve_atomic_formula_units(
        prepared,
        PdfChunkingConfig(target_size=10, hard_max_size=12),
    )

    assert result.units == prepared.units
    assert result.units[0].content_type is ChunkContentType.FORMULA
    assert ChunkingDiagnosticCode.FORMULA_EXCEEDS_HARD_MAX not in diagnostic_codes(result)


def test_oversized_formula_remains_atomic_and_reports_hard_max_exception() -> None:
    text = r"\begin{aligned}a &= b + c \\ d &= e + f\end{aligned}"
    formula = make_block("formula", 0, BlockType.FORMULA, text)
    prepared = prepare_pdf_document(make_document(make_page(1, formula)))

    result = preserve_atomic_formula_units(
        prepared,
        PdfChunkingConfig(target_size=10, hard_max_size=20, overlap_percent=50),
    )

    assert len(result.units) == 1
    assert result.units[0].candidate.content == text
    assert result.units[0].candidate.source_block_ids == ("formula",)
    formula_slice = result.units[0].candidate.block_slices[0]
    assert formula_slice.source_text_start == 0
    assert formula_slice.source_text_end == len(text)
    diagnostic = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.FORMULA_EXCEEDS_HARD_MAX
    )
    assert diagnostic.severity is WarningSeverity.WARNING
    assert diagnostic.page_numbers == (1,)
    assert diagnostic.source_block_ids == ("formula",)
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_linked_figure_caption_remains_atomic_with_picture_relationship() -> None:
    caption = make_block(
        "caption",
        0,
        BlockType.FIGURE_CAPTION,
        "Figure 1. Structure-aware PDF chunking",
        parent_picture_id="picture-1",
    )
    picture = ParsedPicture(
        id="picture-1",
        page_number=1,
        bbox=BoundingBox(50, 40, 520, 300),
        child_block_ids=("caption",),
    )
    prepared = prepare_pdf_document(make_document(make_page(1, caption, pictures=(picture,))))

    result = preserve_atomic_figure_caption_units(prepared, PdfChunkingConfig())

    assert result.units == prepared.units
    assert result.units[0].content_type is ChunkContentType.FIGURE_CAPTION
    caption_slice = result.units[0].candidate.block_slices[0]
    assert caption_slice.block.parent_picture_id == "picture-1"
    assert ChunkingDiagnosticCode.FIGURE_CAPTION_PARENT_MISSING not in diagnostic_codes(result)


def test_unlinked_figure_caption_is_kept_with_visible_diagnostic() -> None:
    caption = make_block(
        "caption",
        0,
        BlockType.FIGURE_CAPTION,
        "Figure 1. Parent was not identified",
    )
    prepared = prepare_pdf_document(make_document(make_page(1, caption)))

    result = preserve_atomic_figure_caption_units(prepared, PdfChunkingConfig())

    assert result.units == prepared.units
    diagnostic = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is ChunkingDiagnosticCode.FIGURE_CAPTION_PARENT_MISSING
    )
    assert diagnostic.severity is WarningSeverity.WARNING
    assert diagnostic.page_numbers == (1,)
    assert diagnostic.source_block_ids == ("caption",)


def test_oversized_figure_caption_remains_atomic_with_exact_source_offsets() -> None:
    text = "Figure 2. " + "A detailed caption. " * 5
    caption = make_block(
        "caption",
        0,
        BlockType.FIGURE_CAPTION,
        text,
        parent_picture_id="picture-2",
    )
    picture = ParsedPicture(
        id="picture-2",
        page_number=1,
        bbox=BoundingBox(50, 40, 520, 300),
        child_block_ids=("caption",),
    )
    prepared = prepare_pdf_document(make_document(make_page(1, caption, pictures=(picture,))))

    result = preserve_atomic_figure_caption_units(
        prepared,
        PdfChunkingConfig(target_size=20, hard_max_size=30, overlap_percent=50),
    )

    assert len(result.units) == 1
    assert result.units[0].candidate.content == text
    caption_slice = result.units[0].candidate.block_slices[0]
    assert caption_slice.source_text_start == 0
    assert caption_slice.source_text_end == len(text)
    assert caption_slice.block.parent_picture_id == "picture-2"
    assert ChunkingDiagnosticCode.FIGURE_CAPTION_EXCEEDS_HARD_MAX in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_code_at_or_below_hard_max_remains_whole() -> None:
    code = make_block("code", 0, BlockType.CODE, "value = compute()")
    prepared = prepare_pdf_document(make_document(make_page(1, code)))

    result = split_oversized_code_units(
        prepared,
        PdfChunkingConfig(target_size=15, hard_max_size=20),
    )

    assert result.units == prepared.units
    assert result.units[0].content_type is ChunkContentType.CODE
    assert ChunkingDiagnosticCode.CODE_BLOCK_SPLIT not in diagnostic_codes(result)


def test_oversized_code_prefers_top_level_definition_boundary() -> None:
    text = "def first():\n    return 1\n\ndef second():\n    return 2\n"
    code = make_block("code", 0, BlockType.CODE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, code)))

    result = split_oversized_code_units(
        prepared,
        PdfChunkingConfig(target_size=28, hard_max_size=40),
    )

    assert len(result.units) == 2
    assert result.units[0].candidate.content == "def first():\n    return 1\n\n"
    assert result.units[1].candidate.content == "def second():\n    return 2\n"
    assert "".join(unit.candidate.content for unit in result.units) == text
    assert all(unit.content_type is ChunkContentType.CODE for unit in result.units)
    assert ChunkingDiagnosticCode.CODE_BLOCK_SPLIT in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.CODE_LINE_SPLIT not in diagnostic_codes(result)


def test_oversized_code_preserves_indentation_and_exact_source_offsets() -> None:
    text = "if ready:\n    first_step()\n    second_step()\n    third_step()\n"
    code = make_block("code", 0, BlockType.CODE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, code)))

    result = split_oversized_code_units(
        prepared,
        PdfChunkingConfig(target_size=25, hard_max_size=32),
    )

    assert "".join(unit.candidate.content for unit in result.units) == text
    assert result.units[1].candidate.content.startswith("    second_step()")
    offsets = [
        (
            unit.candidate.block_slices[0].source_text_start,
            unit.candidate.block_slices[0].source_text_end,
        )
        for unit in result.units
    ]
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    assert all(
        left_end == right_start
        for (_, left_end), (right_start, _) in zip(offsets, offsets[1:], strict=False)
    )


def test_code_line_over_hard_max_uses_fallback_and_reports_split() -> None:
    text = "result = " + "x" * 35
    code = make_block("code", 0, BlockType.CODE, text)
    prepared = prepare_pdf_document(make_document(make_page(1, code)))

    result = split_oversized_code_units(
        prepared,
        PdfChunkingConfig(target_size=20, hard_max_size=25, overlap_percent=50),
    )

    assert "".join(unit.candidate.content for unit in result.units) == text
    assert all(len(unit.candidate.content) <= 25 for unit in result.units)
    assert ChunkingDiagnosticCode.CODE_LINE_SPLIT in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED not in diagnostic_codes(result)


def test_other_unit_is_a_hard_boundary_between_body_units() -> None:
    before = make_block("before", 0, BlockType.PARAGRAPH, "Body before OTHER.")
    other = make_block("other", 1, BlockType.OTHER, "Unknown parser content")
    after = make_block("after", 2, BlockType.PARAGRAPH, "Body after OTHER.")
    config = PdfChunkingConfig(target_size=100, hard_max_size=120)
    prepared = preserve_isolated_other_units(
        prepare_pdf_document(make_document(make_page(1, before, other, after))),
        config,
    )

    result = compose_same_page_body_units(prepared, config)

    assert [unit.candidate.source_block_ids for unit in result.units] == [
        ("before",),
        ("other",),
        ("after",),
    ]
    assert [unit.content_type for unit in result.units] == [
        ChunkContentType.BODY_TEXT,
        ChunkContentType.OTHER,
        ChunkContentType.BODY_TEXT,
    ]
    assert ChunkingDiagnosticCode.UNSUPPORTED_CONTENT_TYPE in diagnostic_codes(result)


def test_oversized_other_remains_isolated_with_visible_diagnostic() -> None:
    text = "unknown:" + "x" * 40
    other = make_block("other", 0, BlockType.OTHER, text)
    prepared = prepare_pdf_document(make_document(make_page(1, other)))

    result = preserve_isolated_other_units(
        prepared,
        PdfChunkingConfig(target_size=15, hard_max_size=20),
    )

    assert len(result.units) == 1
    assert result.units[0].content_type is ChunkContentType.OTHER
    assert result.units[0].candidate.content == text
    other_slice = result.units[0].candidate.block_slices[0]
    assert other_slice.source_text_start == 0
    assert other_slice.source_text_end == len(text)
    assert ChunkingDiagnosticCode.UNSUPPORTED_CONTENT_TYPE in diagnostic_codes(result)
    assert ChunkingDiagnosticCode.OTHER_EXCEEDS_HARD_MAX in diagnostic_codes(result)


def test_top_level_chunker_preserves_mixed_content_order_and_traceability() -> None:
    title = make_block("title", 0, BlockType.TITLE, "RAG Guide")
    body = make_block("body", 1, BlockType.PARAGRAPH, "Body explanation.")
    table = make_block("table", 2, BlockType.TABLE, "name | value\nalpha | 1")
    formula = make_block("formula", 3, BlockType.FORMULA, "E = mc^2")
    caption = make_block(
        "caption",
        4,
        BlockType.FIGURE_CAPTION,
        "Figure 1. Pipeline",
        parent_picture_id="picture-1",
    )
    code = make_block("code", 5, BlockType.CODE, "result = chunk(document)")
    other = make_block("other", 6, BlockType.OTHER, "Custom parser node")
    trailing_body = make_block("body-2", 7, BlockType.PARAGRAPH, "Trailing explanation.")
    picture = ParsedPicture(
        id="picture-1",
        page_number=1,
        bbox=BoundingBox(50, 100, 520, 350),
        child_block_ids=("caption",),
    )
    document = make_document(
        make_page(
            1,
            title,
            body,
            table,
            formula,
            caption,
            code,
            other,
            trailing_body,
            pictures=(picture,),
        )
    )
    config = PdfChunkingConfig(target_size=100, hard_max_size=120)

    result = chunk_pdf_document(document, config)

    assert [chunk.chunk_index for chunk in result.chunks] == list(range(7))
    assert [chunk.content_type for chunk in result.chunks] == [
        ChunkContentType.BODY_TEXT,
        ChunkContentType.TABLE,
        ChunkContentType.FORMULA,
        ChunkContentType.FIGURE_CAPTION,
        ChunkContentType.CODE,
        ChunkContentType.OTHER,
        ChunkContentType.BODY_TEXT,
    ]
    assert [chunk.source_block_ids for chunk in result.chunks] == [
        ("title", "body"),
        ("table",),
        ("formula",),
        ("caption",),
        ("code",),
        ("other",),
        ("body-2",),
    ]
    assert all(chunk.chunking_profile == config.chunking_profile for chunk in result.chunks)
    assert result.chunks[3].source_regions[0].parent_picture_id == "picture-1"
    assert result.status is PdfChunkingStatus.COMPLETED


def test_top_level_chunker_is_deterministic_for_same_document_and_config() -> None:
    body = make_block("body", 0, BlockType.PARAGRAPH, "Stable body content.")
    document = make_document(make_page(1, body))
    config = PdfChunkingConfig(target_size=40, hard_max_size=60)

    first = chunk_pdf_document(document, config)
    second = chunk_pdf_document(document, config)

    assert first == second
    assert first.chunks[0].id == second.chunks[0].id
    assert first.chunks[0].source_fingerprint == second.chunks[0].source_fingerprint


def test_top_level_chunker_reports_failed_result_when_every_page_is_blocked() -> None:
    blocked_body = make_block("body", 0, BlockType.PARAGRAPH, "Unavailable text")
    document = make_document(
        make_page(1, blocked_body, status=ParseStatus.NEEDS_OCR),
    )

    result = chunk_pdf_document(document)

    assert result.chunks == ()
    assert result.status is PdfChunkingStatus.FAILED
    assert ChunkingDiagnosticCode.PAGE_NEEDS_OCR_SKIPPED in {
        diagnostic.code for diagnostic in result.diagnostics
    }
    assert result.diagnostics[-1].code is ChunkingDiagnosticCode.NO_CHUNKS_PRODUCED

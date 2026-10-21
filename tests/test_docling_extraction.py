"""Tests for mapping Docling output into the shared PDF schema."""

from ai_learning_rag.docling_extraction import parsed_document_from_docling
from ai_learning_rag.pdf_schema import BlockType, ExtractionMethod, PageKind, ParseStatus


def _provenance(page: int, *, left: float, top: float, right: float, bottom: float) -> list:
    return [
        {
            "page_no": page,
            "bbox": {
                "l": left,
                "t": top,
                "r": right,
                "b": bottom,
                "coord_origin": "BOTTOMLEFT",
            },
            "charspan": [0, 1000],
        }
    ]


def _docling_fixture() -> dict:
    texts = [
        {
            "self_ref": "#/texts/0",
            "label": "page_header",
            "text": "Journal header",
            "prov": _provenance(1, left=40, top=780, right=160, bottom=760),
            "children": [],
        },
        {
            "self_ref": "#/texts/1",
            "label": "section_header",
            "text": "1. Introduction",
            "prov": _provenance(1, left=40, top=700, right=180, bottom=680),
            "children": [],
        },
        {
            "self_ref": "#/texts/2",
            "label": "text",
            "text": "First paragraph",
            "prov": _provenance(1, left=40, top=650, right=300, bottom=600),
            "children": [],
        },
        {
            "self_ref": "#/texts/3",
            "label": "list_item",
            "text": "A list item",
            "prov": _provenance(1, left=50, top=580, right=250, bottom=560),
            "children": [],
        },
        {
            "self_ref": "#/texts/4",
            "label": "caption",
            "text": "Fig. 1. Architecture",
            "prov": _provenance(2, left=40, top=300, right=240, bottom=280),
            "children": [],
        },
        {
            "self_ref": "#/texts/5",
            "label": "page_footer",
            "text": "2",
            "prov": _provenance(2, left=295, top=30, right=300, bottom=20),
            "children": [],
        },
        {
            "self_ref": "#/texts/6",
            "label": "text",
            "text": "Diagram node",
            "prov": _provenance(2, left=100, top=600, right=200, bottom=570),
            "children": [],
        },
        {
            "self_ref": "#/texts/7",
            "label": "code",
            "text": "def answer():\n    return 42",
            "prov": _provenance(1, left=40, top=540, right=300, bottom=490),
            "children": [],
        },
    ]
    return {
        "pages": {
            "1": {"page_no": 1, "size": {"width": 600, "height": 800}},
            "2": {"page_no": 2, "size": {"width": 600, "height": 800}},
        },
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/groups/0"},
                {"$ref": "#/texts/7"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/5"},
            ]
        },
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "group",
                "children": [{"$ref": "#/texts/2"}, {"$ref": "#/texts/3"}],
            }
        ],
        "texts": texts,
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "children": [{"$ref": "#/texts/4"}, {"$ref": "#/texts/6"}],
                "prov": _provenance(2, left=40, top=700, right=560, bottom=320),
            }
        ],
        "tables": [],
    }


def test_docling_labels_reading_order_and_coordinates_map_to_shared_schema() -> None:
    document = parsed_document_from_docling(
        _docling_fixture(),
        source="data/PDF/layout.pdf",
        file_sha256="a" * 64,
        parser_version="test",
    )

    assert document.parser_name == "docling"
    assert [block.block_type for block in document.pages[0].blocks] == [
        BlockType.HEADER,
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.LIST_ITEM,
        BlockType.CODE,
    ]
    assert [block.reading_order for block in document.pages[0].blocks] == [0, 1, 2, 3, 4]
    assert document.pages[0].blocks[-1].text == "def answer():\n    return 42"
    assert document.pages[0].blocks[0].bbox.as_tuple() == (40, 20, 160, 40)
    assert [block.block_type for block in document.pages[1].blocks] == [
        BlockType.FIGURE_CAPTION,
        BlockType.PICTURE_TEXT,
        BlockType.FOOTER,
    ]
    assert document.pages[1].image_count == 1
    assert len(document.pages[1].pictures) == 1
    picture = document.pages[1].pictures[0]
    assert picture.bbox.as_tuple() == (40, 100, 560, 480)
    assert picture.child_block_ids == tuple(block.id for block in document.pages[1].blocks[:2])
    assert document.pages[1].blocks[0].parent_picture_id == picture.id
    assert document.pages[1].blocks[1].parent_picture_id == picture.id
    assert "PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS" in document.pages[1].blocks[1].warnings


def test_docling_table_cells_become_one_traceable_table_block() -> None:
    fixture = _docling_fixture()
    fixture["tables"] = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "children": [],
            "prov": _provenance(2, left=40, top=700, right=560, bottom=500),
            "data": {
                "num_rows": 2,
                "num_cols": 2,
                "table_cells": [
                    {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "text": "A"},
                    {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "text": "B"},
                    {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "text": "1"},
                    {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "2"},
                ],
            },
        }
    ]
    fixture["body"]["children"].insert(3, {"$ref": "#/tables/0"})

    document = parsed_document_from_docling(
        fixture,
        source="data/PDF/layout.pdf",
        file_sha256="b" * 64,
        parser_version="test",
    )

    table = next(block for block in document.pages[1].blocks if block.block_type is BlockType.TABLE)
    assert table.text == "A\tB\n1\t2"


def test_image_only_page_requires_ocr_instead_of_succeeding_with_empty_text() -> None:
    fixture = _docling_fixture()
    fixture["body"]["children"] = []
    fixture["texts"] = []

    document = parsed_document_from_docling(
        fixture,
        source="data/PDF/scan.pdf",
        file_sha256="c" * 64,
        parser_version="test",
    )

    assert document.pages[0].status is ParseStatus.FAILED
    assert document.pages[1].kind is PageKind.SCANNED_IMAGE
    assert document.pages[1].status is ParseStatus.NEEDS_OCR
    assert "PAGE_2_OCR_REQUIRED" in document.warnings


def test_full_page_ocr_is_marked_as_rerun_ocr_on_scanned_pages() -> None:
    document = parsed_document_from_docling(
        _docling_fixture(),
        source="data/PDF/scan.pdf",
        file_sha256="d" * 64,
        parser_version="test",
        ocr_was_enabled=True,
        full_page_ocr_was_enabled=True,
    )

    assert all(page.kind is PageKind.SCANNED_WITH_OCR for page in document.pages)
    assert all(page.status is ParseStatus.PARSED for page in document.pages)
    assert all(
        block.extraction_method is ExtractionMethod.RERUN_OCR
        for page in document.pages
        for block in page.blocks
    )

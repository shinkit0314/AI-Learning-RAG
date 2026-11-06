"""Docling layout extraction adapted to the parser-independent PDF schema."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ExtractionMethod,
    PageKind,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParsedPicture,
    ParseStatus,
)

DOCLING_LABEL_MAP = {
    "title": BlockType.TITLE,
    "section_header": BlockType.HEADING,
    "text": BlockType.PARAGRAPH,
    "list_item": BlockType.LIST_ITEM,
    "caption": BlockType.FIGURE_CAPTION,
    "formula": BlockType.FORMULA,
    "table": BlockType.TABLE,
    "page_header": BlockType.HEADER,
    "page_footer": BlockType.FOOTER,
    "footnote": BlockType.OTHER,
    "code": BlockType.CODE,
}


def create_docling_converter(
    *,
    do_ocr: bool = False,
    force_full_page_ocr: bool = False,
    ocr_language: str = "ch",
) -> Any:
    """Create a reusable standard PDF converter with an explicit OCR policy."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if force_full_page_ocr and not do_ocr:
        raise ValueError("force_full_page_ocr requires do_ocr=True")
    if do_ocr and not ocr_language.strip():
        raise ValueError("ocr_language must not be empty when OCR is enabled")

    ocr_options = RapidOcrOptions(
        mode=OcrMode.FULL_PAGE if force_full_page_ocr else OcrMode.DEFAULT,
        lang=[ocr_language],
    )
    pipeline_options = PdfPipelineOptions(do_ocr=do_ocr, ocr_options=ocr_options)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def parse_pdf_with_docling(
    source: str | Path,
    *,
    do_ocr: bool = False,
    force_full_page_ocr: bool = False,
    ocr_language: str = "ch",
    converter: Any | None = None,
) -> ParsedDocument:
    """Parse a PDF with Docling and return the shared traceable representation."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF file")

    if force_full_page_ocr and not do_ocr:
        raise ValueError("force_full_page_ocr requires do_ocr=True")
    active_converter = converter or create_docling_converter(
        do_ocr=do_ocr,
        force_full_page_ocr=force_full_page_ocr,
        ocr_language=ocr_language,
    )
    result = active_converter.convert(source_path)
    status_value = getattr(result.status, "value", str(result.status))
    document_warnings = ()
    if status_value not in {"success", "partial_success"}:
        document_warnings = (f"DOCLING_STATUS_{status_value.upper()}",)

    return parsed_document_from_docling(
        result.document.export_to_dict(),
        source=source_path,
        file_sha256=_file_sha256(source_path),
        parser_version=version("docling"),
        document_warnings=document_warnings,
        ocr_was_enabled=do_ocr,
        full_page_ocr_was_enabled=force_full_page_ocr,
    )


def parse_pdf_page_range_with_docling(
    source: str | Path,
    *,
    page_range: tuple[int, int],
    do_ocr: bool = False,
    force_full_page_ocr: bool = False,
    ocr_language: str = "ch",
    converter: Any | None = None,
    file_sha256: str | None = None,
) -> tuple[ParsedPage, ...]:
    """Parse one original PDF page range while preserving original page numbers."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("source must be a PDF file")
    start_page, end_page = page_range
    if start_page < 1 or end_page < start_page:
        raise ValueError("page_range must contain positive ordered page numbers")
    if force_full_page_ocr and not do_ocr:
        raise ValueError("force_full_page_ocr requires do_ocr=True")

    active_converter = converter or create_docling_converter(
        do_ocr=do_ocr,
        force_full_page_ocr=force_full_page_ocr,
        ocr_language=ocr_language,
    )
    result = active_converter.convert(source_path, page_range=page_range)
    pages = parsed_pages_from_docling(
        result.document.export_to_dict(),
        file_sha256=file_sha256 or _file_sha256(source_path),
        expected_page_numbers=tuple(range(start_page, end_page + 1)),
        ocr_was_enabled=do_ocr,
        full_page_ocr_was_enabled=force_full_page_ocr,
    )
    status_value = getattr(result.status, "value", str(result.status))
    if status_value in {"success", "partial_success"}:
        return pages
    warning = f"DOCLING_STATUS_{status_value.upper()}"
    return tuple(replace(page, warnings=page.warnings + (warning,)) for page in pages)


def parsed_document_from_docling(
    docling_document: Mapping[str, Any],
    *,
    source: str | Path,
    file_sha256: str,
    parser_version: str,
    document_warnings: tuple[str, ...] = (),
    ocr_was_enabled: bool = False,
    full_page_ocr_was_enabled: bool = False,
) -> ParsedDocument:
    """Map an exported Docling document without running model inference."""
    source_path = Path(source).resolve()
    pages = parsed_pages_from_docling(
        docling_document,
        file_sha256=file_sha256,
        ocr_was_enabled=ocr_was_enabled,
        full_page_ocr_was_enabled=full_page_ocr_was_enabled,
    )
    all_warnings = list(document_warnings)
    for page in pages:
        all_warnings.extend(f"PAGE_{page.page_number}_{warning}" for warning in page.warnings)

    return ParsedDocument(
        document_id=f"pdf-{file_sha256[:16]}",
        source=source_path.as_uri(),
        file_sha256=file_sha256,
        parser_name="docling",
        parser_version=parser_version,
        pages=pages,
        warnings=tuple(all_warnings),
    )


def parsed_pages_from_docling(
    docling_document: Mapping[str, Any],
    *,
    file_sha256: str,
    expected_page_numbers: tuple[int, ...] | None = None,
    ocr_was_enabled: bool = False,
    full_page_ocr_was_enabled: bool = False,
) -> tuple[ParsedPage, ...]:
    """Map a full document or one Docling page range to original PDF page numbers."""
    raw_page_specs = _page_specs(docling_document, require_start_at_one=False)
    raw_page_numbers = tuple(sorted(raw_page_specs))
    if expected_page_numbers is None:
        expected_page_numbers = tuple(range(1, len(raw_page_numbers) + 1))
        if raw_page_numbers != expected_page_numbers:
            raise ValueError("Docling pages must be contiguous and ordered from 1")
    if len(raw_page_numbers) != len(expected_page_numbers):
        raise ValueError("Docling page range did not return every requested page")
    if expected_page_numbers != tuple(
        range(expected_page_numbers[0], expected_page_numbers[-1] + 1)
    ):
        raise ValueError("expected Docling page numbers must form one contiguous range")
    if raw_page_numbers == expected_page_numbers:
        page_number_map = {number: number for number in raw_page_numbers}
    elif raw_page_numbers == tuple(range(1, len(raw_page_numbers) + 1)):
        page_number_map = dict(zip(raw_page_numbers, expected_page_numbers, strict=True))
    else:
        raise ValueError("Docling page numbers do not match the requested range")

    page_specs = {
        page_number_map[raw_page_number]: dimensions
        for raw_page_number, dimensions in raw_page_specs.items()
    }
    item_index = _item_index(docling_document)
    page_blocks: dict[int, list[ParsedBlock]] = defaultdict(list)
    page_warnings: dict[int, list[str]] = defaultdict(list)
    picture_ids, page_picture_specs = _picture_specs(
        docling_document,
        page_specs,
        file_sha256,
        page_number_map,
    )
    picture_child_ids: dict[str, list[str]] = defaultdict(list)

    for item, parent_picture_ref in _iter_reading_order_items(docling_document, item_index):
        label = str(item.get("label", "unspecified"))
        text = _item_text(item, label)
        if not text:
            continue

        block_type = DOCLING_LABEL_MAP.get(label, BlockType.OTHER)
        block_warnings = []
        if label not in DOCLING_LABEL_MAP:
            block_warnings.append(f"DOCLING_UNMAPPED_LABEL_{label.upper()}")

        for provenance in item.get("prov", ()):
            raw_page_number = int(provenance["page_no"])
            page_number = page_number_map.get(raw_page_number)
            if page_number not in page_specs:
                continue
            page_height = page_specs[page_number][1]
            try:
                bbox = _top_left_bbox(provenance["bbox"], page_height)
            except (KeyError, TypeError, ValueError):
                page_warnings[page_number].append("DOCLING_INVALID_BBOX")
                continue

            page_text = _provenance_text(
                text,
                provenance,
                preserve_indentation=label == "code",
            )
            if not page_text:
                continue
            reading_order = len(page_blocks[page_number])
            block_id = f"{file_sha256[:12]}-docling-p{page_number:04d}-b{reading_order:04d}"
            parent_picture_id = None
            if parent_picture_ref is not None:
                parent_picture_id = picture_ids.get((parent_picture_ref, page_number))
                if parent_picture_id is None:
                    page_warnings[page_number].append("DOCLING_PICTURE_PARENT_MISSING")
                else:
                    picture_child_ids[parent_picture_id].append(block_id)
                    if label != "caption":
                        block_type = BlockType.PICTURE_TEXT
                        block_warnings.append("PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS")
            page_blocks[page_number].append(
                ParsedBlock(
                    id=block_id,
                    page_number=page_number,
                    block_type=block_type,
                    text=page_text,
                    bbox=bbox,
                    reading_order=reading_order,
                    extraction_method=(
                        ExtractionMethod.RERUN_OCR
                        if full_page_ocr_was_enabled
                        else ExtractionMethod.LAYOUT
                    ),
                    parent_picture_id=parent_picture_id,
                    warnings=tuple(block_warnings),
                )
            )

    image_counts = _image_counts(docling_document, page_number_map)
    pages: list[ParsedPage] = []
    for page_number in sorted(page_specs):
        width, height = page_specs[page_number]
        blocks = tuple(page_blocks[page_number])
        pictures = tuple(
            ParsedPicture(
                id=picture_id,
                page_number=page_number,
                bbox=bbox,
                child_block_ids=tuple(picture_child_ids[picture_id]),
            )
            for picture_id, bbox in page_picture_specs[page_number]
        )
        warnings = page_warnings[page_number]
        if blocks:
            kind = (
                PageKind.SCANNED_WITH_OCR
                if full_page_ocr_was_enabled
                else PageKind.COMPLEX_LAYOUT_TEXT
            )
            status = ParseStatus.PARSED
        elif image_counts[page_number] and not ocr_was_enabled:
            kind = PageKind.SCANNED_IMAGE
            status = ParseStatus.NEEDS_OCR
            warnings.append("OCR_REQUIRED")
        else:
            kind = PageKind.COMPLEX_LAYOUT_TEXT
            status = ParseStatus.FAILED
            warnings.append("DOCLING_NO_TEXT_BLOCKS")

        pages.append(
            ParsedPage(
                page_number=page_number,
                width=width,
                height=height,
                kind=kind,
                status=status,
                blocks=blocks,
                pictures=pictures,
                warnings=tuple(warnings),
                text_character_count=sum(len("".join(block.text.split())) for block in blocks),
                image_count=image_counts[page_number],
            )
        )
    return tuple(pages)


def _page_specs(
    docling_document: Mapping[str, Any],
    *,
    require_start_at_one: bool = True,
) -> dict[int, tuple[float, float]]:
    pages = docling_document.get("pages")
    if not isinstance(pages, Mapping) or not pages:
        raise ValueError("Docling document must contain page metadata")

    page_specs: dict[int, tuple[float, float]] = {}
    for page_key, page in pages.items():
        page_number = int(page.get("page_no", page_key))
        size = page["size"]
        page_specs[page_number] = (float(size["width"]), float(size["height"]))

    page_numbers = sorted(page_specs)
    expected_start = 1 if require_start_at_one else page_numbers[0]
    expected = list(range(expected_start, expected_start + len(page_specs)))
    if page_numbers != expected:
        message = (
            "Docling pages must be contiguous and ordered from 1"
            if require_start_at_one
            else "Docling page range must be contiguous"
        )
        raise ValueError(message)
    return page_specs


def _item_index(docling_document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for collection_name in (
        "texts",
        "groups",
        "pictures",
        "tables",
        "key_value_items",
        "form_items",
    ):
        for item in docling_document.get(collection_name, ()):
            self_ref = item.get("self_ref")
            if self_ref:
                index[str(self_ref)] = item
    return index


def _iter_reading_order_items(
    docling_document: Mapping[str, Any],
    item_index: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[Mapping[str, Any], str | None]]:
    yielded: set[str] = set()

    def walk(
        reference: str,
        active: frozenset[str],
        parent_picture_ref: str | None = None,
    ) -> Iterator[tuple[Mapping[str, Any], str | None]]:
        if reference in active or reference not in item_index:
            return
        item = item_index[reference]
        label = str(item.get("label", ""))
        current_picture_ref = reference if label == "picture" else parent_picture_ref
        if label not in {"group", "picture"} and reference not in yielded:
            yielded.add(reference)
            yield item, current_picture_ref
        for child in item.get("children", ()):
            child_reference = child.get("$ref")
            if child_reference:
                yield from walk(
                    str(child_reference),
                    active | {reference},
                    current_picture_ref,
                )

    body = docling_document.get("body", {})
    for child in body.get("children", ()):
        reference = child.get("$ref")
        if reference:
            yield from walk(str(reference), frozenset())


def _picture_specs(
    docling_document: Mapping[str, Any],
    page_specs: Mapping[int, tuple[float, float]],
    file_sha256: str,
    page_number_map: Mapping[int, int] | None = None,
) -> tuple[
    dict[tuple[str, int], str],
    defaultdict[int, list[tuple[str, BoundingBox]]],
]:
    picture_ids: dict[tuple[str, int], str] = {}
    page_pictures: defaultdict[int, list[tuple[str, BoundingBox]]] = defaultdict(list)
    page_indexes: defaultdict[int, int] = defaultdict(int)
    for picture in docling_document.get("pictures", ()):
        picture_ref = str(picture.get("self_ref", ""))
        if not picture_ref:
            continue
        for provenance in picture.get("prov", ()):
            raw_page_number = int(provenance["page_no"])
            page_number = (
                page_number_map.get(raw_page_number)
                if page_number_map is not None
                else raw_page_number
            )
            if page_number not in page_specs:
                continue
            picture_index = page_indexes[page_number]
            page_indexes[page_number] += 1
            picture_id = f"{file_sha256[:12]}-docling-p{page_number:04d}-picture{picture_index:04d}"
            bbox = _top_left_bbox(provenance["bbox"], page_specs[page_number][1])
            picture_ids[(picture_ref, page_number)] = picture_id
            page_pictures[page_number].append((picture_id, bbox))
    return picture_ids, page_pictures


def _item_text(item: Mapping[str, Any], label: str) -> str:
    raw_text = str(item.get("text", ""))
    text = raw_text.strip("\r\n") if label == "code" else raw_text.strip()
    if not text.strip():
        text = ""
    if text or label != "table":
        return text

    data = item.get("data", {})
    row_count = int(data.get("num_rows", 0))
    column_count = int(data.get("num_cols", 0))
    if row_count <= 0 or column_count <= 0:
        return ""
    cells = [["" for _ in range(column_count)] for _ in range(row_count)]
    for cell in data.get("table_cells", ()):
        row = int(cell.get("start_row_offset_idx", -1))
        column = int(cell.get("start_col_offset_idx", -1))
        if 0 <= row < row_count and 0 <= column < column_count:
            cells[row][column] = str(cell.get("text", "")).strip()
    return "\n".join("\t".join(row).rstrip() for row in cells).strip()


def _provenance_text(
    text: str,
    provenance: Mapping[str, Any],
    *,
    preserve_indentation: bool = False,
) -> str:
    charspan = provenance.get("charspan")
    if (
        isinstance(charspan, list | tuple)
        and len(charspan) == 2
        and 0 <= int(charspan[0]) < int(charspan[1]) <= len(text)
    ):
        source_text = text[int(charspan[0]) : int(charspan[1])]
    else:
        source_text = text
    if preserve_indentation:
        source_text = source_text.strip("\r\n")
        return source_text if source_text.strip() else ""
    return source_text.strip()


def _top_left_bbox(raw_bbox: Mapping[str, Any], page_height: float) -> BoundingBox:
    left = float(raw_bbox["l"])
    top = float(raw_bbox["t"])
    right = float(raw_bbox["r"])
    bottom = float(raw_bbox["b"])
    if raw_bbox.get("coord_origin", "TOPLEFT") == "BOTTOMLEFT":
        top, bottom = page_height - top, page_height - bottom
    return BoundingBox(
        x0=min(left, right),
        y0=min(top, bottom),
        x1=max(left, right),
        y1=max(top, bottom),
    )


def _image_counts(
    docling_document: Mapping[str, Any],
    page_number_map: Mapping[int, int] | None = None,
) -> defaultdict[int, int]:
    counts: defaultdict[int, int] = defaultdict(int)
    for picture in docling_document.get("pictures", ()):
        for provenance in picture.get("prov", ()):
            raw_page_number = int(provenance["page_no"])
            page_number = (
                page_number_map.get(raw_page_number)
                if page_number_map is not None
                else raw_page_number
            )
            if page_number is not None:
                counts[page_number] += 1
    return counts


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

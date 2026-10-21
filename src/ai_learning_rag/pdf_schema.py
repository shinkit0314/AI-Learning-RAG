"""Traceable data contracts shared by PDF parsers and downstream chunking."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite


class PageKind(StrEnum):
    """How a PDF page stores its visible content."""

    NATIVE_TEXT = "native_text"
    COMPLEX_LAYOUT_TEXT = "complex_layout_text"
    SCANNED_IMAGE = "scanned_image"
    SCANNED_WITH_OCR = "scanned_with_ocr"
    MIXED = "mixed"


class ParseStatus(StrEnum):
    """Whether a page produced usable blocks or needs another route."""

    PARSED = "parsed"
    NEEDS_OCR = "needs_ocr"
    FAILED = "failed"


class BlockType(StrEnum):
    """Semantic role of a text-bearing block."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    PICTURE_TEXT = "picture_text"
    FORMULA = "formula"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class ExtractionMethod(StrEnum):
    """Origin of the text stored in a parsed block."""

    NATIVE = "native"
    LAYOUT = "layout"
    EMBEDDED_OCR = "embedded_ocr"
    RERUN_OCR = "rerun_ocr"


class ChunkContentType(StrEnum):
    """Semantic family used to route a finished PDF chunk downstream."""

    BODY_TEXT = "body_text"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE_CAPTION = "figure_caption"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A rectangular page region using top-left-origin PDF point coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("bbox coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bbox must have positive width and height")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the storage-friendly (x0, y0, x1, y1) representation."""
        return self.x0, self.y0, self.x1, self.y1


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """One text-bearing block that remains traceable to its source page."""

    id: str
    page_number: int
    block_type: BlockType
    text: str
    bbox: BoundingBox
    reading_order: int
    extraction_method: ExtractionMethod
    parent_picture_id: str | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("block id must not be blank")
        if self.page_number < 1:
            raise ValueError("page_number must start at 1")
        if not self.text.strip():
            raise ValueError("block text must not be blank")
        if self.reading_order < 0:
            raise ValueError("reading_order must not be negative")
        if self.parent_picture_id is not None and not self.parent_picture_id.strip():
            raise ValueError("parent_picture_id must not be blank")
        if self.block_type is BlockType.PICTURE_TEXT and self.parent_picture_id is None:
            raise ValueError("picture_text blocks must reference a parent picture")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ParsedPicture:
    """A detected picture region and the text blocks contained inside it."""

    id: str
    page_number: int
    bbox: BoundingBox
    child_block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("picture id must not be blank")
        if self.page_number < 1:
            raise ValueError("page_number must start at 1")
        if len(self.child_block_ids) != len(set(self.child_block_ids)):
            raise ValueError("picture child block ids must be unique")


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Page diagnostics and ordered text blocks from one parsing route."""

    page_number: int
    width: float
    height: float
    kind: PageKind
    status: ParseStatus
    blocks: tuple[ParsedBlock, ...] = ()
    pictures: tuple[ParsedPicture, ...] = ()
    warnings: tuple[str, ...] = ()
    text_character_count: int = 0
    image_count: int = 0
    image_coverage_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must start at 1")
        if not isfinite(self.width) or not isfinite(self.height):
            raise ValueError("page dimensions must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page dimensions must be positive")
        if self.text_character_count < 0:
            raise ValueError("text_character_count must not be negative")
        if self.image_count < 0:
            raise ValueError("image_count must not be negative")
        if not isfinite(self.image_coverage_ratio):
            raise ValueError("image_coverage_ratio must be finite")
        if not 0 <= self.image_coverage_ratio <= 1:
            raise ValueError("image_coverage_ratio must be between 0 and 1")
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("every block must reference its containing page")

        block_ids = [block.id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block ids must be unique within a page")

        reading_orders = [block.reading_order for block in self.blocks]
        if len(reading_orders) != len(set(reading_orders)):
            raise ValueError("reading_order values must be unique within a page")

        if any(picture.page_number != self.page_number for picture in self.pictures):
            raise ValueError("every picture must reference its containing page")
        picture_ids = [picture.id for picture in self.pictures]
        if len(picture_ids) != len(set(picture_ids)):
            raise ValueError("picture ids must be unique within a page")

        blocks_by_id = {block.id: block for block in self.blocks}
        pictures_by_id = {picture.id: picture for picture in self.pictures}
        for picture in self.pictures:
            for child_id in picture.child_block_ids:
                child = blocks_by_id.get(child_id)
                if child is None or child.parent_picture_id != picture.id:
                    raise ValueError(
                        "picture child ids must reference blocks owned by that picture"
                    )
        for block in self.blocks:
            if block.parent_picture_id is None:
                continue
            picture = pictures_by_id.get(block.parent_picture_id)
            if picture is None or block.id not in picture.child_block_ids:
                raise ValueError("picture-owned blocks must reference a picture on the same page")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser-independent PDF output consumed by later RAG stages."""

    document_id: str
    source: str
    file_sha256: str
    parser_name: str
    parser_version: str
    pages: tuple[ParsedPage, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("document_id", self.document_id),
            ("source", self.source),
            ("parser_name", self.parser_name),
            ("parser_version", self.parser_version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")

        normalized_hash = self.file_sha256.lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_hash
        ):
            raise ValueError("file_sha256 must contain 64 hexadecimal characters")

        expected_page_numbers = list(range(1, len(self.pages) + 1))
        actual_page_numbers = [page.page_number for page in self.pages]
        if actual_page_numbers != expected_page_numbers:
            raise ValueError("document pages must be contiguous and ordered from 1")

    @property
    def page_count(self) -> int:
        """Return the number of pages represented by this document."""
        return len(self.pages)


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    """Body-text parser output offered to a later chunking strategy."""

    document_id: str
    blocks: tuple[ParsedBlock, ...]
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not self.blocks:
            raise ValueError("chunk candidate must contain at least one block")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("section_path entries must not be blank")
        if any(block.block_type is BlockType.PICTURE_TEXT for block in self.blocks):
            raise ValueError("picture text must not enter an ordinary chunk candidate")

        source_locations = [(block.page_number, block.reading_order) for block in self.blocks]
        if source_locations != sorted(source_locations):
            raise ValueError("chunk candidate blocks must follow document reading order")

    @property
    def content(self) -> str:
        """Join block text without discarding the source block boundaries."""
        return "\n".join(block.text for block in self.blocks)

    @property
    def page_start(self) -> int:
        """Return the first source page represented in the candidate."""
        return self.blocks[0].page_number

    @property
    def page_end(self) -> int:
        """Return the last source page represented in the candidate."""
        return self.blocks[-1].page_number

    @property
    def source_block_ids(self) -> tuple[str, ...]:
        """Return stable source block identifiers for citation traceability."""
        return tuple(block.id for block in self.blocks)

    @property
    def bboxes(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return every contributing source region in block order."""
        return tuple(block.bbox.as_tuple() for block in self.blocks)


@dataclass(frozen=True, slots=True)
class ChunkSourceRegion:
    """Map one source block and page region to its text span inside a chunk."""

    block_id: str
    page_number: int
    bbox: BoundingBox
    block_type: BlockType
    extraction_method: ExtractionMethod
    reading_order: int
    content_start: int
    content_end: int
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("source block id must not be blank")
        if self.page_number < 1:
            raise ValueError("source page_number must start at 1")
        if self.reading_order < 0:
            raise ValueError("source reading_order must not be negative")
        if self.content_start < 0:
            raise ValueError("source content_start must not be negative")
        if self.content_end <= self.content_start:
            raise ValueError("source content range must have positive length")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("source confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TraceablePdfChunk:
    """Embedding-ready PDF text with exact source regions for later citations."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    content_type: ChunkContentType
    source_regions: tuple[ChunkSourceRegion, ...]
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("chunk id must not be blank")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if not self.content.strip():
            raise ValueError("chunk content must not be blank")
        if not self.source_regions:
            raise ValueError("chunk must contain at least one source region")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("section_path entries must not be blank")

        block_ids = [region.block_id for region in self.source_regions]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("source block ids must be unique within a chunk")

        source_locations = [
            (region.page_number, region.reading_order) for region in self.source_regions
        ]
        if source_locations != sorted(source_locations):
            raise ValueError("source regions must follow document reading order")

        previous_end = 0
        for region in self.source_regions:
            if region.content_end > len(self.content):
                raise ValueError("source content range must stay inside chunk content")
            if region.content_start < previous_end:
                raise ValueError("source content ranges must not overlap")
            if self.content[previous_end : region.content_start].strip():
                raise ValueError("chunk content must not contain unmapped source text")
            previous_end = region.content_end
        if self.content[previous_end:].strip():
            raise ValueError("chunk content must not contain unmapped source text")

    @property
    def page_start(self) -> int:
        """Return the first source page represented by this chunk."""
        return self.source_regions[0].page_number

    @property
    def page_end(self) -> int:
        """Return the last source page represented by this chunk."""
        return self.source_regions[-1].page_number

    @property
    def source_block_ids(self) -> tuple[str, ...]:
        """Return source block identifiers in chunk reading order."""
        return tuple(region.block_id for region in self.source_regions)

    @property
    def bboxes(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return every contributing PDF region in chunk reading order."""
        return tuple(region.bbox.as_tuple() for region in self.source_regions)


def build_traceable_pdf_chunk(
    candidate: ChunkCandidate,
    *,
    chunk_index: int,
    content_type: ChunkContentType = ChunkContentType.BODY_TEXT,
) -> TraceablePdfChunk:
    """Turn one accepted candidate into a deterministic, embedding-ready chunk."""
    if chunk_index < 0:
        raise ValueError("chunk_index must not be negative")

    content = candidate.content
    source_regions: list[ChunkSourceRegion] = []
    content_cursor = 0
    for block_index, block in enumerate(candidate.blocks):
        if block_index:
            content_cursor += 1  # ChunkCandidate joins adjacent blocks with one newline.
        content_start = content_cursor
        content_end = content_start + len(block.text)
        source_regions.append(
            ChunkSourceRegion(
                block_id=block.id,
                page_number=block.page_number,
                bbox=block.bbox,
                block_type=block.block_type,
                extraction_method=block.extraction_method,
                reading_order=block.reading_order,
                content_start=content_start,
                content_end=content_end,
                confidence=block.confidence,
                warnings=block.warnings,
            )
        )
        content_cursor = content_end

    identity = "\0".join(
        (
            candidate.document_id,
            content_type.value,
            "\x1f".join(candidate.section_path),
            "\x1f".join(candidate.source_block_ids),
            content,
        )
    )
    chunk_id = f"pdf_chunk_{sha256(identity.encode()).hexdigest()[:16]}"
    return TraceablePdfChunk(
        id=chunk_id,
        document_id=candidate.document_id,
        chunk_index=chunk_index,
        content=content,
        content_type=content_type,
        source_regions=tuple(source_regions),
        section_path=candidate.section_path,
    )

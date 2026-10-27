"""Traceable data contracts shared by PDF parsers and downstream chunking."""

import json
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
    OTHER = "other"


class WarningSeverity(StrEnum):
    """How strongly a chunk warning should affect downstream processing."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ChunkWarning:
    """A structured, machine-readable warning attached to one finished chunk."""

    code: str
    severity: WarningSeverity
    message: str
    source_block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("warning code must not be blank")
        if not self.message.strip():
            raise ValueError("warning message must not be blank")
        if any(not block_id.strip() for block_id in self.source_block_ids):
            raise ValueError("warning source block ids must not be blank")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("warning source block ids must be unique")


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
class ChunkBlockSlice:
    """A contiguous character range selected from one parsed source block."""

    block: ParsedBlock
    source_text_start: int
    source_text_end: int

    def __post_init__(self) -> None:
        if self.source_text_start < 0:
            raise ValueError("source_text_start must not be negative")
        if self.source_text_end <= self.source_text_start:
            raise ValueError("source text range must have positive length")
        if self.source_text_end > len(self.block.text):
            raise ValueError("source text range must stay inside the source block")
        if not self.text.strip():
            raise ValueError("source text range must contain non-whitespace text")

    @classmethod
    def whole(cls, block: ParsedBlock) -> "ChunkBlockSlice":
        """Select the complete text of one parsed block."""
        return cls(block=block, source_text_start=0, source_text_end=len(block.text))

    @property
    def text(self) -> str:
        """Return the selected source text without modifying its characters."""
        return self.block.text[self.source_text_start : self.source_text_end]


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    """Source block slices offered to a later chunking strategy."""

    document_id: str
    block_slices: tuple[ChunkBlockSlice, ...]
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not self.block_slices:
            raise ValueError("chunk candidate must contain at least one block slice")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("section_path entries must not be blank")
        if any(
            block_slice.block.block_type is BlockType.PICTURE_TEXT
            for block_slice in self.block_slices
        ):
            raise ValueError("picture text must not enter an ordinary chunk candidate")

        source_locations = [
            (block_slice.block.page_number, block_slice.block.reading_order)
            for block_slice in self.block_slices
        ]
        if source_locations != sorted(source_locations):
            raise ValueError("chunk candidate blocks must follow document reading order")

    @classmethod
    def from_blocks(
        cls,
        *,
        document_id: str,
        blocks: tuple[ParsedBlock, ...],
        section_path: tuple[str, ...] = (),
    ) -> "ChunkCandidate":
        """Create a candidate that uses the complete text of every source block."""
        return cls(
            document_id=document_id,
            block_slices=tuple(ChunkBlockSlice.whole(block) for block in blocks),
            section_path=section_path,
        )

    @property
    def blocks(self) -> tuple[ParsedBlock, ...]:
        """Return source blocks in candidate reading order."""
        return tuple(block_slice.block for block_slice in self.block_slices)

    @property
    def content(self) -> str:
        """Join selected source text without discarding slice boundaries."""
        return "\n".join(block_slice.text for block_slice in self.block_slices)

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
    chunk_content_start: int
    chunk_content_end: int
    source_text_start: int
    source_text_end: int
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("source block id must not be blank")
        if self.page_number < 1:
            raise ValueError("source page_number must start at 1")
        if self.reading_order < 0:
            raise ValueError("source reading_order must not be negative")
        if self.chunk_content_start < 0:
            raise ValueError("chunk_content_start must not be negative")
        if self.chunk_content_end <= self.chunk_content_start:
            raise ValueError("chunk content range must have positive length")
        if self.source_text_start < 0:
            raise ValueError("source_text_start must not be negative")
        if self.source_text_end <= self.source_text_start:
            raise ValueError("source text range must have positive length")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("source confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TraceablePdfChunk:
    """Embedding-ready PDF text with exact source regions for later citations."""

    id: str
    source_fingerprint: str
    document_id: str
    chunk_index: int
    chunking_profile: str
    content: str
    content_type: ChunkContentType
    source_regions: tuple[ChunkSourceRegion, ...]
    section_path: tuple[str, ...] = ()
    warnings: tuple[ChunkWarning, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("chunk id must not be blank")
        if not _is_sha256(self.source_fingerprint):
            raise ValueError("source_fingerprint must contain 64 hexadecimal characters")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if not self.chunking_profile.strip():
            raise ValueError("chunking_profile must not be blank")
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

        source_block_ids = set(block_ids)
        if any(
            not set(warning.source_block_ids).issubset(source_block_ids)
            for warning in self.warnings
        ):
            raise ValueError("chunk warnings must reference source blocks in the chunk")

        previous_end = 0
        for region in self.source_regions:
            if region.chunk_content_end > len(self.content):
                raise ValueError("source content range must stay inside chunk content")
            if region.chunk_content_start < previous_end:
                raise ValueError("source content ranges must not overlap")
            if self.content[previous_end : region.chunk_content_start].strip():
                raise ValueError("chunk content must not contain unmapped source text")
            previous_end = region.chunk_content_end
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
    chunking_profile: str,
    content_type: ChunkContentType = ChunkContentType.BODY_TEXT,
    warnings: tuple[ChunkWarning, ...] = (),
) -> TraceablePdfChunk:
    """Turn one accepted candidate into a deterministic, embedding-ready chunk."""
    if chunk_index < 0:
        raise ValueError("chunk_index must not be negative")
    if not chunking_profile.strip():
        raise ValueError("chunking_profile must not be blank")

    content = candidate.content
    source_regions: list[ChunkSourceRegion] = []
    content_cursor = 0
    for slice_index, block_slice in enumerate(candidate.block_slices):
        if slice_index:
            content_cursor += 1  # ChunkCandidate joins adjacent blocks with one newline.
        block = block_slice.block
        chunk_content_start = content_cursor
        chunk_content_end = chunk_content_start + len(block_slice.text)
        source_regions.append(
            ChunkSourceRegion(
                block_id=block.id,
                page_number=block.page_number,
                bbox=block.bbox,
                block_type=block.block_type,
                extraction_method=block.extraction_method,
                reading_order=block.reading_order,
                chunk_content_start=chunk_content_start,
                chunk_content_end=chunk_content_end,
                source_text_start=block_slice.source_text_start,
                source_text_end=block_slice.source_text_end,
                confidence=block.confidence,
                warnings=block.warnings,
            )
        )
        content_cursor = chunk_content_end

    source_identity = json.dumps(
        {
            "document_id": candidate.document_id,
            "content_type": content_type.value,
            "section_path": candidate.section_path,
            "source_ranges": [
                {
                    "block_id": region.block_id,
                    "source_text_start": region.source_text_start,
                    "source_text_end": region.source_text_end,
                }
                for region in source_regions
            ],
            "content": content,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    source_fingerprint = sha256(source_identity.encode()).hexdigest()
    chunk_identity = f"{chunking_profile}\0{source_fingerprint}"
    chunk_id = f"pdf_chunk_{sha256(chunk_identity.encode()).hexdigest()[:8]}"
    return TraceablePdfChunk(
        id=chunk_id,
        source_fingerprint=source_fingerprint,
        document_id=candidate.document_id,
        chunk_index=chunk_index,
        chunking_profile=chunking_profile,
        content=content,
        content_type=content_type,
        source_regions=tuple(source_regions),
        section_path=candidate.section_path,
        warnings=warnings,
    )


def _is_sha256(value: str) -> bool:
    normalized = value.lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )

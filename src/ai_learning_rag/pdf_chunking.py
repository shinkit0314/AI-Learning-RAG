"""Configuration, filtering, and section state for structure-aware PDF chunking."""

from dataclasses import dataclass
from enum import StrEnum
from re import compile, fullmatch

from ai_learning_rag.pdf_schema import (
    BlockType,
    ChunkBlockSlice,
    ChunkCandidate,
    ChunkContentType,
    ChunkWarning,
    ParsedBlock,
    ParsedDocument,
    ParseStatus,
    TraceablePdfChunk,
    WarningSeverity,
    build_traceable_pdf_chunk,
)

_PROFILE_NAME_PATTERN = r"[a-z0-9][a-z0-9._-]*"
_DECIMAL_HEADING_PATTERN = compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)、]|\s+|$)")
_DOCUMENT_PAGE_WARNING_PATTERN = compile(r"^PAGE_\d+_.+")
_SENTENCE_BOUNDARY_PATTERN = compile(
    r"(?:[。！？](?:[\"'”’）】])?|[.!?](?:[\"'”’）】])?(?=\s|$))\s*"
)
_NEWLINE_BOUNDARY_PATTERN = compile(r"\r\n|\r|\n")
_WHITESPACE_BOUNDARY_PATTERN = compile(r"\s+")
_BLANK_LINE_BOUNDARY_PATTERN = compile(r"(?:\r\n|\r|\n)[ \t]*(?:\r\n|\r|\n)+")
_CODE_DEFINITION_BOUNDARY_PATTERN = compile(r"(?m)^(?=(?:async\s+def|def|class|function)\b)")

_FILTERED_BLOCK_TYPES = frozenset(
    {
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PRINTED_PAGE_NUMBER,
        BlockType.PICTURE_TEXT,
    }
)
_BODY_BLOCK_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.LIST_ITEM})
_ALLOWED_PAGE_WARNINGS = frozenset(
    {
        "EMBEDDED_OCR_TEXT_LAYER",
        "EMBEDDED_OCR_VERIFIED",
        "RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER",
        "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR",
        "MANUAL_REVIEW_ACCEPTED_RERUN_OCR",
        "DOCLING_INVALID_BBOX",
        "DOCLING_PICTURE_PARENT_MISSING",
    }
)
_BLOCKING_PAGE_WARNINGS = frozenset(
    {
        "OCR_REQUIRED",
        "NO_EXTRACTABLE_TEXT",
        "DOCLING_NO_TEXT_BLOCKS",
    }
)
_ALLOWED_BLOCK_WARNINGS = frozenset({"PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS"})
_ALLOWED_BLOCK_WARNING_PREFIXES = ("DOCLING_UNMAPPED_LABEL_",)
_BLOCKING_BLOCK_WARNINGS: frozenset[str] = frozenset()

_CONTENT_TYPE_BY_BLOCK_TYPE = {
    BlockType.PARAGRAPH: ChunkContentType.BODY_TEXT,
    BlockType.LIST_ITEM: ChunkContentType.BODY_TEXT,
    BlockType.TABLE: ChunkContentType.TABLE,
    BlockType.FORMULA: ChunkContentType.FORMULA,
    BlockType.FIGURE_CAPTION: ChunkContentType.FIGURE_CAPTION,
    BlockType.CODE: ChunkContentType.CODE,
    BlockType.OTHER: ChunkContentType.OTHER,
}


class PdfChunkingStatus(StrEnum):
    """Overall outcome of one document chunking attempt."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ChunkingDiagnosticCode(StrEnum):
    """Stable reason codes emitted by the structure-aware chunker."""

    NO_CHUNKS_PRODUCED = "NO_CHUNKS_PRODUCED"
    SOURCE_DOCUMENT_WARNING_RECORDED = "SOURCE_DOCUMENT_WARNING_RECORDED"
    PAGE_NEEDS_OCR_SKIPPED = "PAGE_NEEDS_OCR_SKIPPED"
    PAGE_PARSE_FAILED_SKIPPED = "PAGE_PARSE_FAILED_SKIPPED"
    EMPTY_PAGE_SKIPPED = "EMPTY_PAGE_SKIPPED"
    PAGE_BLOCKED_BY_WARNING = "PAGE_BLOCKED_BY_WARNING"
    UNKNOWN_PAGE_WARNING = "UNKNOWN_PAGE_WARNING"
    BLOCK_FILTERED = "BLOCK_FILTERED"
    BLOCK_BLOCKED_BY_WARNING = "BLOCK_BLOCKED_BY_WARNING"
    UNKNOWN_BLOCK_WARNING = "UNKNOWN_BLOCK_WARNING"
    SOURCE_WARNING_ACCEPTED = "SOURCE_WARNING_ACCEPTED"
    HEADING_LEVEL_INFERRED = "HEADING_LEVEL_INFERRED"
    HEADING_LEVEL_CONFLICT = "HEADING_LEVEL_CONFLICT"
    HEADING_LEVEL_UNKNOWN = "HEADING_LEVEL_UNKNOWN"
    HEADING_LEVEL_GAP = "HEADING_LEVEL_GAP"
    ADDITIONAL_TITLE_AS_HEADING = "ADDITIONAL_TITLE_AS_HEADING"
    HEADING_WITHOUT_BODY = "HEADING_WITHOUT_BODY"
    SECTION_CONTEXT_RESET = "SECTION_CONTEXT_RESET"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    BODY_BLOCK_SPLIT = "BODY_BLOCK_SPLIT"
    BODY_OVERLAP_APPLIED = "BODY_OVERLAP_APPLIED"
    CROSS_PAGE_BODY_MERGED = "CROSS_PAGE_BODY_MERGED"
    CROSS_PAGE_EXTRACTION_METHOD_CHANGED = "CROSS_PAGE_EXTRACTION_METHOD_CHANGED"
    CROSS_PAGE_HYPHEN_PRESERVED = "CROSS_PAGE_HYPHEN_PRESERVED"
    HEADING_PREFIX_EXCEEDS_HARD_MAX = "HEADING_PREFIX_EXCEEDS_HARD_MAX"
    TABLE_BLOCK_SPLIT = "TABLE_BLOCK_SPLIT"
    TABLE_ROW_SPLIT = "TABLE_ROW_SPLIT"
    FORMULA_EXCEEDS_HARD_MAX = "FORMULA_EXCEEDS_HARD_MAX"
    FIGURE_CAPTION_PARENT_MISSING = "FIGURE_CAPTION_PARENT_MISSING"
    FIGURE_CAPTION_EXCEEDS_HARD_MAX = "FIGURE_CAPTION_EXCEEDS_HARD_MAX"
    CODE_BLOCK_SPLIT = "CODE_BLOCK_SPLIT"
    CODE_LINE_SPLIT = "CODE_LINE_SPLIT"
    OTHER_EXCEEDS_HARD_MAX = "OTHER_EXCEEDS_HARD_MAX"


@dataclass(frozen=True, slots=True)
class PdfChunkingConfig:
    """Deterministic parameters that can change PDF chunk boundaries."""

    strategy_version: str = "pdf-structure-v1"
    length_unit: str = "characters"
    target_size: int = 400
    hard_max_size: int = 600
    overlap_percent: int = 20

    def __post_init__(self) -> None:
        for field_name, value in (
            ("strategy_version", self.strategy_version),
            ("length_unit", self.length_unit),
        ):
            if fullmatch(_PROFILE_NAME_PATTERN, value) is None:
                raise ValueError(
                    f"{field_name} must use lowercase letters, digits, dots, "
                    "underscores, or hyphens"
                )
        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if self.hard_max_size < self.target_size:
            raise ValueError("hard_max_size must be greater than or equal to target_size")
        if not 0 <= self.overlap_percent < 100:
            raise ValueError("overlap_percent must be between 0 and 99")

    @property
    def overlap_max(self) -> int:
        """Return the maximum complete body unit allowed as overlap."""

        return self.target_size * self.overlap_percent // 100

    @property
    def chunking_profile(self) -> str:
        """Return the canonical, human-readable identity of this configuration."""

        return (
            f"{self.strategy_version};length={self.length_unit};target={self.target_size};"
            f"hard_max={self.hard_max_size};overlap_percent={self.overlap_percent}"
        )


@dataclass(frozen=True, slots=True)
class ChunkingDiagnostic:
    """A document-level observation that may not belong to a produced chunk."""

    code: ChunkingDiagnosticCode
    severity: WarningSeverity
    message: str
    page_numbers: tuple[int, ...] = ()
    source_block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, ChunkingDiagnosticCode):
            raise ValueError("diagnostic code must be a ChunkingDiagnosticCode")
        if not self.message.strip():
            raise ValueError("diagnostic message must not be blank")
        if any(page_number < 1 for page_number in self.page_numbers):
            raise ValueError("diagnostic page numbers must start at 1")
        if tuple(sorted(set(self.page_numbers))) != self.page_numbers:
            raise ValueError("diagnostic page numbers must be sorted and unique")
        if any(not block_id.strip() for block_id in self.source_block_ids):
            raise ValueError("diagnostic source block ids must not be blank")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("diagnostic source block ids must be unique")


@dataclass(frozen=True, slots=True)
class PdfChunkingResult:
    """Chunks and diagnostics produced for one parsed PDF document."""

    chunks: tuple[TraceablePdfChunk, ...] = ()
    diagnostics: tuple[ChunkingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        chunk_ids = [chunk.id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk ids must be unique within a result")
        if [chunk.chunk_index for chunk in self.chunks] != list(range(len(self.chunks))):
            raise ValueError("chunk indexes must be contiguous and follow result order")

        no_chunks_diagnostics = [
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.code is ChunkingDiagnosticCode.NO_CHUNKS_PRODUCED
        ]
        if no_chunks_diagnostics and any(
            diagnostic.severity is not WarningSeverity.ERROR for diagnostic in no_chunks_diagnostics
        ):
            raise ValueError("NO_CHUNKS_PRODUCED must have error severity")
        if not self.chunks and not no_chunks_diagnostics:
            object.__setattr__(
                self,
                "diagnostics",
                self.diagnostics
                + (
                    ChunkingDiagnostic(
                        code=ChunkingDiagnosticCode.NO_CHUNKS_PRODUCED,
                        severity=WarningSeverity.ERROR,
                        message="The parsed document did not produce any chunks.",
                    ),
                ),
            )

    @property
    def status(self) -> PdfChunkingStatus:
        """Derive the overall status without storing duplicate mutable state."""

        if not self.chunks:
            return PdfChunkingStatus.FAILED
        if any(diagnostic.severity is WarningSeverity.ERROR for diagnostic in self.diagnostics):
            return PdfChunkingStatus.PARTIAL
        return PdfChunkingStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class PreparedChunkUnit:
    """One accepted source unit before length and content-specific splitting."""

    candidate: ChunkCandidate
    content_type: ChunkContentType
    warnings: tuple[ChunkWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class PdfChunkPreparationResult:
    """Filtered source units plus diagnostics from the preparation pass."""

    units: tuple[PreparedChunkUnit, ...] = ()
    diagnostics: tuple[ChunkingDiagnostic, ...] = ()


@dataclass(slots=True)
class _SectionState:
    root_title: str | None = None
    headings: dict[int, str] | None = None
    pending_headings: list[tuple[int, ParsedBlock]] | None = None
    pending_warnings: dict[str, tuple[ChunkWarning, ...]] | None = None
    saw_title: bool = False

    def __post_init__(self) -> None:
        self.headings = {} if self.headings is None else self.headings
        self.pending_headings = [] if self.pending_headings is None else self.pending_headings
        self.pending_warnings = {} if self.pending_warnings is None else self.pending_warnings

    @property
    def path(self) -> tuple[str, ...]:
        assert self.headings is not None
        values = ([self.root_title] if self.root_title is not None else []) + [
            self.headings[level] for level in sorted(self.headings)
        ]
        return tuple(values)

    def reset(self) -> tuple[tuple[ParsedBlock, ...], bool]:
        assert self.headings is not None
        assert self.pending_headings is not None
        assert self.pending_warnings is not None
        had_context = bool(self.root_title or self.headings or self.pending_headings)
        pending = tuple(block for _, block in self.pending_headings)
        self.root_title = None
        self.headings.clear()
        self.pending_headings.clear()
        self.pending_warnings.clear()
        self.saw_title = False
        return pending, had_context


def chunk_pdf_document(
    document: ParsedDocument,
    config: PdfChunkingConfig | None = None,
) -> PdfChunkingResult:
    """Convert one parser-independent document into traceable ordered chunks."""

    resolved_config = PdfChunkingConfig() if config is None else config
    preparation = prepare_pdf_document(document)
    preparation = split_oversized_body_units(preparation, resolved_config)
    preparation = compose_same_page_body_units(preparation, resolved_config)
    preparation = compose_cross_page_body_units(preparation, resolved_config)
    preparation = split_oversized_table_units(preparation, resolved_config)
    preparation = preserve_atomic_formula_units(preparation, resolved_config)
    preparation = preserve_atomic_figure_caption_units(preparation, resolved_config)
    preparation = split_oversized_code_units(preparation, resolved_config)
    preparation = preserve_isolated_other_units(preparation, resolved_config)

    chunks = tuple(
        build_traceable_pdf_chunk(
            unit.candidate,
            chunk_index=chunk_index,
            chunking_profile=resolved_config.chunking_profile,
            content_type=unit.content_type,
            warnings=unit.warnings,
        )
        for chunk_index, unit in enumerate(preparation.units)
    )
    return PdfChunkingResult(chunks=chunks, diagnostics=preparation.diagnostics)


def prepare_pdf_document(document: ParsedDocument) -> PdfChunkPreparationResult:
    """Filter unsafe input and assign section context before chunk composition."""

    units: list[PreparedChunkUnit] = []
    diagnostics: list[ChunkingDiagnostic] = []
    section = _SectionState()

    for warning in document.warnings:
        if _DOCUMENT_PAGE_WARNING_PATTERN.fullmatch(warning):
            continue
        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.SOURCE_DOCUMENT_WARNING_RECORDED,
                WarningSeverity.WARNING,
                f"Document warning was recorded without repeating page gating: {warning}.",
            )
        )

    for page in document.pages:
        page_block_reason = _page_block_reason(page.status, page.warnings)
        if page_block_reason is not None:
            code, message = page_block_reason
            diagnostics.append(
                _diagnostic(code, WarningSeverity.ERROR, message, pages=(page.page_number,))
            )
            pending, had_context = section.reset()
            if pending:
                diagnostics.extend(_heading_without_body_diagnostics(pending))
            if had_context:
                diagnostics.append(
                    _diagnostic(
                        ChunkingDiagnosticCode.SECTION_CONTEXT_RESET,
                        WarningSeverity.WARNING,
                        "Section context was reset after an unusable page.",
                        pages=(page.page_number,),
                    )
                )
            continue

        page_chunk_warnings = _accepted_page_warnings(page.warnings)
        for warning in page_chunk_warnings:
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.SOURCE_WARNING_ACCEPTED,
                    warning.severity,
                    warning.message,
                    pages=(page.page_number,),
                )
            )

        if not page.blocks:
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.EMPTY_PAGE_SKIPPED,
                    WarningSeverity.INFO,
                    "Parsed page contained no text blocks.",
                    pages=(page.page_number,),
                )
            )
            continue

        for block in sorted(page.blocks, key=lambda item: item.reading_order):
            block_gate = _block_warning_gate(block)
            if block_gate is not None:
                code, message = block_gate
                diagnostics.append(
                    _diagnostic(
                        code,
                        WarningSeverity.ERROR,
                        message,
                        pages=(block.page_number,),
                        blocks=(block.id,),
                    )
                )
                continue

            if block.block_type in _FILTERED_BLOCK_TYPES:
                diagnostics.append(
                    _diagnostic(
                        ChunkingDiagnosticCode.BLOCK_FILTERED,
                        WarningSeverity.INFO,
                        f"Filtered {block.block_type.value} block from text chunks.",
                        pages=(block.page_number,),
                        blocks=(block.id,),
                    )
                )
                continue

            block_chunk_warnings = _accepted_block_warnings(block)
            for warning in block_chunk_warnings:
                diagnostics.append(
                    _diagnostic(
                        ChunkingDiagnosticCode.SOURCE_WARNING_ACCEPTED,
                        warning.severity,
                        warning.message,
                        pages=(block.page_number,),
                        blocks=(block.id,),
                    )
                )

            if block.block_type in {BlockType.TITLE, BlockType.HEADING}:
                diagnostics.extend(
                    _accept_heading(
                        section,
                        block,
                        _bind_warnings_to_block(page_chunk_warnings, block) + block_chunk_warnings,
                    )
                )
                continue

            content_type = _CONTENT_TYPE_BY_BLOCK_TYPE[block.block_type]
            candidate_blocks = (block,)
            if block.block_type in _BODY_BLOCK_TYPES:
                assert section.pending_headings is not None
                assert section.pending_warnings is not None
                candidate_blocks = tuple(heading for _, heading in section.pending_headings) + (
                    block,
                )
                pending_warnings = tuple(
                    warning
                    for _, heading in section.pending_headings
                    for warning in section.pending_warnings.get(heading.id, ())
                )
                section.pending_headings.clear()
                section.pending_warnings.clear()
            else:
                pending_warnings = ()

            candidate = ChunkCandidate.from_blocks(
                document_id=document.document_id,
                blocks=candidate_blocks,
                section_path=section.path,
                routing_profile=document.routing_profile,
            )
            source_warnings = (
                pending_warnings
                + _bind_warnings_to_block(page_chunk_warnings, block)
                + block_chunk_warnings
            )
            units.append(
                PreparedChunkUnit(
                    candidate=candidate,
                    content_type=content_type,
                    warnings=_bind_warnings_to_candidate(source_warnings, candidate),
                )
            )
            if content_type is ChunkContentType.OTHER:
                diagnostics.append(
                    _diagnostic(
                        ChunkingDiagnosticCode.UNSUPPORTED_CONTENT_TYPE,
                        WarningSeverity.WARNING,
                        "Preserved unsupported content as an isolated OTHER unit.",
                        pages=(block.page_number,),
                        blocks=(block.id,),
                    )
                )

    assert section.pending_headings is not None
    diagnostics.extend(
        _heading_without_body_diagnostics(tuple(block for _, block in section.pending_headings))
    )
    return PdfChunkPreparationResult(
        units=tuple(units),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def split_oversized_body_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Split oversized body blocks while preserving exact source offsets."""

    units: list[PreparedChunkUnit] = []
    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if (
            unit.content_type is not ChunkContentType.BODY_TEXT
            or len(unit.candidate.content) <= config.hard_max_size
        ):
            units.append(unit)
            continue

        body_slice = unit.candidate.block_slices[-1]
        if body_slice.block.block_type not in _BODY_BLOCK_TYPES:
            raise ValueError("body unit must end with a paragraph or list item")

        prefix_slices = unit.candidate.block_slices[:-1]
        prefix_cost = sum(len(block_slice.text) for block_slice in prefix_slices) + len(
            prefix_slices
        )
        first_hard_limit = max(1, config.hard_max_size - prefix_cost)
        first_target_limit = max(
            1,
            min(first_hard_limit, config.target_size - prefix_cost),
        )
        if prefix_cost >= config.hard_max_size:
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.HEADING_PREFIX_EXCEEDS_HARD_MAX,
                    WarningSeverity.WARNING,
                    "Pending heading text left no room for body text under hard_max.",
                    pages=tuple(
                        sorted({block_slice.block.page_number for block_slice in prefix_slices})
                    ),
                    blocks=tuple(block_slice.block.id for block_slice in prefix_slices),
                )
            )

        relative_ranges = _split_text_ranges(
            body_slice.text,
            first_target_limit=first_target_limit,
            first_hard_limit=first_hard_limit,
            target_limit=config.target_size,
            hard_limit=config.hard_max_size,
        )
        for slice_index, (relative_start, relative_end) in enumerate(relative_ranges):
            split_slice = ChunkBlockSlice(
                block=body_slice.block,
                source_text_start=body_slice.source_text_start + relative_start,
                source_text_end=body_slice.source_text_start + relative_end,
            )
            candidate = ChunkCandidate(
                document_id=unit.candidate.document_id,
                block_slices=(
                    prefix_slices + (split_slice,) if slice_index == 0 else (split_slice,)
                ),
                section_path=unit.candidate.section_path,
                routing_profile=unit.candidate.routing_profile,
            )
            units.append(
                PreparedChunkUnit(
                    candidate=candidate,
                    content_type=unit.content_type,
                    warnings=_bind_warnings_to_candidate(unit.warnings, candidate),
                )
            )

        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.BODY_BLOCK_SPLIT,
                WarningSeverity.INFO,
                "Oversized body block was split into traceable source slices.",
                pages=(body_slice.block.page_number,),
                blocks=(body_slice.block.id,),
            )
        )

    return PdfChunkPreparationResult(
        units=tuple(units),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def compose_same_page_body_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Combine adjacent body units on one page within target and hard limits."""

    output: list[PreparedChunkUnit] = []
    buffer: list[PreparedChunkUnit] = []
    diagnostics = list(preparation.diagnostics)

    def flush() -> None:
        if buffer:
            output.append(_compose_units(tuple(buffer)))
            buffer.clear()

    for unit in preparation.units:
        if unit.content_type is not ChunkContentType.BODY_TEXT:
            flush()
            output.append(unit)
            continue

        if _is_atomic_body_unit(unit, config):
            flush()
            output.append(unit)
            continue

        if not buffer:
            buffer.append(unit)
            continue

        if not _same_body_group(buffer[-1], unit):
            flush()
            buffer.append(unit)
            continue

        composed = _compose_units(tuple(buffer))
        added_length = 1 + len(unit.candidate.content)
        hits_length_boundary = (
            len(composed.candidate.content) >= config.target_size
            or len(composed.candidate.content) + added_length > config.hard_max_size
        )
        if not hits_length_boundary:
            buffer.append(unit)
            continue

        previous = buffer[-1]
        flush()
        overlap = _overlap_seed(previous, unit, config)
        if overlap is not None:
            buffer.append(overlap)
            overlap_slice = overlap.candidate.block_slices[-1]
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED,
                    WarningSeverity.INFO,
                    "Previous complete body unit was repeated across a length boundary.",
                    pages=(overlap_slice.block.page_number,),
                    blocks=(overlap_slice.block.id,),
                )
            )
        buffer.append(unit)

    flush()
    return PdfChunkPreparationResult(
        units=tuple(output),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def compose_cross_page_body_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Combine compatible body units across adjacent physical pages."""

    output: list[PreparedChunkUnit] = []
    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if not output:
            output.append(unit)
            continue

        previous = output[-1]
        boundary = _cross_page_boundary(previous, unit)
        if boundary is None or len(previous.candidate.content) >= config.target_size:
            output.append(unit)
            continue

        left_slice, right_slice = boundary
        combined_length = len(previous.candidate.content) + 1 + len(unit.candidate.content)
        if combined_length <= config.hard_max_size:
            output[-1] = _compose_units((previous, unit))
            diagnostics.append(
                _cross_page_diagnostic(
                    ChunkingDiagnosticCode.CROSS_PAGE_BODY_MERGED,
                    "Compatible body units were combined across adjacent pages.",
                    left_slice,
                    right_slice,
                )
            )
            diagnostics.extend(_cross_page_quality_diagnostics(left_slice, right_slice))
            continue

        overlap = _overlap_seed(previous, unit, config)
        if overlap is None:
            output.append(unit)
            continue

        output.append(_compose_units((overlap, unit)))
        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.BODY_OVERLAP_APPLIED,
                WarningSeverity.INFO,
                "Previous complete body unit was repeated across a length boundary.",
                pages=(left_slice.block.page_number,),
                blocks=(left_slice.block.id,),
            )
        )
        diagnostics.extend(_cross_page_quality_diagnostics(left_slice, right_slice))

    return PdfChunkPreparationResult(
        units=tuple(output),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def split_oversized_table_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Split oversized tables without repeating headers or applying overlap."""

    output: list[PreparedChunkUnit] = []
    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if (
            unit.content_type is not ChunkContentType.TABLE
            or len(unit.candidate.content) <= config.hard_max_size
        ):
            output.append(unit)
            continue

        if len(unit.candidate.block_slices) != 1:
            raise ValueError("table unit must contain exactly one source block slice")
        table_slice = unit.candidate.block_slices[0]
        relative_ranges = _split_table_text_ranges(
            table_slice.text,
            target_limit=config.target_size,
            hard_limit=config.hard_max_size,
        )
        for relative_start, relative_end in relative_ranges:
            split_slice = ChunkBlockSlice(
                block=table_slice.block,
                source_text_start=table_slice.source_text_start + relative_start,
                source_text_end=table_slice.source_text_start + relative_end,
            )
            candidate = ChunkCandidate(
                document_id=unit.candidate.document_id,
                block_slices=(split_slice,),
                section_path=unit.candidate.section_path,
                routing_profile=unit.candidate.routing_profile,
            )
            output.append(
                PreparedChunkUnit(
                    candidate=candidate,
                    content_type=ChunkContentType.TABLE,
                    warnings=_bind_warnings_to_candidate(unit.warnings, candidate),
                )
            )

        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.TABLE_BLOCK_SPLIT,
                WarningSeverity.INFO,
                "Oversized table was split without repeating its header row.",
                pages=(table_slice.block.page_number,),
                blocks=(table_slice.block.id,),
            )
        )
        if any(
            len(line) > config.hard_max_size for line in table_slice.text.splitlines(keepends=True)
        ):
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.TABLE_ROW_SPLIT,
                    WarningSeverity.WARNING,
                    "At least one table row exceeded hard_max and was split.",
                    pages=(table_slice.block.page_number,),
                    blocks=(table_slice.block.id,),
                )
            )

    return PdfChunkPreparationResult(
        units=tuple(output),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def preserve_atomic_formula_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Keep formulas atomic and report formulas that exceed the hard limit."""

    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if (
            unit.content_type is not ChunkContentType.FORMULA
            or len(unit.candidate.content) <= config.hard_max_size
        ):
            continue

        formula_slices = unit.candidate.block_slices
        if len(formula_slices) != 1:
            raise ValueError("formula unit must contain exactly one source block slice")
        formula_slice = formula_slices[0]
        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.FORMULA_EXCEEDS_HARD_MAX,
                WarningSeverity.WARNING,
                "Oversized formula was preserved as one atomic unit.",
                pages=(formula_slice.block.page_number,),
                blocks=(formula_slice.block.id,),
            )
        )

    return PdfChunkPreparationResult(
        units=preparation.units,
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def preserve_atomic_figure_caption_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Keep figure captions atomic and preserve parser-provided picture links."""

    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if unit.content_type is not ChunkContentType.FIGURE_CAPTION:
            continue

        caption_slices = unit.candidate.block_slices
        if len(caption_slices) != 1:
            raise ValueError("figure caption unit must contain exactly one source block slice")
        caption_slice = caption_slices[0]
        if caption_slice.block.parent_picture_id is None:
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.FIGURE_CAPTION_PARENT_MISSING,
                    WarningSeverity.WARNING,
                    "Figure caption was preserved without a confirmed parent picture.",
                    pages=(caption_slice.block.page_number,),
                    blocks=(caption_slice.block.id,),
                )
            )
        if len(unit.candidate.content) > config.hard_max_size:
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.FIGURE_CAPTION_EXCEEDS_HARD_MAX,
                    WarningSeverity.WARNING,
                    "Oversized figure caption was preserved as one atomic unit.",
                    pages=(caption_slice.block.page_number,),
                    blocks=(caption_slice.block.id,),
                )
            )

    return PdfChunkPreparationResult(
        units=preparation.units,
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def split_oversized_code_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Split oversized code at structural or line boundaries without reformatting."""

    output: list[PreparedChunkUnit] = []
    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if (
            unit.content_type is not ChunkContentType.CODE
            or len(unit.candidate.content) <= config.hard_max_size
        ):
            output.append(unit)
            continue

        code_slices = unit.candidate.block_slices
        if len(code_slices) != 1:
            raise ValueError("code unit must contain exactly one source block slice")
        code_slice = code_slices[0]
        relative_ranges = _split_code_text_ranges(
            code_slice.text,
            target_limit=config.target_size,
            hard_limit=config.hard_max_size,
        )
        for relative_start, relative_end in relative_ranges:
            split_slice = ChunkBlockSlice(
                block=code_slice.block,
                source_text_start=code_slice.source_text_start + relative_start,
                source_text_end=code_slice.source_text_start + relative_end,
            )
            candidate = ChunkCandidate(
                document_id=unit.candidate.document_id,
                block_slices=(split_slice,),
                section_path=unit.candidate.section_path,
                routing_profile=unit.candidate.routing_profile,
            )
            output.append(
                PreparedChunkUnit(
                    candidate=candidate,
                    content_type=ChunkContentType.CODE,
                    warnings=_bind_warnings_to_candidate(unit.warnings, candidate),
                )
            )

        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.CODE_BLOCK_SPLIT,
                WarningSeverity.INFO,
                "Oversized code block was split without changing source text.",
                pages=(code_slice.block.page_number,),
                blocks=(code_slice.block.id,),
            )
        )
        if any(len(line) > config.hard_max_size for line in code_slice.text.splitlines(True)):
            diagnostics.append(
                _diagnostic(
                    ChunkingDiagnosticCode.CODE_LINE_SPLIT,
                    WarningSeverity.WARNING,
                    "At least one code line exceeded hard_max and was split.",
                    pages=(code_slice.block.page_number,),
                    blocks=(code_slice.block.id,),
                )
            )

    return PdfChunkPreparationResult(
        units=tuple(output),
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def preserve_isolated_other_units(
    preparation: PdfChunkPreparationResult,
    config: PdfChunkingConfig,
) -> PdfChunkPreparationResult:
    """Keep unsupported content isolated and report hard-limit exceptions."""

    diagnostics = list(preparation.diagnostics)
    for unit in preparation.units:
        if (
            unit.content_type is not ChunkContentType.OTHER
            or len(unit.candidate.content) <= config.hard_max_size
        ):
            continue

        other_slices = unit.candidate.block_slices
        if len(other_slices) != 1:
            raise ValueError("OTHER unit must contain exactly one source block slice")
        other_slice = other_slices[0]
        diagnostics.append(
            _diagnostic(
                ChunkingDiagnosticCode.OTHER_EXCEEDS_HARD_MAX,
                WarningSeverity.WARNING,
                "Oversized unsupported content was preserved as one isolated unit.",
                pages=(other_slice.block.page_number,),
                blocks=(other_slice.block.id,),
            )
        )

    return PdfChunkPreparationResult(
        units=preparation.units,
        diagnostics=_aggregate_diagnostics(diagnostics),
    )


def _split_code_text_ranges(
    text: str,
    *,
    target_limit: int,
    hard_limit: int,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= hard_limit:
            end = len(text)
        else:
            end = _choose_code_boundary(
                text,
                start=start,
                target_limit=target_limit,
                hard_limit=hard_limit,
            )
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _choose_code_boundary(
    text: str,
    *,
    start: int,
    target_limit: int,
    hard_limit: int,
) -> int:
    target_end = min(len(text), start + target_limit)
    hard_end = min(len(text), start + hard_limit)
    minimum_end = start + max(1, target_limit // 2)
    window = text[start:hard_end]

    structural_candidates = sorted(
        {
            start + match.end()
            for pattern in (
                _CODE_DEFINITION_BOUNDARY_PATTERN,
                _BLANK_LINE_BOUNDARY_PATTERN,
            )
            for match in pattern.finditer(window)
            if start + match.end() >= minimum_end
        }
    )
    if structural_candidates:
        at_or_after_target = [
            candidate for candidate in structural_candidates if candidate >= target_end
        ]
        if at_or_after_target:
            return at_or_after_target[0]
        return structural_candidates[-1]

    for pattern in (_NEWLINE_BOUNDARY_PATTERN, _WHITESPACE_BOUNDARY_PATTERN):
        candidates = [
            start + match.end()
            for match in pattern.finditer(window)
            if start + match.end() >= minimum_end
        ]
        if not candidates:
            continue
        at_or_after_target = [candidate for candidate in candidates if candidate >= target_end]
        if at_or_after_target:
            return at_or_after_target[0]
        return candidates[-1]
    return hard_end


def _split_table_text_ranges(
    text: str,
    *,
    target_limit: int,
    hard_limit: int,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= hard_limit:
            end = len(text)
        else:
            end = _choose_table_boundary(
                text,
                start=start,
                target_limit=target_limit,
                hard_limit=hard_limit,
            )
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _choose_table_boundary(
    text: str,
    *,
    start: int,
    target_limit: int,
    hard_limit: int,
) -> int:
    target_end = min(len(text), start + target_limit)
    hard_end = min(len(text), start + hard_limit)
    minimum_end = start + max(1, target_limit // 2)
    window = text[start:hard_end]
    for pattern in (_NEWLINE_BOUNDARY_PATTERN, _WHITESPACE_BOUNDARY_PATTERN):
        candidates = [
            start + match.end()
            for match in pattern.finditer(window)
            if start + match.end() >= minimum_end
        ]
        if not candidates:
            continue
        at_or_after_target = [candidate for candidate in candidates if candidate >= target_end]
        if at_or_after_target:
            return at_or_after_target[0]
        return candidates[-1]
    return hard_end


def _cross_page_boundary(
    left: PreparedChunkUnit,
    right: PreparedChunkUnit,
) -> tuple[ChunkBlockSlice, ChunkBlockSlice] | None:
    if (
        left.content_type is not ChunkContentType.BODY_TEXT
        or right.content_type is not ChunkContentType.BODY_TEXT
        or left.candidate.section_path != right.candidate.section_path
        or _has_heading_prefix(right)
    ):
        return None

    left_body_slices = tuple(
        block_slice
        for block_slice in left.candidate.block_slices
        if block_slice.block.block_type in _BODY_BLOCK_TYPES
    )
    right_body_slices = tuple(
        block_slice
        for block_slice in right.candidate.block_slices
        if block_slice.block.block_type in _BODY_BLOCK_TYPES
    )
    if not left_body_slices or not right_body_slices:
        return None
    left_slice = left_body_slices[-1]
    right_slice = right_body_slices[0]
    if right_slice.block.page_number != left_slice.block.page_number + 1:
        return None
    return left_slice, right_slice


def _cross_page_quality_diagnostics(
    left: ChunkBlockSlice,
    right: ChunkBlockSlice,
) -> tuple[ChunkingDiagnostic, ...]:
    diagnostics: list[ChunkingDiagnostic] = []
    if left.block.extraction_method is not right.block.extraction_method:
        diagnostics.append(
            _cross_page_diagnostic(
                ChunkingDiagnosticCode.CROSS_PAGE_EXTRACTION_METHOD_CHANGED,
                "Cross-page body uses different extraction methods.",
                left,
                right,
            )
        )
    if left.text.rstrip().endswith(("-", "‐", "‑", "–")):
        diagnostics.append(
            _cross_page_diagnostic(
                ChunkingDiagnosticCode.CROSS_PAGE_HYPHEN_PRESERVED,
                "Line-ending hyphen was preserved without source-text repair.",
                left,
                right,
            )
        )
    return tuple(diagnostics)


def _cross_page_diagnostic(
    code: ChunkingDiagnosticCode,
    message: str,
    left: ChunkBlockSlice,
    right: ChunkBlockSlice,
) -> ChunkingDiagnostic:
    return _diagnostic(
        code,
        WarningSeverity.INFO,
        message,
        pages=(left.block.page_number, right.block.page_number),
        blocks=(left.block.id, right.block.id),
    )


def _compose_units(units: tuple[PreparedChunkUnit, ...]) -> PreparedChunkUnit:
    first = units[0]
    candidate = ChunkCandidate(
        document_id=first.candidate.document_id,
        block_slices=tuple(
            block_slice for unit in units for block_slice in unit.candidate.block_slices
        ),
        section_path=first.candidate.section_path,
        routing_profile=first.candidate.routing_profile,
    )
    return PreparedChunkUnit(
        candidate=candidate,
        content_type=ChunkContentType.BODY_TEXT,
        warnings=_aggregate_chunk_warnings(
            tuple(warning for unit in units for warning in unit.warnings)
        ),
    )


def _same_body_group(left: PreparedChunkUnit, right: PreparedChunkUnit) -> bool:
    return (
        left.content_type is ChunkContentType.BODY_TEXT
        and right.content_type is ChunkContentType.BODY_TEXT
        and left.candidate.section_path == right.candidate.section_path
        and left.candidate.page_end == right.candidate.page_end
        and not _has_heading_prefix(right)
    )


def _has_heading_prefix(unit: PreparedChunkUnit) -> bool:
    return any(
        block_slice.block.block_type in {BlockType.TITLE, BlockType.HEADING}
        for block_slice in unit.candidate.block_slices[:-1]
    )


def _is_atomic_body_unit(
    unit: PreparedChunkUnit,
    config: PdfChunkingConfig,
) -> bool:
    body_slice = unit.candidate.block_slices[-1]
    return (
        body_slice.source_text_start != 0
        or body_slice.source_text_end != len(body_slice.block.text)
        or len(unit.candidate.content) > config.hard_max_size
    )


def _overlap_seed(
    previous: PreparedChunkUnit,
    following: PreparedChunkUnit,
    config: PdfChunkingConfig,
) -> PreparedChunkUnit | None:
    if config.overlap_max == 0:
        return None
    body_slice = previous.candidate.block_slices[-1]
    if (
        body_slice.block.block_type not in _BODY_BLOCK_TYPES
        or body_slice.source_text_start != 0
        or body_slice.source_text_end != len(body_slice.block.text)
        or len(body_slice.text) > config.overlap_max
    ):
        return None

    candidate = ChunkCandidate(
        document_id=previous.candidate.document_id,
        block_slices=(body_slice,),
        section_path=previous.candidate.section_path,
        routing_profile=previous.candidate.routing_profile,
    )
    if len(candidate.content) + 1 + len(following.candidate.content) > config.hard_max_size:
        return None
    return PreparedChunkUnit(
        candidate=candidate,
        content_type=ChunkContentType.BODY_TEXT,
        warnings=_bind_warnings_to_candidate(previous.warnings, candidate),
    )


def _aggregate_chunk_warnings(
    warnings: tuple[ChunkWarning, ...],
) -> tuple[ChunkWarning, ...]:
    grouped: dict[
        tuple[str, WarningSeverity, str],
        list[str],
    ] = {}
    for warning in warnings:
        key = (warning.code, warning.severity, warning.message)
        block_ids = grouped.setdefault(key, [])
        block_ids.extend(
            block_id for block_id in warning.source_block_ids if block_id not in block_ids
        )
    return tuple(
        ChunkWarning(
            code=code,
            severity=severity,
            message=message,
            source_block_ids=tuple(block_ids),
        )
        for (code, severity, message), block_ids in grouped.items()
    )


def _split_text_ranges(
    text: str,
    *,
    first_target_limit: int,
    first_hard_limit: int,
    target_limit: int,
    hard_limit: int,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        is_first = not ranges
        current_target = first_target_limit if is_first else target_limit
        current_hard = first_hard_limit if is_first else hard_limit
        if len(text) - start <= current_hard:
            end = len(text)
        else:
            end = _choose_text_boundary(
                text,
                start=start,
                target_limit=current_target,
                hard_limit=current_hard,
            )
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _choose_text_boundary(
    text: str,
    *,
    start: int,
    target_limit: int,
    hard_limit: int,
) -> int:
    target_end = min(len(text), start + target_limit)
    hard_end = min(len(text), start + hard_limit)
    minimum_end = start + max(1, target_limit // 2)
    window = text[start:hard_end]

    for pattern in (
        _SENTENCE_BOUNDARY_PATTERN,
        _NEWLINE_BOUNDARY_PATTERN,
        _WHITESPACE_BOUNDARY_PATTERN,
    ):
        candidates = [
            start + match.end()
            for match in pattern.finditer(window)
            if start + match.end() >= minimum_end
        ]
        if not candidates:
            continue
        at_or_after_target = [candidate for candidate in candidates if candidate >= target_end]
        if at_or_after_target:
            return at_or_after_target[0]
        return candidates[-1]

    return hard_end


def _page_block_reason(
    status: ParseStatus,
    warnings: tuple[str, ...],
) -> tuple[ChunkingDiagnosticCode, str] | None:
    if status is ParseStatus.NEEDS_OCR:
        return (
            ChunkingDiagnosticCode.PAGE_NEEDS_OCR_SKIPPED,
            "Page requires OCR and cannot enter chunking.",
        )
    if status is ParseStatus.FAILED:
        return (
            ChunkingDiagnosticCode.PAGE_PARSE_FAILED_SKIPPED,
            "Page parsing failed and cannot enter chunking.",
        )
    for warning in warnings:
        if warning in _BLOCKING_PAGE_WARNINGS:
            return (
                ChunkingDiagnosticCode.PAGE_BLOCKED_BY_WARNING,
                f"Page was blocked by source warning: {warning}.",
            )
        if warning not in _ALLOWED_PAGE_WARNINGS:
            return (
                ChunkingDiagnosticCode.UNKNOWN_PAGE_WARNING,
                f"Page was blocked by unknown source warning: {warning}.",
            )
    return None


def _block_warning_gate(
    block: ParsedBlock,
) -> tuple[ChunkingDiagnosticCode, str] | None:
    for warning in block.warnings:
        if warning in _BLOCKING_BLOCK_WARNINGS:
            return (
                ChunkingDiagnosticCode.BLOCK_BLOCKED_BY_WARNING,
                f"Block was blocked by source warning: {warning}.",
            )
        if not _is_allowed_block_warning(warning):
            return (
                ChunkingDiagnosticCode.UNKNOWN_BLOCK_WARNING,
                f"Block was blocked by unknown source warning: {warning}.",
            )
    return None


def _is_allowed_block_warning(warning: str) -> bool:
    return warning in _ALLOWED_BLOCK_WARNINGS or warning.startswith(_ALLOWED_BLOCK_WARNING_PREFIXES)


def _accepted_page_warnings(warnings: tuple[str, ...]) -> tuple[ChunkWarning, ...]:
    return tuple(
        ChunkWarning(
            code=warning,
            severity=(
                WarningSeverity.INFO
                if warning == "EMBEDDED_OCR_TEXT_LAYER"
                else WarningSeverity.WARNING
            ),
            message=f"Accepted source page warning: {warning}.",
        )
        for warning in warnings
        if warning in _ALLOWED_PAGE_WARNINGS
    )


def _accepted_block_warnings(block: ParsedBlock) -> tuple[ChunkWarning, ...]:
    return tuple(
        ChunkWarning(
            code=warning,
            severity=(
                WarningSeverity.INFO
                if warning == "PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS"
                else WarningSeverity.WARNING
            ),
            message=f"Accepted source block warning: {warning}.",
            source_block_ids=(block.id,),
        )
        for warning in block.warnings
        if _is_allowed_block_warning(warning)
    )


def _bind_warnings_to_block(
    warnings: tuple[ChunkWarning, ...],
    block: ParsedBlock,
) -> tuple[ChunkWarning, ...]:
    return tuple(
        ChunkWarning(
            code=warning.code,
            severity=warning.severity,
            message=warning.message,
            source_block_ids=(block.id,),
        )
        for warning in warnings
    )


def _accept_heading(
    section: _SectionState,
    block: ParsedBlock,
    warnings: tuple[ChunkWarning, ...],
) -> tuple[ChunkingDiagnostic, ...]:
    diagnostics: list[ChunkingDiagnostic] = []
    assert section.headings is not None
    assert section.pending_headings is not None
    assert section.pending_warnings is not None

    if block.block_type is BlockType.TITLE and not section.saw_title:
        level = 0
        section.root_title = block.text
        section.saw_title = True
    elif block.block_type is BlockType.TITLE:
        level = 1
        diagnostics.append(
            _heading_diagnostic(
                ChunkingDiagnosticCode.ADDITIONAL_TITLE_AS_HEADING,
                WarningSeverity.WARNING,
                "Additional document title was treated as a level-1 heading.",
                block,
            )
        )
    else:
        inferred_level = _infer_decimal_heading_level(block.text)
        if block.heading_level is not None:
            level = block.heading_level
            if inferred_level is not None and inferred_level != level:
                diagnostics.append(
                    _heading_diagnostic(
                        ChunkingDiagnosticCode.HEADING_LEVEL_CONFLICT,
                        WarningSeverity.WARNING,
                        "Parser heading level won over conflicting decimal numbering.",
                        block,
                    )
                )
        elif inferred_level is not None:
            level = inferred_level
            diagnostics.append(
                _heading_diagnostic(
                    ChunkingDiagnosticCode.HEADING_LEVEL_INFERRED,
                    WarningSeverity.INFO,
                    "Heading level was inferred from anchored decimal numbering.",
                    block,
                )
            )
        else:
            level = 1
            diagnostics.append(
                _heading_diagnostic(
                    ChunkingDiagnosticCode.HEADING_LEVEL_UNKNOWN,
                    WarningSeverity.WARNING,
                    "Heading without an explicit or decimal level was treated as level 1.",
                    block,
                )
            )

    if level > 0:
        existing_levels = sorted(section.headings)
        expected_next = (existing_levels[-1] + 1) if existing_levels else 1
        if level > expected_next:
            diagnostics.append(
                _heading_diagnostic(
                    ChunkingDiagnosticCode.HEADING_LEVEL_GAP,
                    WarningSeverity.WARNING,
                    "Heading level skipped one or more parent levels.",
                    block,
                )
            )

        closed_pending = [
            pending_block
            for pending_level, pending_block in section.pending_headings
            if pending_level >= level
        ]
        diagnostics.extend(_heading_without_body_diagnostics(tuple(closed_pending)))
        for closed_block in closed_pending:
            section.pending_warnings.pop(closed_block.id, None)
        section.pending_headings[:] = [
            (pending_level, pending_block)
            for pending_level, pending_block in section.pending_headings
            if pending_level < level
        ]
        for existing_level in tuple(section.headings):
            if existing_level >= level:
                del section.headings[existing_level]
        section.headings[level] = block.text

    section.pending_headings.append((level, block))
    section.pending_warnings[block.id] = warnings
    return tuple(diagnostics)


def _infer_decimal_heading_level(text: str) -> int | None:
    match = _DECIMAL_HEADING_PATTERN.match(text)
    if match is None:
        return None
    return match.group(1).count(".") + 1


def _bind_warnings_to_candidate(
    warnings: tuple[ChunkWarning, ...],
    candidate: ChunkCandidate,
) -> tuple[ChunkWarning, ...]:
    bound: list[ChunkWarning] = []
    candidate_ids = set(candidate.source_block_ids)
    for warning in warnings:
        source_ids = warning.source_block_ids or candidate.source_block_ids
        source_ids = tuple(block_id for block_id in source_ids if block_id in candidate_ids)
        if not source_ids:
            continue
        bound.append(
            ChunkWarning(
                code=warning.code,
                severity=warning.severity,
                message=warning.message,
                source_block_ids=source_ids,
            )
        )
    return tuple(bound)


def _heading_without_body_diagnostics(
    blocks: tuple[ParsedBlock, ...],
) -> tuple[ChunkingDiagnostic, ...]:
    return tuple(
        _heading_diagnostic(
            ChunkingDiagnosticCode.HEADING_WITHOUT_BODY,
            WarningSeverity.WARNING,
            "Heading produced no body chunk before its section closed.",
            block,
        )
        for block in blocks
    )


def _heading_diagnostic(
    code: ChunkingDiagnosticCode,
    severity: WarningSeverity,
    message: str,
    block: ParsedBlock,
) -> ChunkingDiagnostic:
    return _diagnostic(
        code,
        severity,
        message,
        pages=(block.page_number,),
        blocks=(block.id,),
    )


def _diagnostic(
    code: ChunkingDiagnosticCode,
    severity: WarningSeverity,
    message: str,
    *,
    pages: tuple[int, ...] = (),
    blocks: tuple[str, ...] = (),
) -> ChunkingDiagnostic:
    return ChunkingDiagnostic(
        code=code,
        severity=severity,
        message=message,
        page_numbers=pages,
        source_block_ids=blocks,
    )


def _aggregate_diagnostics(
    diagnostics: list[ChunkingDiagnostic],
) -> tuple[ChunkingDiagnostic, ...]:
    grouped: dict[
        tuple[ChunkingDiagnosticCode, WarningSeverity, str],
        tuple[list[int], list[str]],
    ] = {}
    for diagnostic in diagnostics:
        key = (diagnostic.code, diagnostic.severity, diagnostic.message)
        pages, block_ids = grouped.setdefault(key, ([], []))
        pages.extend(page for page in diagnostic.page_numbers if page not in pages)
        block_ids.extend(
            block_id for block_id in diagnostic.source_block_ids if block_id not in block_ids
        )
    return tuple(
        ChunkingDiagnostic(
            code=code,
            severity=severity,
            message=message,
            page_numbers=tuple(sorted(pages)),
            source_block_ids=tuple(block_ids),
        )
        for (code, severity, message), (pages, block_ids) in grouped.items()
    )

"""Gate, embed, and atomically index traceable PDF chunks."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceResult,
    DocumentReplaceStatus,
    VectorIndex,
    VectorRecord,
)
from ai_learning_rag.embedding import TextEmbedder
from ai_learning_rag.pdf_schema import (
    ChunkContentType,
    ExtractionMethod,
    ExtractionProvenance,
    TraceablePdfChunk,
    WarningSeverity,
)


class TokenCounter(Protocol):
    """Model-tokenizer boundary used to reject inputs before encoding."""

    def count_tokens(self, text: str) -> int:
        """Return the untruncated input token count, including special tokens."""


class ChunkRejectionCode(StrEnum):
    """Stable, auditable reasons why a chunk did not enter the vector index."""

    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    EMPTY_CONTENT = "empty_content"
    ERROR_WARNING = "error_warning"
    UNKNOWN_WARNING = "unknown_warning"
    MISSING_PROVENANCE = "missing_provenance"
    UNKNOWN_PROVENANCE = "unknown_provenance"
    UNRESOLVED_OCR = "unresolved_ocr"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"


@dataclass(frozen=True, slots=True)
class PdfIndexingPolicy:
    """Versioned fail-closed rules for admitting PDF chunks to embedding."""

    profile: str = "pdf-index-gate-v1"
    document_input_profile: str = "section-path-prefix-v1"
    provenance_schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("profile must not be blank")
        if not self.document_input_profile.strip():
            raise ValueError("document_input_profile must not be blank")
        if self.provenance_schema_version != 1:
            raise ValueError("unsupported provenance_schema_version")


@dataclass(frozen=True, slots=True)
class ChunkIndexDecision:
    """One accepted or rejected chunk decision for a human-readable report."""

    chunk_id: str
    chunk_fingerprint: str
    content_type: str
    page_numbers: tuple[int, ...]
    source_block_ids: tuple[str, ...]
    accepted: bool
    token_count: int | None = None
    rejection_code: ChunkRejectionCode | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.content_type.strip():
            raise ValueError("content_type must not be blank")
        if not self.page_numbers or any(page < 1 for page in self.page_numbers):
            raise ValueError("page_numbers must contain positive page numbers")
        if tuple(sorted(set(self.page_numbers))) != self.page_numbers:
            raise ValueError("page_numbers must be sorted and unique")
        if not self.source_block_ids or any(
            not block_id.strip() for block_id in self.source_block_ids
        ):
            raise ValueError("source_block_ids must contain non-blank ids")
        if tuple(dict.fromkeys(self.source_block_ids)) != self.source_block_ids:
            raise ValueError("source_block_ids must be unique")
        if self.accepted and self.rejection_code is not None:
            raise ValueError("accepted decisions must not define a rejection code")
        if not self.accepted and self.rejection_code is None:
            raise ValueError("rejected decisions must define a rejection code")


@dataclass(frozen=True, slots=True)
class PdfIndexingReport:
    """Auditable result of applying the indexing gate to one document."""

    document_id: str
    index_id: str
    generation_id: str
    gate_profile: str
    decisions: tuple[ChunkIndexDecision, ...]

    @property
    def accepted_count(self) -> int:
        """Return the number of chunks sent to embedding."""

        return sum(decision.accepted for decision in self.decisions)

    @property
    def rejected_count(self) -> int:
        """Return the number of chunks rejected before embedding."""

        return len(self.decisions) - self.accepted_count

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible report payload."""

        return {
            "schema_version": 1,
            "document_id": self.document_id,
            "index_id": self.index_id,
            "generation_id": self.generation_id,
            "gate_profile": self.gate_profile,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "decisions": [
                {
                    "chunk_id": decision.chunk_id,
                    "chunk_fingerprint": decision.chunk_fingerprint,
                    "content_type": decision.content_type,
                    "page_numbers": list(decision.page_numbers),
                    "source_block_ids": list(decision.source_block_ids),
                    "accepted": decision.accepted,
                    "token_count": decision.token_count,
                    "rejection_code": (
                        None if decision.rejection_code is None else decision.rejection_code.value
                    ),
                    "detail": decision.detail,
                }
                for decision in self.decisions
            ],
        }

    def to_json(self) -> str:
        """Render deterministic, UTF-8-friendly JSON for review artifacts."""

        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        """Render the same decisions as a compact human-review table."""

        lines = [
            "# PDF indexing gate report",
            "",
            f"- Document: `{self.document_id}`",
            f"- Index: `{self.index_id}`",
            f"- Generation: `{self.generation_id}`",
            f"- Gate profile: `{self.gate_profile}`",
            f"- Accepted: {self.accepted_count}",
            f"- Rejected: {self.rejected_count}",
            "",
            "| Chunk | Type | Pages | Source block IDs | Accepted | Tokens | Reason |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
        for decision in self.decisions:
            reason = "" if decision.rejection_code is None else decision.rejection_code.value
            tokens = "" if decision.token_count is None else str(decision.token_count)
            pages = ", ".join(str(page) for page in decision.page_numbers)
            block_ids = "<br>".join(f"`{block_id}`" for block_id in decision.source_block_ids)
            lines.append(
                f"| `{decision.chunk_id}` | {decision.content_type} | {pages} | {block_ids} "
                f"| {'yes' if decision.accepted else 'no'} | {tokens} | {reason} |"
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class PdfIndexingResult:
    """Combined gate report and optional atomic replacement result."""

    report: PdfIndexingReport | None
    replacement: DocumentReplaceResult | None


@dataclass(frozen=True, slots=True)
class _AcceptedChunk:
    chunk: TraceablePdfChunk
    embedding_input: str
    token_count: int


_ALLOWED_CONTENT_TYPES = frozenset(
    {
        ChunkContentType.BODY_TEXT,
        ChunkContentType.TABLE,
        ChunkContentType.FORMULA,
        ChunkContentType.FIGURE_CAPTION,
        ChunkContentType.CODE,
    }
)
_KNOWN_WARNING_CODES = frozenset(
    {
        "EMBEDDED_OCR_TEXT_LAYER",
        "EMBEDDED_OCR_VERIFIED",
        "RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER",
        "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR",
        "MANUAL_REVIEW_ACCEPTED_RERUN_OCR",
        "DOCLING_INVALID_BBOX",
        "DOCLING_PICTURE_PARENT_MISSING",
        "PICTURE_TEXT_EXCLUDED_FROM_BODY_CHUNKS",
    }
)
_KNOWN_WARNING_PREFIXES = ("DOCLING_UNMAPPED_LABEL_",)
_OCR_ACCEPTANCE_CODES = frozenset(
    {
        "EMBEDDED_OCR_TEXT_LAYER",
        "EMBEDDED_OCR_VERIFIED",
        "RERUN_OCR_REPLACED_BROKEN_TEXT_LAYER",
        "MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR",
        "MANUAL_REVIEW_ACCEPTED_RERUN_OCR",
    }
)
_OCR_METHODS = frozenset({ExtractionMethod.EMBEDDED_OCR, ExtractionMethod.RERUN_OCR})


class PdfChunkIndexer:
    """Own PDF-specific gating and mapping while hiding persistence details."""

    def __init__(
        self,
        *,
        index: VectorIndex,
        embedder: TextEmbedder,
        token_counter: TokenCounter,
        max_input_tokens: int,
        policy: PdfIndexingPolicy | None = None,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero")
        self._index = index
        self._embedder = embedder
        self._token_counter = token_counter
        self._max_input_tokens = max_input_tokens
        self._policy = PdfIndexingPolicy() if policy is None else policy

    def index_document(
        self,
        *,
        generation: DocumentGeneration,
        chunks: Sequence[TraceablePdfChunk],
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> PdfIndexingResult:
        """Gate and embed outside storage, then atomically replace one document."""

        state = self._index.get_document_state(
            generation.index_id,
            generation.document_id,
        )
        current = state.active_generation
        if current is not None and current.generation_id == generation.generation_id:
            return PdfIndexingResult(
                report=None,
                replacement=DocumentReplaceResult(
                    status=DocumentReplaceStatus.ALREADY_INDEXED,
                    generation_id=current.generation_id,
                    previous_generation_id=current.generation_id,
                    record_count=None,
                    document_revision=state.revision,
                ),
            )

        accepted, report = self._gate_chunks(
            generation=generation,
            chunks=chunks,
            provenance_registry=provenance_registry,
        )
        if not accepted:
            return PdfIndexingResult(report=report, replacement=None)

        vectors = self._embedder.embed_documents(
            [candidate.embedding_input for candidate in accepted]
        )
        if len(vectors) != len(accepted):
            raise RuntimeError("embedder returned an unexpected number of vectors")

        records = tuple(
            self._to_record(
                generation,
                candidate,
                tuple(vector),
                provenance_registry,
            )
            for candidate, vector in zip(accepted, vectors, strict=True)
        )
        replacement = self._index.replace_document(
            generation,
            records,
            expected_revision=state.revision,
        )
        return PdfIndexingResult(report=report, replacement=replacement)

    def build_gate_report(
        self,
        *,
        generation: DocumentGeneration,
        chunks: Sequence[TraceablePdfChunk],
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> PdfIndexingReport:
        """Rebuild an audit report without embedding or changing the index."""

        _accepted, report = self._gate_chunks(
            generation=generation,
            chunks=chunks,
            provenance_registry=provenance_registry,
        )
        return report

    def _gate_chunks(
        self,
        *,
        generation: DocumentGeneration,
        chunks: Sequence[TraceablePdfChunk],
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> tuple[list[_AcceptedChunk], PdfIndexingReport]:
        accepted: list[_AcceptedChunk] = []
        decisions: list[ChunkIndexDecision] = []
        for chunk in chunks:
            if chunk.document_id != generation.document_id:
                raise ValueError("every chunk must belong to the generation document")
            candidate, decision = self._apply_gate(chunk, provenance_registry)
            decisions.append(decision)
            if candidate is not None:
                accepted.append(candidate)

        return accepted, PdfIndexingReport(
            document_id=generation.document_id,
            index_id=generation.index_id,
            generation_id=generation.generation_id,
            gate_profile=self._policy.profile,
            decisions=tuple(decisions),
        )

    def _apply_gate(
        self,
        chunk: TraceablePdfChunk,
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> tuple[_AcceptedChunk | None, ChunkIndexDecision]:
        decision_source = {
            "content_type": chunk.content_type.value,
            "page_numbers": tuple(sorted({region.page_number for region in chunk.source_regions})),
            "source_block_ids": tuple(
                dict.fromkeys(region.block_id for region in chunk.source_regions)
            ),
        }
        rejection = self._gate_rejection(chunk, provenance_registry)
        if rejection is not None:
            code, detail = rejection
            return None, ChunkIndexDecision(
                chunk_id=chunk.id,
                chunk_fingerprint=chunk.source_fingerprint,
                **decision_source,
                accepted=False,
                rejection_code=code,
                detail=detail,
            )

        embedding_input = build_document_embedding_input(chunk)
        token_count = self._token_counter.count_tokens(embedding_input)
        if token_count <= 0:
            raise RuntimeError("token counter returned a non-positive token count")
        if token_count > self._max_input_tokens:
            return None, ChunkIndexDecision(
                chunk_id=chunk.id,
                chunk_fingerprint=chunk.source_fingerprint,
                **decision_source,
                accepted=False,
                token_count=token_count,
                rejection_code=ChunkRejectionCode.TOKEN_LIMIT_EXCEEDED,
                detail=f"{token_count} tokens exceeds limit {self._max_input_tokens}",
            )
        return _AcceptedChunk(chunk, embedding_input, token_count), ChunkIndexDecision(
            chunk_id=chunk.id,
            chunk_fingerprint=chunk.source_fingerprint,
            **decision_source,
            accepted=True,
            token_count=token_count,
        )

    @staticmethod
    def _gate_rejection(
        chunk: TraceablePdfChunk,
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> tuple[ChunkRejectionCode, str] | None:
        if chunk.content_type not in _ALLOWED_CONTENT_TYPES:
            return ChunkRejectionCode.UNSUPPORTED_CONTENT_TYPE, chunk.content_type.value
        if not chunk.content.strip():
            return ChunkRejectionCode.EMPTY_CONTENT, "chunk content is blank"

        warning_codes = {warning.code for warning in chunk.warnings}
        if any(warning.severity is WarningSeverity.ERROR for warning in chunk.warnings):
            return ChunkRejectionCode.ERROR_WARNING, "chunk contains an ERROR warning"
        unknown_codes = sorted(
            code
            for code in warning_codes
            if code not in _KNOWN_WARNING_CODES
            and not any(code.startswith(prefix) for prefix in _KNOWN_WARNING_PREFIXES)
        )
        if unknown_codes:
            return ChunkRejectionCode.UNKNOWN_WARNING, ", ".join(unknown_codes)

        provenance_ids = [region.extraction_provenance_id for region in chunk.source_regions]
        if any(provenance_id is None for provenance_id in provenance_ids):
            return ChunkRejectionCode.MISSING_PROVENANCE, "source region has no provenance id"
        unknown_ids = sorted(
            provenance_id
            for provenance_id in provenance_ids
            if provenance_id not in provenance_registry
        )
        if unknown_ids:
            return ChunkRejectionCode.UNKNOWN_PROVENANCE, ", ".join(unknown_ids)

        if any(
            region.extraction_method in _OCR_METHODS for region in chunk.source_regions
        ) and not warning_codes.intersection(_OCR_ACCEPTANCE_CODES):
            return ChunkRejectionCode.UNRESOLVED_OCR, "OCR source lacks an accepted decision"
        return None

    def _to_record(
        self,
        generation: DocumentGeneration,
        accepted: _AcceptedChunk,
        vector: tuple[float, ...],
        provenance_registry: Mapping[str, ExtractionProvenance],
    ) -> VectorRecord:
        chunk = accepted.chunk
        record_identity = (
            f"{generation.index_id}\0{generation.generation_id}\0{chunk.source_fingerprint}"
        )
        return VectorRecord(
            record_id=f"rec_{sha256(record_identity.encode()).hexdigest()}",
            index_id=generation.index_id,
            generation_id=generation.generation_id,
            document_id=generation.document_id,
            chunk_fingerprint=chunk.source_fingerprint,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            content_type=chunk.content_type.value,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            embedding_input_hash=sha256(accepted.embedding_input.encode()).hexdigest(),
            embedding_input_token_count=accepted.token_count,
            vector=vector,
            provenance=_serialize_provenance(
                chunk,
                provenance_registry,
                schema_version=self._policy.provenance_schema_version,
            ),
            provenance_schema_version=self._policy.provenance_schema_version,
        )


def build_document_embedding_input(chunk: TraceablePdfChunk) -> str:
    """Prefix source content with its deterministic section path."""

    if not chunk.section_path:
        return chunk.content
    return f"{' > '.join(chunk.section_path)}\n\n{chunk.content}"


def _serialize_provenance(
    chunk: TraceablePdfChunk,
    registry: Mapping[str, ExtractionProvenance],
    *,
    schema_version: int,
) -> dict[str, object]:
    provenance_ids = tuple(
        dict.fromkeys(
            region.extraction_provenance_id
            for region in chunk.source_regions
            if region.extraction_provenance_id is not None
        )
    )
    return {
        "schema_version": schema_version,
        "chunk_id": chunk.id,
        "routing_profile": chunk.routing_profile,
        "chunking_profile": chunk.chunking_profile,
        "source_regions": [
            {
                "block_id": region.block_id,
                "page_number": region.page_number,
                "bbox": list(region.bbox.as_tuple()),
                "block_type": region.block_type.value,
                "extraction_method": region.extraction_method.value,
                "reading_order": region.reading_order,
                "chunk_content_start": region.chunk_content_start,
                "chunk_content_end": region.chunk_content_end,
                "source_text_start": region.source_text_start,
                "source_text_end": region.source_text_end,
                "parent_picture_id": region.parent_picture_id,
                "confidence": region.confidence,
                "warnings": list(region.warnings),
                "extraction_provenance_id": region.extraction_provenance_id,
            }
            for region in chunk.source_regions
        ],
        "warnings": [
            {
                "code": warning.code,
                "severity": warning.severity.value,
                "message": warning.message,
                "source_block_ids": list(warning.source_block_ids),
            }
            for warning in chunk.warnings
        ],
        "extraction_provenances": [
            {
                "id": provenance.id,
                "parser_name": provenance.parser_name,
                "parser_version": provenance.parser_version,
                "extraction_method": provenance.extraction_method.value,
                "configuration_profile": provenance.configuration_profile,
                "ocr_engine_name": provenance.ocr_engine_name,
                "ocr_engine_version": provenance.ocr_engine_version,
            }
            for provenance_id in provenance_ids
            for provenance in (registry[provenance_id],)
        ],
    }

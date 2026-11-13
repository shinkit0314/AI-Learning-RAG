"""Document-level vector index contracts shared by memory and database backends."""

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Protocol

from ai_learning_rag.vector_math import cosine_similarity


class IndexStatus(StrEnum):
    """Lifecycle state for one immutable embedding space."""

    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    FAILED = "failed"
    RETIRED = "retired"


class DocumentReplaceStatus(StrEnum):
    """Observable outcome of one atomic document replacement."""

    INSERTED = "inserted"
    REPLACED = "replaced"
    ALREADY_INDEXED = "already_indexed"
    CONCURRENT_UPDATE = "concurrent_update"


def _canonical_sha256(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode()).hexdigest()


def _require_text(value: str, *, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")


def _require_sha256(value: str, *, label: str) -> None:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must contain 64 hexadecimal characters")


@dataclass(frozen=True, slots=True)
class VectorIndexManifest:
    """Complete semantic identity and lifecycle state of an embedding space."""

    index_id: str
    logical_name: str
    embedding_dimension: int
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    normalized: bool
    pooling: str
    dtype: str
    query_profile: str
    document_input_profile: str
    max_input_tokens: int
    oversize_policy: str
    adapter_version: str
    required_pgvector_version: str
    status: IndexStatus = IndexStatus.BUILDING
    audit_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("index_id", self.index_id),
            ("logical_name", self.logical_name),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_id", self.tokenizer_id),
            ("tokenizer_revision", self.tokenizer_revision),
            ("pooling", self.pooling),
            ("dtype", self.dtype),
            ("query_profile", self.query_profile),
            ("document_input_profile", self.document_input_profile),
            ("oversize_policy", self.oversize_policy),
            ("adapter_version", self.adapter_version),
            ("required_pgvector_version", self.required_pgvector_version),
        ):
            _require_text(value, label=label)
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero")
        if any(not key.strip() or not value.strip() for key, value in self.audit_metadata.items()):
            raise ValueError("audit_metadata keys and values must not be blank")
        if self.index_id != self.expected_index_id:
            raise ValueError("index_id does not match the embedding-space identity")

    @classmethod
    def create(
        cls,
        *,
        logical_name: str,
        embedding_dimension: int,
        model_id: str,
        model_revision: str,
        tokenizer_id: str,
        tokenizer_revision: str,
        normalized: bool,
        pooling: str,
        dtype: str,
        query_profile: str,
        document_input_profile: str,
        max_input_tokens: int,
        oversize_policy: str,
        adapter_version: str,
        required_pgvector_version: str,
        status: IndexStatus = IndexStatus.BUILDING,
        audit_metadata: dict[str, str] | None = None,
    ) -> "VectorIndexManifest":
        """Build a manifest whose ID covers every semantic vector-space setting."""

        identity = {
            "adapter_version": adapter_version,
            "document_input_profile": document_input_profile,
            "dtype": dtype,
            "embedding_dimension": embedding_dimension,
            "logical_name": logical_name,
            "max_input_tokens": max_input_tokens,
            "model_id": model_id,
            "model_revision": model_revision,
            "normalized": normalized,
            "oversize_policy": oversize_policy,
            "pooling": pooling,
            "query_profile": query_profile,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        }
        return cls(
            index_id=f"idx_{_canonical_sha256(identity)}",
            logical_name=logical_name,
            embedding_dimension=embedding_dimension,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            normalized=normalized,
            pooling=pooling,
            dtype=dtype,
            query_profile=query_profile,
            document_input_profile=document_input_profile,
            max_input_tokens=max_input_tokens,
            oversize_policy=oversize_policy,
            adapter_version=adapter_version,
            required_pgvector_version=required_pgvector_version,
            status=status,
            audit_metadata={} if audit_metadata is None else dict(audit_metadata),
        )

    @property
    def expected_index_id(self) -> str:
        """Recompute the stable ID without runtime-only audit metadata or status."""

        return f"idx_{_canonical_sha256(self.identity_payload)}"

    @property
    def identity_payload(self) -> dict[str, object]:
        """Return the canonical fields that define vector compatibility."""

        return {
            "adapter_version": self.adapter_version,
            "document_input_profile": self.document_input_profile,
            "dtype": self.dtype,
            "embedding_dimension": self.embedding_dimension,
            "logical_name": self.logical_name,
            "max_input_tokens": self.max_input_tokens,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "normalized": self.normalized,
            "oversize_policy": self.oversize_policy,
            "pooling": self.pooling,
            "query_profile": self.query_profile,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class DocumentGeneration:
    """Immutable identity of one document processing result inside an index."""

    generation_id: str
    index_id: str
    document_id: str
    file_sha256: str
    routing_profile: str
    parser_profile: str
    ocr_decision_fingerprint: str
    chunking_profile: str
    gate_profile: str
    provenance_registry_fingerprint: str

    def __post_init__(self) -> None:
        for label, value in (
            ("generation_id", self.generation_id),
            ("index_id", self.index_id),
            ("document_id", self.document_id),
            ("routing_profile", self.routing_profile),
            ("parser_profile", self.parser_profile),
            ("chunking_profile", self.chunking_profile),
            ("gate_profile", self.gate_profile),
        ):
            _require_text(value, label=label)
        for label, value in (
            ("file_sha256", self.file_sha256),
            ("ocr_decision_fingerprint", self.ocr_decision_fingerprint),
            ("provenance_registry_fingerprint", self.provenance_registry_fingerprint),
        ):
            _require_sha256(value, label=label)
        if self.generation_id != self.expected_generation_id:
            raise ValueError("generation_id does not match the document processing identity")

    @classmethod
    def create(
        cls,
        *,
        index_id: str,
        document_id: str,
        file_sha256: str,
        routing_profile: str,
        parser_profile: str,
        ocr_decision_fingerprint: str,
        chunking_profile: str,
        gate_profile: str,
        provenance_registry_fingerprint: str,
    ) -> "DocumentGeneration":
        """Create a full-length generation ID from every replacement-sensitive input."""

        identity = {
            "chunking_profile": chunking_profile,
            "document_id": document_id,
            "file_sha256": file_sha256.lower(),
            "gate_profile": gate_profile,
            "index_id": index_id,
            "ocr_decision_fingerprint": ocr_decision_fingerprint.lower(),
            "parser_profile": parser_profile,
            "provenance_registry_fingerprint": provenance_registry_fingerprint.lower(),
            "routing_profile": routing_profile,
        }
        return cls(
            generation_id=f"gen_{_canonical_sha256(identity)}",
            index_id=index_id,
            document_id=document_id,
            file_sha256=file_sha256.lower(),
            routing_profile=routing_profile,
            parser_profile=parser_profile,
            ocr_decision_fingerprint=ocr_decision_fingerprint.lower(),
            chunking_profile=chunking_profile,
            gate_profile=gate_profile,
            provenance_registry_fingerprint=provenance_registry_fingerprint.lower(),
        )

    @property
    def expected_generation_id(self) -> str:
        """Recompute the generation ID for persistence validation."""

        identity = {
            "chunking_profile": self.chunking_profile,
            "document_id": self.document_id,
            "file_sha256": self.file_sha256.lower(),
            "gate_profile": self.gate_profile,
            "index_id": self.index_id,
            "ocr_decision_fingerprint": self.ocr_decision_fingerprint.lower(),
            "parser_profile": self.parser_profile,
            "provenance_registry_fingerprint": self.provenance_registry_fingerprint.lower(),
            "routing_profile": self.routing_profile,
        }
        return f"gen_{_canonical_sha256(identity)}"


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One traceable chunk vector belonging to exactly one document generation."""

    record_id: str
    index_id: str
    generation_id: str
    document_id: str
    chunk_fingerprint: str
    chunk_index: int
    content: str
    content_type: str
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    embedding_input_hash: str
    embedding_input_token_count: int
    vector: tuple[float, ...]
    provenance: dict[str, object]
    provenance_schema_version: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("record_id", self.record_id),
            ("index_id", self.index_id),
            ("generation_id", self.generation_id),
            ("document_id", self.document_id),
            ("content", self.content),
            ("content_type", self.content_type),
        ):
            _require_text(value, label=label)
        _require_sha256(self.chunk_fingerprint, label="chunk_fingerprint")
        _require_sha256(self.embedding_input_hash, label="embedding_input_hash")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("section_path entries must not be blank")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("page range must be positive and ordered")
        if self.embedding_input_token_count <= 0:
            raise ValueError("embedding_input_token_count must be greater than zero")
        if self.provenance_schema_version != 1:
            raise ValueError("unsupported provenance_schema_version")
        if self.provenance.get("schema_version") != self.provenance_schema_version:
            raise ValueError("provenance schema_version does not match the record")


@dataclass(frozen=True, slots=True)
class VectorSearchQuery:
    """Exact-search request scoped to one active embedding space."""

    index_id: str
    vector: tuple[float, ...]
    top_k: int = 5

    def __post_init__(self) -> None:
        _require_text(self.index_id, label="index_id")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieved record and its cosine similarity score."""

    record: VectorRecord
    score: float


@dataclass(frozen=True, slots=True)
class DocumentIndexState:
    """Current published generation and optimistic revision for one document slot."""

    active_generation: DocumentGeneration | None
    revision: int

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("document revision must not be negative")
        if self.active_generation is None and self.revision != 0:
            raise ValueError("an empty document slot must have revision zero")


@dataclass(frozen=True, slots=True)
class DocumentReplaceResult:
    """Result returned after an atomic generation replacement."""

    status: DocumentReplaceStatus
    generation_id: str
    previous_generation_id: str | None
    record_count: int | None
    document_revision: int

    def __post_init__(self) -> None:
        if self.document_revision < 0:
            raise ValueError("document_revision must not be negative")


class VectorIndex(Protocol):
    """Database-independent, document-level vector index boundary."""

    def get_manifest(self, index_id: str) -> VectorIndexManifest | None:
        """Return current immutable identity and publish state for one index."""

    def ensure_index(self, manifest: VectorIndexManifest) -> None:
        """Create an index manifest or apply a valid lifecycle transition."""

    def get_active_generation(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentGeneration | None:
        """Return the currently visible generation for a document slot."""

    def get_document_state(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentIndexState:
        """Return the active generation and revision observed before expensive work."""

    def replace_document(
        self,
        generation: DocumentGeneration,
        records: tuple[VectorRecord, ...],
        *,
        expected_revision: int,
    ) -> DocumentReplaceResult:
        """Publish only if the document revision still matches the caller's snapshot."""

    def search(self, query: VectorSearchQuery) -> list[SearchResult]:
        """Search records from active document generations only."""


class InMemoryVectorIndex:
    """Reference implementation of document replacement and exact search semantics."""

    _ALLOWED_TRANSITIONS = {
        IndexStatus.BUILDING: {IndexStatus.BUILDING, IndexStatus.READY, IndexStatus.FAILED},
        IndexStatus.READY: {IndexStatus.READY, IndexStatus.ACTIVE, IndexStatus.FAILED},
        IndexStatus.ACTIVE: {IndexStatus.ACTIVE, IndexStatus.RETIRED},
        IndexStatus.FAILED: {IndexStatus.FAILED},
        IndexStatus.RETIRED: {IndexStatus.RETIRED},
    }

    def __init__(self) -> None:
        self._manifests: dict[str, VectorIndexManifest] = {}
        self._generations: dict[str, DocumentGeneration] = {}
        self._records_by_generation: dict[str, tuple[VectorRecord, ...]] = {}
        self._active_generation_ids: dict[tuple[str, str], str] = {}
        self._document_revisions: dict[tuple[str, str], int] = {}

    def ensure_index(self, manifest: VectorIndexManifest) -> None:
        """Create a building manifest or apply a safe publish-state transition."""

        current = self._manifests.get(manifest.index_id)
        if current is None:
            if manifest.status is not IndexStatus.BUILDING:
                raise ValueError("a new index must start in building status")
            self._manifests[manifest.index_id] = manifest
            return

        if current.identity_payload != manifest.identity_payload:
            raise ValueError("index identity fields cannot change in place")
        if manifest.status not in self._ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(
                "invalid index status transition: "
                f"{current.status.value} -> {manifest.status.value}"
            )

        if manifest.status is IndexStatus.ACTIVE:
            for index_id, other in tuple(self._manifests.items()):
                if (
                    index_id != manifest.index_id
                    and other.logical_name == manifest.logical_name
                    and other.status is IndexStatus.ACTIVE
                ):
                    self._manifests[index_id] = replace(other, status=IndexStatus.RETIRED)
        self._manifests[manifest.index_id] = manifest

    def get_manifest(self, index_id: str) -> VectorIndexManifest | None:
        """Return current immutable identity and publish state."""

        return self._manifests.get(index_id)

    def get_active_generation(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentGeneration | None:
        """Return the generation selected by the stable document slot."""

        return self.get_document_state(index_id, document_id).active_generation

    def get_document_state(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentIndexState:
        """Return one atomic in-memory view used for optimistic publication."""

        slot = (index_id, document_id)
        generation_id = self._active_generation_ids.get(slot)
        generation = None if generation_id is None else self._generations[generation_id]
        return DocumentIndexState(
            active_generation=generation,
            revision=self._document_revisions.get(slot, 0),
        )

    def replace_document(
        self,
        generation: DocumentGeneration,
        records: tuple[VectorRecord, ...],
        *,
        expected_revision: int,
    ) -> DocumentReplaceResult:
        """Validate the complete candidate before changing any visible state."""

        if expected_revision < 0:
            raise ValueError("expected_revision must not be negative")

        manifest = self._manifests.get(generation.index_id)
        if manifest is None:
            raise ValueError("generation references an unknown index")
        if manifest.status in {IndexStatus.FAILED, IndexStatus.RETIRED}:
            raise ValueError("cannot write to a failed or retired index")

        slot = (generation.index_id, generation.document_id)
        previous_generation_id = self._active_generation_ids.get(slot)
        current_revision = self._document_revisions.get(slot, 0)
        if previous_generation_id == generation.generation_id:
            return DocumentReplaceResult(
                status=DocumentReplaceStatus.ALREADY_INDEXED,
                generation_id=generation.generation_id,
                previous_generation_id=previous_generation_id,
                record_count=len(self._records_by_generation[generation.generation_id]),
                document_revision=current_revision,
            )
        if current_revision != expected_revision:
            return DocumentReplaceResult(
                status=DocumentReplaceStatus.CONCURRENT_UPDATE,
                generation_id=generation.generation_id,
                previous_generation_id=previous_generation_id,
                record_count=None,
                document_revision=current_revision,
            )

        self._validate_records(generation, records, manifest)

        self._generations[generation.generation_id] = generation
        self._records_by_generation[generation.generation_id] = records
        self._active_generation_ids[slot] = generation.generation_id
        next_revision = current_revision + 1
        self._document_revisions[slot] = next_revision
        return DocumentReplaceResult(
            status=(
                DocumentReplaceStatus.INSERTED
                if previous_generation_id is None
                else DocumentReplaceStatus.REPLACED
            ),
            generation_id=generation.generation_id,
            previous_generation_id=previous_generation_id,
            record_count=len(records),
            document_revision=next_revision,
        )

    def search(self, query: VectorSearchQuery) -> list[SearchResult]:
        """Rank active-generation records with exact cosine similarity."""

        manifest = self._manifests.get(query.index_id)
        if manifest is None:
            raise ValueError("query references an unknown index")
        if manifest.status is not IndexStatus.ACTIVE:
            raise ValueError("query index must be active")
        self._validate_vector(query.vector, manifest.embedding_dimension, label="query vector")

        active_generation_ids = {
            generation_id
            for (index_id, _document_id), generation_id in self._active_generation_ids.items()
            if index_id == query.index_id
        }
        results = [
            SearchResult(record=record, score=cosine_similarity(query.vector, record.vector))
            for generation_id in active_generation_ids
            for record in self._records_by_generation[generation_id]
        ]
        results.sort(key=lambda result: (-result.score, result.record.record_id))
        return results[: query.top_k]

    @classmethod
    def _validate_records(
        cls,
        generation: DocumentGeneration,
        records: tuple[VectorRecord, ...],
        manifest: VectorIndexManifest,
    ) -> None:
        if not records:
            raise ValueError("document generation must contain at least one record")
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record ids must be unique within a generation")
        for record in records:
            if record.index_id != generation.index_id:
                raise ValueError("record index_id does not match the generation")
            if record.generation_id != generation.generation_id:
                raise ValueError("record generation_id does not match the generation")
            if record.document_id != generation.document_id:
                raise ValueError("record document_id does not match the generation")
            cls._validate_vector(
                record.vector,
                manifest.embedding_dimension,
                label="record vector",
            )

    @staticmethod
    def _validate_vector(vector: tuple[float, ...], dimension: int, *, label: str) -> None:
        if len(vector) != dimension:
            raise ValueError(f"{label} must have exactly {dimension} dimensions")
        if not all(isfinite(value) for value in vector):
            raise ValueError(f"{label} must contain only finite values")
        if not any(value != 0.0 for value in vector):
            raise ValueError(f"{label} must not be a zero vector")

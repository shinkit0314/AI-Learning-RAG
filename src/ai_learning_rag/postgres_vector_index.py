"""PostgreSQL/pgvector implementation of the document-level index contract."""

from math import isfinite
from re import findall
from typing import Any

from sqlalchemy import Connection, Engine, text

from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentIndexState,
    DocumentReplaceResult,
    DocumentReplaceStatus,
    IndexStatus,
    SearchResult,
    VectorIndexManifest,
    VectorRecord,
    VectorSearchQuery,
)

_ALLOWED_TRANSITIONS = {
    IndexStatus.BUILDING: {IndexStatus.BUILDING, IndexStatus.READY, IndexStatus.FAILED},
    IndexStatus.READY: {IndexStatus.READY, IndexStatus.ACTIVE, IndexStatus.FAILED},
    IndexStatus.ACTIVE: {IndexStatus.ACTIVE, IndexStatus.RETIRED},
    IndexStatus.FAILED: {IndexStatus.FAILED},
    IndexStatus.RETIRED: {IndexStatus.RETIRED},
}


class PostgresVectorIndex:
    """Persist complete document generations and search active records exactly."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_manifest(self, index_id: str) -> VectorIndexManifest | None:
        """Return current immutable identity and publish state."""

        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT index_id, logical_name, embedding_dimension, identity,
                               audit_metadata, required_pgvector_version, status
                        FROM rag.vector_indexes
                        WHERE index_id = :index_id
                        """
                    ),
                    {"index_id": index_id},
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else _manifest_from_row(row)

    def ensure_index(self, manifest: VectorIndexManifest) -> None:
        """Create a building manifest or apply an atomic publish transition."""

        with self._engine.begin() as connection:
            extension_version = connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            if extension_version is None:
                raise RuntimeError("pgvector extension is not installed in the current database")
            if not _version_at_least(
                str(extension_version),
                manifest.required_pgvector_version,
            ):
                raise RuntimeError(
                    "pgvector version is too old: "
                    f"{extension_version} < {manifest.required_pgvector_version}"
                )

            current_row = (
                connection.execute(
                    text(
                        """
                    SELECT index_id, logical_name, embedding_dimension, identity,
                           audit_metadata, required_pgvector_version, status
                    FROM rag.vector_indexes
                    WHERE index_id = :index_id
                    FOR UPDATE
                    """
                    ),
                    {"index_id": manifest.index_id},
                )
                .mappings()
                .one_or_none()
            )

            if current_row is None:
                if manifest.status is not IndexStatus.BUILDING:
                    raise ValueError("a new index must start in building status")
                connection.execute(
                    text(
                        """
                        INSERT INTO rag.vector_indexes (
                            index_id, logical_name, embedding_dimension, identity,
                            audit_metadata, required_pgvector_version, status
                        ) VALUES (
                            :index_id, :logical_name, :embedding_dimension,
                            CAST(:identity AS jsonb), CAST(:audit_metadata AS jsonb),
                            :required_pgvector_version, :status
                        )
                        """
                    ),
                    _manifest_parameters(manifest),
                )
                return

            current = _manifest_from_row(current_row)
            if current.identity_payload != manifest.identity_payload:
                raise ValueError("index identity fields cannot change in place")
            if manifest.status not in _ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(
                    "invalid index status transition: "
                    f"{current.status.value} -> {manifest.status.value}"
                )

            if manifest.status is IndexStatus.ACTIVE:
                connection.execute(
                    text(
                        """
                        UPDATE rag.vector_indexes
                        SET status = 'retired', updated_at = CURRENT_TIMESTAMP
                        WHERE logical_name = :logical_name
                          AND status = 'active'
                          AND index_id <> :index_id
                        """
                    ),
                    {
                        "logical_name": manifest.logical_name,
                        "index_id": manifest.index_id,
                    },
                )
            connection.execute(
                text(
                    """
                    UPDATE rag.vector_indexes
                    SET audit_metadata = CAST(:audit_metadata AS jsonb),
                        required_pgvector_version = :required_pgvector_version,
                        status = :status,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE index_id = :index_id
                    """
                ),
                _manifest_parameters(manifest),
            )

    def get_active_generation(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentGeneration | None:
        """Read the generation selected by the stable document slot."""

        return self.get_document_state(index_id, document_id).active_generation

    def get_document_state(
        self,
        index_id: str,
        document_id: str,
    ) -> DocumentIndexState:
        """Read the active generation and optimistic revision in one snapshot."""

        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    SELECT doc.revision,
                           gen.generation_id, gen.index_id, gen.document_id,
                           gen.file_sha256, gen.routing_profile, gen.parser_profile,
                           gen.ocr_decision_fingerprint, gen.chunking_profile,
                           gen.gate_profile, gen.provenance_registry_fingerprint
                    FROM rag.indexed_documents AS doc
                    LEFT JOIN rag.document_generations AS gen
                      ON gen.generation_id = doc.active_generation_id
                     AND gen.index_id = doc.index_id
                     AND gen.document_id = doc.document_id
                    WHERE doc.index_id = :index_id AND doc.document_id = :document_id
                    """
                    ),
                    {"index_id": index_id, "document_id": document_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return DocumentIndexState(active_generation=None, revision=0)
        generation = None if row["generation_id"] is None else _generation_from_row(row)
        return DocumentIndexState(active_generation=generation, revision=int(row["revision"]))

    def replace_document(
        self,
        generation: DocumentGeneration,
        records: tuple[VectorRecord, ...],
        *,
        expected_revision: int,
    ) -> DocumentReplaceResult:
        """Lock one stable slot and atomically publish a complete generation."""

        if expected_revision < 0:
            raise ValueError("expected_revision must not be negative")

        with self._engine.begin() as connection:
            manifest_row = (
                connection.execute(
                    text(
                        """
                    SELECT index_id, logical_name, embedding_dimension, identity,
                           audit_metadata, required_pgvector_version, status
                    FROM rag.vector_indexes
                    WHERE index_id = :index_id
                    """
                    ),
                    {"index_id": generation.index_id},
                )
                .mappings()
                .one_or_none()
            )
            if manifest_row is None:
                raise ValueError("generation references an unknown index")
            manifest = _manifest_from_row(manifest_row)
            if manifest.status in {IndexStatus.FAILED, IndexStatus.RETIRED}:
                raise ValueError("cannot write to a failed or retired index")
            _validate_records(generation, records, manifest.embedding_dimension)

            connection.execute(
                text(
                    """
                    INSERT INTO rag.indexed_documents (index_id, document_id)
                    VALUES (:index_id, :document_id)
                    ON CONFLICT (index_id, document_id) DO NOTHING
                    """
                ),
                {"index_id": generation.index_id, "document_id": generation.document_id},
            )
            slot_row = (
                connection.execute(
                    text(
                        """
                    SELECT active_generation_id, revision
                    FROM rag.indexed_documents
                    WHERE index_id = :index_id AND document_id = :document_id
                    FOR UPDATE
                    """
                    ),
                    {"index_id": generation.index_id, "document_id": generation.document_id},
                )
                .mappings()
                .one()
            )
            active_generation_id = slot_row["active_generation_id"]
            current_revision = int(slot_row["revision"])

            if active_generation_id == generation.generation_id:
                record_count = connection.execute(
                    text(
                        """
                        SELECT count(*)
                        FROM rag.vector_records
                        WHERE generation_id = :generation_id
                        """
                    ),
                    {"generation_id": generation.generation_id},
                ).scalar_one()
                return DocumentReplaceResult(
                    status=DocumentReplaceStatus.ALREADY_INDEXED,
                    generation_id=generation.generation_id,
                    previous_generation_id=active_generation_id,
                    record_count=int(record_count),
                    document_revision=current_revision,
                )
            if current_revision != expected_revision:
                return DocumentReplaceResult(
                    status=DocumentReplaceStatus.CONCURRENT_UPDATE,
                    generation_id=generation.generation_id,
                    previous_generation_id=active_generation_id,
                    record_count=None,
                    document_revision=current_revision,
                )

            _insert_generation(connection, generation)
            for record in records:
                _insert_record(
                    connection,
                    record,
                    embedding_dimension=manifest.embedding_dimension,
                )

            if active_generation_id is not None:
                connection.execute(
                    text(
                        """
                        UPDATE rag.document_generations
                        SET status = 'superseded', superseded_at = CURRENT_TIMESTAMP
                        WHERE generation_id = :generation_id
                        """
                    ),
                    {"generation_id": active_generation_id},
                )
            update_result = connection.execute(
                text(
                    """
                    UPDATE rag.indexed_documents
                    SET active_generation_id = :generation_id,
                        revision = revision + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE index_id = :index_id AND document_id = :document_id
                      AND revision = :expected_revision
                    """
                ),
                {
                    "generation_id": generation.generation_id,
                    "index_id": generation.index_id,
                    "document_id": generation.document_id,
                    "expected_revision": expected_revision,
                },
            )
            if update_result.rowcount != 1:
                raise RuntimeError("document revision changed while its row was locked")
            next_revision = current_revision + 1
            return DocumentReplaceResult(
                status=(
                    DocumentReplaceStatus.INSERTED
                    if active_generation_id is None
                    else DocumentReplaceStatus.REPLACED
                ),
                generation_id=generation.generation_id,
                previous_generation_id=active_generation_id,
                record_count=len(records),
                document_revision=next_revision,
            )

    def search(self, query: VectorSearchQuery) -> list[SearchResult]:
        """Use exact cosine distance over active generations in one active index."""

        with self._engine.connect() as connection:
            manifest_row = (
                connection.execute(
                    text(
                        """
                    SELECT index_id, logical_name, embedding_dimension, identity,
                           audit_metadata, required_pgvector_version, status
                    FROM rag.vector_indexes
                    WHERE index_id = :index_id
                    """
                    ),
                    {"index_id": query.index_id},
                )
                .mappings()
                .one_or_none()
            )
            if manifest_row is None:
                raise ValueError("query references an unknown index")
            manifest = _manifest_from_row(manifest_row)
            if manifest.status is not IndexStatus.ACTIVE:
                raise ValueError("query index must be active")
            _validate_vector(query.vector, manifest.embedding_dimension, label="query vector")
            query_literal = _vector_literal(query.vector)

            rows = (
                connection.execute(
                    text(
                        """
                    SELECT vr.record_id, vr.index_id, vr.generation_id, vr.document_id,
                           vr.chunk_fingerprint, vr.chunk_index, vr.content,
                           vr.content_type, vr.section_path, vr.page_start, vr.page_end,
                           vr.embedding_input_hash, vr.embedding_input_token_count,
                           vr.embedding::text AS embedding_text,
                           vr.provenance, vr.provenance_schema_version,
                           1 - (vr.embedding <=> CAST(:query_vector AS vector)) AS score
                    FROM rag.vector_records AS vr
                    JOIN rag.indexed_documents AS doc
                      ON doc.index_id = vr.index_id
                     AND doc.document_id = vr.document_id
                     AND doc.active_generation_id = vr.generation_id
                    WHERE vr.index_id = :index_id
                      AND vr.embedding_dimension = :embedding_dimension
                    ORDER BY vr.embedding <=> CAST(:query_vector AS vector) ASC,
                             vr.record_id ASC
                    LIMIT :top_k
                    """
                    ),
                    {
                        "query_vector": query_literal,
                        "index_id": query.index_id,
                        "embedding_dimension": manifest.embedding_dimension,
                        "top_k": query.top_k,
                    },
                )
                .mappings()
                .all()
            )

        return [
            SearchResult(
                record=VectorRecord(
                    record_id=row["record_id"],
                    index_id=row["index_id"],
                    generation_id=row["generation_id"],
                    document_id=row["document_id"],
                    chunk_fingerprint=row["chunk_fingerprint"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    content_type=row["content_type"],
                    section_path=tuple(row["section_path"]),
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    embedding_input_hash=row["embedding_input_hash"],
                    embedding_input_token_count=row["embedding_input_token_count"],
                    vector=_parse_vector_literal(row["embedding_text"]),
                    provenance=dict(row["provenance"]),
                    provenance_schema_version=row["provenance_schema_version"],
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]


def _manifest_parameters(manifest: VectorIndexManifest) -> dict[str, object]:
    import json

    return {
        "index_id": manifest.index_id,
        "logical_name": manifest.logical_name,
        "embedding_dimension": manifest.embedding_dimension,
        "identity": json.dumps(manifest.identity_payload, separators=(",", ":"), sort_keys=True),
        "audit_metadata": json.dumps(
            manifest.audit_metadata,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "required_pgvector_version": manifest.required_pgvector_version,
        "status": manifest.status.value,
    }


def _manifest_from_row(row: Any) -> VectorIndexManifest:
    identity = dict(row["identity"])
    return VectorIndexManifest(
        index_id=row["index_id"],
        logical_name=row["logical_name"],
        embedding_dimension=row["embedding_dimension"],
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        tokenizer_id=identity["tokenizer_id"],
        tokenizer_revision=identity["tokenizer_revision"],
        normalized=identity["normalized"],
        pooling=identity["pooling"],
        dtype=identity["dtype"],
        query_profile=identity["query_profile"],
        document_input_profile=identity["document_input_profile"],
        max_input_tokens=identity["max_input_tokens"],
        oversize_policy=identity["oversize_policy"],
        adapter_version=identity["adapter_version"],
        required_pgvector_version=row["required_pgvector_version"],
        status=IndexStatus(row["status"]),
        audit_metadata=dict(row["audit_metadata"]),
    )


def _generation_from_row(row: Any) -> DocumentGeneration:
    return DocumentGeneration(
        generation_id=row["generation_id"],
        index_id=row["index_id"],
        document_id=row["document_id"],
        file_sha256=row["file_sha256"],
        routing_profile=row["routing_profile"],
        parser_profile=row["parser_profile"],
        ocr_decision_fingerprint=row["ocr_decision_fingerprint"],
        chunking_profile=row["chunking_profile"],
        gate_profile=row["gate_profile"],
        provenance_registry_fingerprint=row["provenance_registry_fingerprint"],
    )


def _insert_generation(connection: Connection, generation: DocumentGeneration) -> None:
    connection.execute(
        text(
            """
            INSERT INTO rag.document_generations (
                generation_id, index_id, document_id, file_sha256,
                routing_profile, parser_profile, ocr_decision_fingerprint,
                chunking_profile, gate_profile, provenance_registry_fingerprint,
                status
            ) VALUES (
                :generation_id, :index_id, :document_id, :file_sha256,
                :routing_profile, :parser_profile, :ocr_decision_fingerprint,
                :chunking_profile, :gate_profile, :provenance_registry_fingerprint,
                'active'
            )
            """
        ),
        {
            "generation_id": generation.generation_id,
            "index_id": generation.index_id,
            "document_id": generation.document_id,
            "file_sha256": generation.file_sha256,
            "routing_profile": generation.routing_profile,
            "parser_profile": generation.parser_profile,
            "ocr_decision_fingerprint": generation.ocr_decision_fingerprint,
            "chunking_profile": generation.chunking_profile,
            "gate_profile": generation.gate_profile,
            "provenance_registry_fingerprint": generation.provenance_registry_fingerprint,
        },
    )


def _insert_record(
    connection: Connection,
    record: VectorRecord,
    *,
    embedding_dimension: int,
) -> None:
    import json

    connection.execute(
        text(
            """
            INSERT INTO rag.vector_records (
                record_id, index_id, embedding_dimension, generation_id,
                document_id, chunk_fingerprint, chunk_index, content,
                content_type, section_path, page_start, page_end,
                embedding_input_hash, embedding_input_token_count, embedding,
                provenance_schema_version, provenance
            ) VALUES (
                :record_id, :index_id, :embedding_dimension, :generation_id,
                :document_id, :chunk_fingerprint, :chunk_index, :content,
                :content_type, CAST(:section_path AS jsonb), :page_start, :page_end,
                :embedding_input_hash, :embedding_input_token_count,
                CAST(:embedding AS vector), :provenance_schema_version,
                CAST(:provenance AS jsonb)
            )
            """
        ),
        {
            "record_id": record.record_id,
            "index_id": record.index_id,
            "embedding_dimension": embedding_dimension,
            "generation_id": record.generation_id,
            "document_id": record.document_id,
            "chunk_fingerprint": record.chunk_fingerprint,
            "chunk_index": record.chunk_index,
            "content": record.content,
            "content_type": record.content_type,
            "section_path": json.dumps(record.section_path, ensure_ascii=False),
            "page_start": record.page_start,
            "page_end": record.page_end,
            "embedding_input_hash": record.embedding_input_hash,
            "embedding_input_token_count": record.embedding_input_token_count,
            "embedding": _vector_literal(record.vector),
            "provenance_schema_version": record.provenance_schema_version,
            "provenance": json.dumps(record.provenance, ensure_ascii=False),
        },
    )


def _validate_records(
    generation: DocumentGeneration,
    records: tuple[VectorRecord, ...],
    dimension: int,
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
        _validate_vector(record.vector, dimension, label="record vector")


def _validate_vector(vector: tuple[float, ...], dimension: int, *, label: str) -> None:
    if len(vector) != dimension:
        raise ValueError(f"{label} must have exactly {dimension} dimensions")
    if not all(isfinite(value) for value in vector):
        raise ValueError(f"{label} must contain only finite values")
    if not any(value != 0.0 for value in vector):
        raise ValueError(f"{label} must not be a zero vector")


def _vector_literal(vector: tuple[float, ...]) -> str:
    """Serialize finite values for an explicit PostgreSQL vector cast."""

    if not vector or not all(isfinite(value) for value in vector):
        raise ValueError("vector literal requires finite values")
    return f"[{','.join(format(value, '.17g') for value in vector)}]"


def _parse_vector_literal(value: str) -> tuple[float, ...]:
    """Parse the stable text representation returned by pgvector."""

    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError("invalid pgvector text representation")
    body = stripped[1:-1]
    if not body:
        raise ValueError("pgvector text representation must not be empty")
    vector = tuple(float(part) for part in body.split(","))
    if not all(isfinite(number) for number in vector):
        raise ValueError("pgvector text representation contains non-finite values")
    return vector


def _version_at_least(actual: str, required: str) -> bool:
    """Compare numeric extension versions while tolerating development suffixes."""

    actual_parts = tuple(int(part) for part in findall(r"\d+", actual))
    required_parts = tuple(int(part) for part in findall(r"\d+", required))
    if not actual_parts or not required_parts:
        raise ValueError("pgvector versions must contain numeric components")
    width = max(len(actual_parts), len(required_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) >= required_parts + (0,) * (
        width - len(required_parts)
    )

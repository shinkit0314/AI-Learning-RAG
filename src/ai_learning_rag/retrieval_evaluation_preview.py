"""Bridge real PDF retrieval previews into the deterministic evaluator."""

from __future__ import annotations

from collections.abc import Mapping

from ai_learning_rag.document_index import (
    IndexStatus,
    SearchResult,
    VectorIndexManifest,
    VectorRecord,
)
from ai_learning_rag.retrieval_evaluation import (
    ChunkIdentity,
    CorpusCompatibilityError,
    CorpusInventory,
    CorpusSample,
    CorpusSnapshot,
)


def evaluation_inputs_from_preview(
    payload: Mapping[str, object],
) -> tuple[CorpusInventory, VectorIndexManifest, dict[str, list[SearchResult]]]:
    """Rebuild exact corpus, index identity, and rankings from one real preview."""

    if payload.get("schema_version") != "pdf-indexing-retrieval-preview-v1":
        raise ValueError("unsupported retrieval preview payload")
    inventory_payload = _mapping(payload.get("corpus_inventory"), "corpus_inventory")
    snapshot_payload = _mapping(inventory_payload.get("snapshot"), "snapshot")
    sample_values = _list(snapshot_payload.get("samples"), "snapshot.samples")
    chunk_values = _list(inventory_payload.get("chunks"), "corpus_inventory.chunks")
    samples = tuple(_sample(value) for value in sample_values)
    inventory = CorpusInventory.create(
        corpus_id=str(snapshot_payload.get("corpus_id", "")),
        samples=samples,
        chunking_profile=str(snapshot_payload.get("chunking_profile", "")),
        gate_profile=str(snapshot_payload.get("gate_profile", "")),
        chunks=tuple(_chunk(value) for value in chunk_values),
    )
    declared_snapshot = CorpusSnapshot(
        corpus_id=str(snapshot_payload.get("corpus_id", "")),
        samples=samples,
        chunking_profile=str(snapshot_payload.get("chunking_profile", "")),
        gate_profile=str(snapshot_payload.get("gate_profile", "")),
        chunk_count=_integer(snapshot_payload.get("chunk_count"), "chunk_count"),
        chunk_set_fingerprint=str(snapshot_payload.get("chunk_set_fingerprint", "")),
    )
    if inventory.snapshot != declared_snapshot:
        raise CorpusCompatibilityError("preview corpus inventory does not match its snapshot")

    manifest = _manifest(_mapping(payload.get("index_manifest"), "index_manifest"))
    queries = _list(payload.get("queries"), "queries")
    rankings: dict[str, list[SearchResult]] = {}
    for value in queries:
        query = _mapping(value, "query")
        query_id = str(query.get("query_id", ""))
        if not query_id.strip() or query_id in rankings:
            raise ValueError("preview query ids must be non-blank and unique")
        hits = _list(query.get("hits"), "query.hits")
        ranking = [_search_result(hit, expected_rank=index) for index, hit in enumerate(hits, 1)]
        rankings[query_id] = ranking
    return inventory, manifest, rankings


def _manifest(payload: Mapping[str, object]) -> VectorIndexManifest:
    identity = _mapping(payload.get("identity"), "index_manifest.identity")
    audit = _mapping(payload.get("audit_metadata"), "index_manifest.audit_metadata")
    manifest = VectorIndexManifest.create(
        logical_name=str(identity.get("logical_name", "")),
        embedding_dimension=_integer(identity.get("embedding_dimension"), "embedding_dimension"),
        model_id=str(identity.get("model_id", "")),
        model_revision=str(identity.get("model_revision", "")),
        tokenizer_id=str(identity.get("tokenizer_id", "")),
        tokenizer_revision=str(identity.get("tokenizer_revision", "")),
        normalized=_boolean(identity.get("normalized"), "normalized"),
        pooling=str(identity.get("pooling", "")),
        dtype=str(identity.get("dtype", "")),
        query_profile=str(identity.get("query_profile", "")),
        document_input_profile=str(identity.get("document_input_profile", "")),
        max_input_tokens=_integer(identity.get("max_input_tokens"), "max_input_tokens"),
        oversize_policy=str(identity.get("oversize_policy", "")),
        adapter_version=str(identity.get("adapter_version", "")),
        required_pgvector_version=str(payload.get("required_pgvector_version", "0.8.7")),
        status=IndexStatus(str(payload.get("status", ""))),
        audit_metadata={str(key): str(value) for key, value in audit.items()},
    )
    if manifest.index_id != payload.get("index_id"):
        raise ValueError("preview index_id does not match its manifest identity")
    return manifest


def _search_result(value: object, *, expected_rank: int) -> SearchResult:
    payload = _mapping(value, "hit")
    if _integer(payload.get("rank"), "rank") != expected_rank:
        raise ValueError("preview hit ranks must be contiguous and start at one")
    return SearchResult(
        record=VectorRecord(
            record_id=str(payload.get("record_id", "")),
            index_id=str(payload.get("index_id", "")),
            generation_id=str(payload.get("generation_id", "")),
            document_id=str(payload.get("document_id", "")),
            chunk_fingerprint=str(payload.get("chunk_fingerprint", "")),
            chunk_index=_integer(payload.get("chunk_index"), "chunk_index"),
            content=str(payload.get("content", "")),
            content_type=str(payload.get("content_type", "")),
            section_path=tuple(
                str(item) for item in _list(payload.get("section_path"), "section_path")
            ),
            page_start=_integer(payload.get("page_start"), "page_start"),
            page_end=_integer(payload.get("page_end"), "page_end"),
            embedding_input_hash=str(payload.get("embedding_input_hash", "")),
            embedding_input_token_count=_integer(
                payload.get("embedding_input_token_count"), "embedding_input_token_count"
            ),
            vector=(),
            provenance=dict(_mapping(payload.get("provenance"), "provenance")),
        ),
        score=_number(payload.get("score"), "score"),
    )


def _sample(value: object) -> CorpusSample:
    payload = _mapping(value, "sample")
    return CorpusSample(
        sample_id=str(payload.get("sample_id", "")),
        path=str(payload.get("path", "")),
        document_id=str(payload.get("document_id", "")),
        file_sha256=str(payload.get("file_sha256", "")),
    )


def _chunk(value: object) -> ChunkIdentity:
    payload = _mapping(value, "chunk")
    return ChunkIdentity(
        document_id=str(payload.get("document_id", "")),
        chunk_fingerprint=str(payload.get("chunk_fingerprint", "")),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    return float(value)

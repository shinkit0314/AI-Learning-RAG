"""Tests for fail-closed PDF chunk indexing orchestration."""

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from hashlib import sha256

import pytest

from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    InMemoryVectorIndex,
    VectorIndexManifest,
    VectorSearchQuery,
)
from ai_learning_rag.pdf_indexing import ChunkRejectionCode, PdfChunkIndexer
from ai_learning_rag.pdf_schema import (
    BlockType,
    BoundingBox,
    ChunkBlockSlice,
    ChunkCandidate,
    ChunkContentType,
    ChunkWarning,
    ExtractionMethod,
    ExtractionProvenance,
    ParsedBlock,
    TraceablePdfChunk,
    WarningSeverity,
    build_traceable_pdf_chunk,
)


class FakeEmbedder:
    """Record exact inputs and return deterministic two-dimensional vectors."""

    dimension = 2

    def __init__(self, *, before_embed: Callable[[], None] | None = None) -> None:
        self.inputs: list[str] = []
        self.before_embed = before_embed

    def embed_query(self, query: str) -> list[float]:
        raise AssertionError("query embedding is not used during indexing")

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        callback = self.before_embed
        self.before_embed = None
        if callback is not None:
            callback()
        self.inputs.extend(documents)
        return [[1.0, 0.0] for _document in documents]


class WordTokenCounter:
    """Small deterministic stand-in for the real model tokenizer."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def make_provenance(
    *,
    method: ExtractionMethod = ExtractionMethod.NATIVE,
) -> ExtractionProvenance:
    """Create one registered extraction configuration."""

    return ExtractionProvenance(
        id=f"provenance-{method.value}",
        parser_name="fixture-parser",
        parser_version="1",
        extraction_method=method,
        configuration_profile="fixture-v1",
        ocr_engine_name="fixture-ocr" if method in _OCR_METHODS else None,
        ocr_engine_version="1" if method in _OCR_METHODS else None,
    )


_OCR_METHODS = {ExtractionMethod.EMBEDDED_OCR, ExtractionMethod.RERUN_OCR}


def make_manifest() -> VectorIndexManifest:
    """Build the embedding-space manifest exercised by PDF indexing tests."""

    return VectorIndexManifest.create(
        logical_name="pdf-knowledge-base",
        embedding_dimension=2,
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="revision-1",
        tokenizer_id="Qwen/Qwen3-Embedding-0.6B",
        tokenizer_revision="revision-1",
        normalized=True,
        pooling="model-default",
        dtype="float16",
        query_profile="qwen-query-v1",
        document_input_profile="section-path-prefix-v1",
        max_input_tokens=512,
        oversize_policy="reject-v1",
        adapter_version="qwen3-adapter-v1",
        required_pgvector_version="0.8.7",
    )


def make_generation(
    manifest: VectorIndexManifest,
    *,
    file_marker: str = "file-a",
) -> DocumentGeneration:
    """Build the complete document processing identity used in this module."""

    return DocumentGeneration.create(
        index_id=manifest.index_id,
        document_id="document-a",
        file_sha256=sha256(file_marker.encode()).hexdigest(),
        routing_profile="pdf-page-routing-v1",
        parser_profile="pymupdf-1",
        ocr_decision_fingerprint=sha256(b"ocr-decisions").hexdigest(),
        chunking_profile="pdf-structure-v1",
        gate_profile="pdf-index-gate-v1",
        provenance_registry_fingerprint=sha256(b"provenance-registry").hexdigest(),
    )


def make_chunk(
    *,
    content: str = "vector databases preserve provenance",
    content_type: ChunkContentType = ChunkContentType.BODY_TEXT,
    method: ExtractionMethod = ExtractionMethod.NATIVE,
    warnings: tuple[ChunkWarning, ...] = (),
    provenance_id: str | None = "provenance-native",
) -> TraceablePdfChunk:
    """Build a real TraceablePdfChunk instead of bypassing its source invariants."""

    block = ParsedBlock(
        id=f"block-{sha256(content.encode()).hexdigest()[:8]}",
        page_number=1,
        block_type=(
            BlockType.OTHER if content_type is ChunkContentType.OTHER else BlockType.PARAGRAPH
        ),
        text=content,
        bbox=BoundingBox(0, 0, 100, 20),
        reading_order=0,
        extraction_method=method,
        extraction_provenance_id=provenance_id,
    )
    candidate = ChunkCandidate(
        document_id="document-a",
        block_slices=(ChunkBlockSlice.whole(block),),
        section_path=("RAG", "Indexing"),
        routing_profile="pdf-page-routing-v1",
    )
    return build_traceable_pdf_chunk(
        candidate,
        chunk_index=0,
        chunking_profile="pdf-structure-v1",
        content_type=content_type,
        warnings=warnings,
    )


def make_indexer(
    *,
    max_input_tokens: int = 512,
) -> tuple[InMemoryVectorIndex, FakeEmbedder, PdfChunkIndexer, DocumentGeneration]:
    """Create an active in-memory index and its PDF orchestration boundary."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    index.ensure_index(manifest)
    index.ensure_index(replace(manifest, status=IndexStatus.READY))
    index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))
    embedder = FakeEmbedder()
    indexer = PdfChunkIndexer(
        index=index,
        embedder=embedder,
        token_counter=WordTokenCounter(),
        max_input_tokens=max_input_tokens,
    )
    return index, embedder, indexer, make_generation(manifest)


def test_indexer_prefixes_section_path_and_preserves_original_evidence() -> None:
    """Embedding context must not overwrite the content used for citations."""

    index, embedder, indexer, generation = make_indexer()
    provenance = make_provenance()
    chunk = make_chunk()

    result = indexer.index_document(
        generation=generation,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )
    retrieved = index.search(VectorSearchQuery(generation.index_id, (1.0, 0.0)))[0]

    assert result.replacement.status is DocumentReplaceStatus.INSERTED
    assert embedder.inputs == ["RAG > Indexing\n\nvector databases preserve provenance"]
    assert retrieved.record.content == "vector databases preserve provenance"
    assert retrieved.record.embedding_input_token_count == 7
    assert retrieved.record.provenance["schema_version"] == 1


@pytest.mark.parametrize(
    ("chunk", "registry", "reason"),
    [
        (
            make_chunk(content_type=ChunkContentType.OTHER),
            {"provenance-native": make_provenance()},
            ChunkRejectionCode.UNSUPPORTED_CONTENT_TYPE,
        ),
        (
            make_chunk(
                warnings=(
                    ChunkWarning(
                        code="FUTURE_WARNING",
                        severity=WarningSeverity.WARNING,
                        message="unknown to this gate",
                    ),
                )
            ),
            {"provenance-native": make_provenance()},
            ChunkRejectionCode.UNKNOWN_WARNING,
        ),
        (
            make_chunk(provenance_id=None),
            {},
            ChunkRejectionCode.MISSING_PROVENANCE,
        ),
        (
            make_chunk(
                method=ExtractionMethod.RERUN_OCR,
                provenance_id="provenance-rerun_ocr",
            ),
            {"provenance-rerun_ocr": make_provenance(method=ExtractionMethod.RERUN_OCR)},
            ChunkRejectionCode.UNRESOLVED_OCR,
        ),
    ],
)
def test_gate_rejects_unsafe_chunks_without_embedding(
    chunk: TraceablePdfChunk,
    registry: dict[str, ExtractionProvenance],
    reason: ChunkRejectionCode,
) -> None:
    """Unsupported, unknown, untraceable, and unresolved input must fail closed."""

    _index, embedder, indexer, generation = make_indexer()

    result = indexer.index_document(
        generation=generation,
        chunks=[chunk],
        provenance_registry=registry,
    )

    assert result.replacement is None
    assert result.report is not None
    assert result.report.decisions[0].rejection_code is reason
    assert embedder.inputs == []


def test_token_limit_is_checked_before_embedding_and_reported() -> None:
    """Model-library truncation must never silently change indexed evidence."""

    _index, embedder, indexer, generation = make_indexer(max_input_tokens=3)
    provenance = make_provenance()

    result = indexer.index_document(
        generation=generation,
        chunks=[make_chunk()],
        provenance_registry={provenance.id: provenance},
    )

    assert result.replacement is None
    assert result.report is not None
    assert result.report.decisions[0].rejection_code is ChunkRejectionCode.TOKEN_LIMIT_EXCEEDED
    assert embedder.inputs == []


def test_identical_generation_skips_gate_tokenization_and_embedding() -> None:
    """An already active generation should be a cheap idempotent no-op."""

    _index, embedder, indexer, generation = make_indexer()
    provenance = make_provenance()
    chunk = make_chunk()
    first = indexer.index_document(
        generation=generation,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )
    embedder.inputs.clear()

    repeated = indexer.index_document(
        generation=generation,
        chunks=[],
        provenance_registry={},
    )

    assert first.report is not None
    assert repeated.report is None
    assert repeated.replacement is not None
    assert repeated.replacement.status is DocumentReplaceStatus.ALREADY_INDEXED
    assert embedder.inputs == []


def test_report_renders_the_same_decisions_as_json_and_markdown() -> None:
    """Gate decisions should be reviewable without inspecting Python objects."""

    _index, _embedder, indexer, generation = make_indexer()
    provenance = make_provenance()
    chunk = make_chunk()
    result = indexer.index_document(
        generation=generation,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )

    assert result.report is not None
    decision = result.report.decisions[0]
    payload = json.loads(result.report.to_json())
    markdown = result.report.to_markdown()
    source_block_id = chunk.source_regions[0].block_id

    assert decision.page_numbers == (1,)
    assert decision.source_block_ids == (source_block_id,)
    assert payload["decisions"][0]["page_numbers"] == [1]
    assert payload["decisions"][0]["source_block_ids"] == [source_block_id]
    assert "| `pdf_chunk_" in markdown
    assert f"| body_text | 1 | `{source_block_id}` |" in markdown


def test_gate_report_can_be_rebuilt_without_embedding_or_index_writes() -> None:
    """An idempotent preview can refresh audit artifacts without reindexing."""

    index, embedder, indexer, generation = make_indexer()
    provenance = make_provenance()
    chunk = make_chunk()
    first = indexer.index_document(
        generation=generation,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )
    embedder.inputs.clear()

    report = indexer.build_gate_report(
        generation=generation,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )

    assert first.replacement is not None
    assert report.accepted_count == 1
    assert embedder.inputs == []
    assert index.get_active_generation(generation.index_id, generation.document_id) == generation


def test_indexer_reports_conflict_when_generation_changes_during_embedding() -> None:
    """Expensive work must not let a stale candidate overwrite a concurrent publish."""

    index = InMemoryVectorIndex()
    manifest = make_manifest()
    index.ensure_index(manifest)
    index.ensure_index(replace(manifest, status=IndexStatus.READY))
    index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))
    candidate = make_generation(manifest, file_marker="candidate")
    concurrent = make_generation(manifest, file_marker="concurrent")
    provenance = make_provenance()
    chunk = make_chunk()
    concurrent_indexer = PdfChunkIndexer(
        index=index,
        embedder=FakeEmbedder(),
        token_counter=WordTokenCounter(),
        max_input_tokens=512,
    )

    def publish_concurrent_generation() -> None:
        result = concurrent_indexer.index_document(
            generation=concurrent,
            chunks=[chunk],
            provenance_registry={provenance.id: provenance},
        )
        assert result.replacement is not None
        assert result.replacement.status is DocumentReplaceStatus.INSERTED

    candidate_indexer = PdfChunkIndexer(
        index=index,
        embedder=FakeEmbedder(before_embed=publish_concurrent_generation),
        token_counter=WordTokenCounter(),
        max_input_tokens=512,
    )

    result = candidate_indexer.index_document(
        generation=candidate,
        chunks=[chunk],
        provenance_registry={provenance.id: provenance},
    )

    assert result.replacement is not None
    assert result.replacement.status is DocumentReplaceStatus.CONCURRENT_UPDATE
    assert result.replacement.previous_generation_id == concurrent.generation_id
    assert result.replacement.document_revision == 1
    state = index.get_document_state(manifest.index_id, candidate.document_id)
    assert state.active_generation == concurrent
    assert state.revision == 1

"""Index the three reviewed PDFs and generate fixed retrieval previews."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.document_index import (
    DocumentGeneration,
    DocumentReplaceStatus,
    IndexStatus,
    VectorIndexManifest,
    VectorSearchQuery,
)
from ai_learning_rag.embedding import EmbeddingConfig, Qwen3Embedder
from ai_learning_rag.pdf_auto_routing import parse_pdf_auto
from ai_learning_rag.pdf_chunking import PdfChunkingConfig, chunk_pdf_document
from ai_learning_rag.pdf_indexing import PdfChunkIndexer, PdfIndexingPolicy
from ai_learning_rag.pdf_ocr_review import load_ocr_review_decisions
from ai_learning_rag.pdf_retrieval_preview import (
    RetrievalPreviewCase,
    build_query_preview,
    build_retrieval_markdown,
)
from ai_learning_rag.pdf_routing import PdfRoutingConfig
from ai_learning_rag.pdf_routing_cache import PdfRoutingCache, RoutingCacheMode
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/pdf-indexing-preview-v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-indexing-preview"),
    )
    args = parser.parse_args()
    preview_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if preview_manifest.get("schema_version") != "pdf-indexing-preview-v1":
        raise ValueError("unsupported PDF indexing preview manifest")

    embedding_config = EmbeddingConfig(local_files_only=True)
    _require_local_model_snapshot(embedding_config)
    embedder = Qwen3Embedder(embedding_config)
    database_settings = DatabaseSettings.from_env()
    engine = database_settings.create_engine()
    index = PostgresVectorIndex(engine)
    policy = PdfIndexingPolicy()
    manifest = VectorIndexManifest.create(
        logical_name=preview_manifest["logical_index_name"],
        embedding_dimension=embedding_config.dimension,
        model_id=embedding_config.model_id,
        model_revision=embedding_config.model_revision,
        tokenizer_id=embedding_config.tokenizer_id,
        tokenizer_revision=embedding_config.tokenizer_revision,
        normalized=embedding_config.normalize_embeddings,
        pooling=embedding_config.pooling,
        dtype=embedding_config.dtype,
        query_profile=embedding_config.query_profile,
        document_input_profile=policy.document_input_profile,
        max_input_tokens=embedding_config.max_sequence_length,
        oversize_policy="reject-v1",
        adapter_version=embedding_config.adapter_version,
        required_pgvector_version="0.8.7",
        audit_metadata={
            "device": embedding_config.device,
            "sentence_transformers": version("sentence-transformers"),
            "torch": version("torch"),
        },
    )
    _ensure_buildable_index(index, manifest)

    routing_config = PdfRoutingConfig.from_json(Path(preview_manifest["routing_config"]))
    routing_cache = PdfRoutingCache(
        Path(preview_manifest["routing_cache_dir"]),
        RoutingCacheMode.READ_ONLY,
    )
    chunking_config = PdfChunkingConfig()
    indexer = PdfChunkIndexer(
        index=index,
        embedder=embedder,
        token_counter=embedder,
        max_input_tokens=embedding_config.max_sequence_length,
        policy=policy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document_samples: dict[str, str] = {}
    sample_summaries = []
    try:
        for sample in preview_manifest["samples"]:
            summary, document_id = _index_sample(
                sample=sample,
                routing_config=routing_config,
                routing_cache=routing_cache,
                chunking_config=chunking_config,
                indexer=indexer,
                manifest=manifest,
                policy=policy,
                output_dir=args.output_dir,
            )
            document_samples[document_id] = sample["sample_id"]
            sample_summaries.append(summary)

        current = index.get_manifest(manifest.index_id)
        assert current is not None
        if current.status is IndexStatus.BUILDING:
            index.ensure_index(replace(manifest, status=IndexStatus.READY))
            current = replace(manifest, status=IndexStatus.READY)
        if current.status is IndexStatus.READY:
            index.ensure_index(replace(manifest, status=IndexStatus.ACTIVE))
        elif current.status is not IndexStatus.ACTIVE:
            raise RuntimeError(f"index cannot be activated from {current.status.value}")

        query_payloads = []
        for query_item in preview_manifest["queries"]:
            case = RetrievalPreviewCase(
                query_id=query_item["query_id"],
                query=query_item["query"],
                expected_sample_id=query_item["expected_sample_id"],
                expected_pages=tuple(query_item["expected_pages"]),
                top_k=query_item["top_k"],
            )
            if embedder.count_tokens(case.query) > embedding_config.max_sequence_length:
                raise ValueError(f"query exceeds token limit: {case.query_id}")
            results = index.search(
                VectorSearchQuery(
                    index_id=manifest.index_id,
                    vector=tuple(embedder.embed_query(case.query)),
                    top_k=case.top_k,
                )
            )
            query_payloads.append(build_query_preview(case, results, document_samples))

        payload = {
            "schema_version": "pdf-indexing-retrieval-preview-v1",
            "index_id": manifest.index_id,
            "logical_index_name": manifest.logical_name,
            "model_id": manifest.model_id,
            "model_revision": manifest.model_revision,
            "embedding_dimension": manifest.embedding_dimension,
            "document_input_profile": manifest.document_input_profile,
            "samples": sample_summaries,
            "queries": query_payloads,
        }
        (args.output_dir / "retrieval-preview.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "retrieval-preview.md").write_text(
            build_retrieval_markdown(payload),
            encoding="utf-8",
        )
        print(
            f"index={manifest.index_id} samples={len(sample_summaries)} "
            f"queries={len(query_payloads)} -> {args.output_dir}"
        )
    finally:
        engine.dispose()


def _ensure_buildable_index(
    index: PostgresVectorIndex,
    manifest: VectorIndexManifest,
) -> None:
    current = index.get_manifest(manifest.index_id)
    if current is None:
        index.ensure_index(manifest)
        return
    if current.identity_payload != manifest.identity_payload:
        raise RuntimeError("stored vector index identity does not match the preview manifest")
    if current.status in {IndexStatus.FAILED, IndexStatus.RETIRED}:
        raise RuntimeError(f"cannot reuse index in {current.status.value} status")


def _index_sample(
    *,
    sample: dict[str, object],
    routing_config: PdfRoutingConfig,
    routing_cache: PdfRoutingCache,
    chunking_config: PdfChunkingConfig,
    indexer: PdfChunkIndexer,
    manifest: VectorIndexManifest,
    policy: PdfIndexingPolicy,
    output_dir: Path,
) -> tuple[dict[str, object], str]:
    pdf_path = Path(str(sample["path"]))
    decisions_path_value = sample.get("ocr_review_decisions")
    decisions_path = None if decisions_path_value is None else Path(str(decisions_path_value))
    decisions = None if decisions_path is None else load_ocr_review_decisions(decisions_path)
    routing_result = parse_pdf_auto(
        pdf_path,
        routing_config,
        cache=routing_cache,
        ocr_review_decisions=decisions,
    )
    document = routing_result.document
    chunks = chunk_pdf_document(document, chunking_config)
    registry_fingerprint = _provenance_registry_fingerprint(document.extraction_provenances)
    generation = DocumentGeneration.create(
        index_id=manifest.index_id,
        document_id=document.document_id,
        file_sha256=document.file_sha256,
        routing_profile=document.routing_profile,
        parser_profile=f"{document.parser_name}:{document.parser_version}",
        ocr_decision_fingerprint=(
            sha256(b"no-manual-ocr-review-v1").hexdigest()
            if decisions_path is None
            else sha256(decisions_path.read_bytes()).hexdigest()
        ),
        chunking_profile=chunking_config.chunking_profile,
        gate_profile=policy.profile,
        provenance_registry_fingerprint=registry_fingerprint,
    )
    registry = {provenance.id: provenance for provenance in document.extraction_provenances}
    result = indexer.index_document(
        generation=generation,
        chunks=chunks.chunks,
        provenance_registry=registry,
    )
    if result.replacement is None:
        raise RuntimeError(f"all chunks were rejected for {sample['sample_id']}")
    if result.replacement.status is DocumentReplaceStatus.CONCURRENT_UPDATE:
        raise RuntimeError(
            f"document changed while indexing {sample['sample_id']}; "
            "rerun against the latest source"
        )
    report = result.report or indexer.build_gate_report(
        generation=generation,
        chunks=chunks.chunks,
        provenance_registry=registry,
    )
    stem = str(sample["sample_id"])
    (output_dir / f"{stem}.gate.json").write_text(
        report.to_json(),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.gate.md").write_text(
        report.to_markdown(),
        encoding="utf-8",
    )
    return (
        {
            "sample_id": sample["sample_id"],
            "path": str(pdf_path),
            "document_id": document.document_id,
            "generation_id": generation.generation_id,
            "chunking_status": chunks.status.value,
            "produced_chunks": len(chunks.chunks),
            "replacement_status": result.replacement.status.value,
            "document_revision": result.replacement.document_revision,
            "indexed_records": report.accepted_count,
            "records_written_this_run": (
                0
                if result.replacement.status is DocumentReplaceStatus.ALREADY_INDEXED
                else result.replacement.record_count
            ),
            "gate_accepted": report.accepted_count,
            "gate_rejected": report.rejected_count,
        },
        document.document_id,
    )


def _provenance_registry_fingerprint(provenances: tuple[object, ...]) -> str:
    payload = json.dumps(
        [asdict(provenance) for provenance in provenances],  # type: ignore[arg-type]
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()


def _require_local_model_snapshot(config: EmbeddingConfig) -> None:
    cache_name = f"models--{config.model_id.replace('/', '--')}"
    snapshot = config.cache_dir / cache_name / "snapshots" / config.model_revision
    if not snapshot.is_dir():
        raise RuntimeError(
            "the pinned embedding model is not cached; this preview never downloads models: "
            f"{snapshot}"
        )


if __name__ == "__main__":
    main()

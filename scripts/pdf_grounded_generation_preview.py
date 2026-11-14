"""Generate grounded answers and page-level citations from the reviewed PDF index."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from ai_learning_rag.database import DatabaseSettings
from ai_learning_rag.deepseek_generation import (
    DeepSeekGenerationConfig,
    DeepSeekTextGenerator,
    load_deepseek_dotenv,
)
from ai_learning_rag.document_index import IndexStatus, VectorIndexManifest
from ai_learning_rag.embedding import EmbeddingConfig, Qwen3Embedder
from ai_learning_rag.pdf_extraction import pdf_file_sha256
from ai_learning_rag.pdf_generation import PdfAnswerStatus, PdfGroundedAnswerService
from ai_learning_rag.pdf_generation_preview import (
    PdfGenerationPreviewCase,
    build_pdf_generation_case_preview,
    build_pdf_generation_markdown,
)
from ai_learning_rag.pdf_indexing import PdfIndexingPolicy
from ai_learning_rag.pdf_retrieval import PdfRetriever
from ai_learning_rag.postgres_vector_index import PostgresVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/pdf-grounded-generation-preview-v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf-grounded-generation-preview"),
    )
    args = parser.parse_args()
    preview_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if preview_manifest.get("schema_version") != "pdf-grounded-generation-preview-v1":
        raise ValueError("unsupported PDF grounded-generation preview manifest")

    project_root = Path(__file__).resolve().parents[1]
    load_deepseek_dotenv(project_root / ".env")
    generation_settings = dict(preview_manifest["generation"])
    provider = generation_settings.pop("provider", None)
    if provider != "deepseek":
        raise ValueError("the PDF grounded-generation preview requires provider=deepseek")
    generation_config = DeepSeekGenerationConfig.from_env(**generation_settings)
    generator = DeepSeekTextGenerator(generation_config)

    embedding_config = EmbeddingConfig(local_files_only=True)
    _require_local_model_snapshot(embedding_config)
    embedder = Qwen3Embedder(embedding_config)
    index_manifest = _build_index_manifest(preview_manifest, embedding_config)

    engine = DatabaseSettings.from_env().create_engine()
    index = PostgresVectorIndex(engine)
    try:
        stored_manifest = index.get_manifest(index_manifest.index_id)
        if stored_manifest is None:
            raise RuntimeError("the reviewed PDF vector index has not been built")
        if stored_manifest.status is not IndexStatus.ACTIVE:
            raise RuntimeError("the reviewed PDF vector index is not active")
        if stored_manifest.identity_payload != index_manifest.identity_payload:
            raise RuntimeError("stored PDF vector index identity does not match the preview")

        retriever = PdfRetriever(index=index, manifest=index_manifest, embedder=embedder)
        service = PdfGroundedAnswerService(
            retriever=retriever,
            generator=generator,
            index_id=index_manifest.index_id,
            max_context_characters=int(preview_manifest["max_context_characters"]),
        )
        document_samples, sample_sources = _document_references(preview_manifest["samples"])
        query_payloads = []
        query_items = preview_manifest["queries"]
        for position, item in enumerate(query_items, start=1):
            case = PdfGenerationPreviewCase(
                query_id=item["query_id"],
                query=item["query"],
                top_k=item["top_k"],
                expected_status=PdfAnswerStatus(item["expected_status"]),
                expected_sample_id=item["expected_sample_id"],
                expected_pages=tuple(item["expected_pages"]),
                expected_min_citations=item["expected_min_citations"],
            )
            print(
                f"[{position}/{len(query_items)}] Generating query={case.query_id}",
                flush=True,
            )
            response = service.answer(case.query, top_k=case.top_k)
            print(
                f"      status={response.status.value} citations={len(response.citations)}",
                flush=True,
            )
            query_payloads.append(
                build_pdf_generation_case_preview(
                    case,
                    response,
                    document_samples=document_samples,
                    sample_sources=sample_sources,
                )
            )

        payload = {
            "schema_version": "pdf-grounded-generation-preview-v1",
            "index_id": index_manifest.index_id,
            "embedding_model_id": embedding_config.model_id,
            "embedding_model_revision": embedding_config.model_revision,
            "generation_provider": provider,
            "generation_model_id": generation_config.model,
            "max_context_characters": preview_manifest["max_context_characters"],
            "queries": query_payloads,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "grounded-generation-preview.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "grounded-generation-preview.md").write_text(
            build_pdf_generation_markdown(payload),
            encoding="utf-8",
        )
        print(f"index={index_manifest.index_id} queries={len(query_payloads)} -> {args.output_dir}")
    finally:
        engine.dispose()


def _build_index_manifest(
    preview_manifest: dict[str, object],
    config: EmbeddingConfig,
) -> VectorIndexManifest:
    policy = PdfIndexingPolicy()
    return VectorIndexManifest.create(
        logical_name=str(preview_manifest["logical_index_name"]),
        embedding_dimension=config.dimension,
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=config.tokenizer_revision,
        normalized=config.normalize_embeddings,
        pooling=config.pooling,
        dtype=config.dtype,
        query_profile=config.query_profile,
        document_input_profile=policy.document_input_profile,
        max_input_tokens=config.max_sequence_length,
        oversize_policy="reject-v1",
        adapter_version=config.adapter_version,
        required_pgvector_version="0.8.7",
        audit_metadata={
            "device": config.device,
            "sentence_transformers": version("sentence-transformers"),
            "torch": version("torch"),
        },
    )


def _document_references(
    samples: list[dict[str, object]],
) -> tuple[dict[str, str], dict[str, str]]:
    document_samples: dict[str, str] = {}
    sample_sources: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        path = Path(str(sample["path"]))
        file_hash = pdf_file_sha256(path)
        document_samples[f"pdf-{file_hash[:16]}"] = sample_id
        sample_sources[sample_id] = str(path)
    return document_samples, sample_sources


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

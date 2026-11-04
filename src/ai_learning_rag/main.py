"""FastAPI application entry point."""

from typing import Literal, Self

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator

from ai_learning_rag.text_processing import build_document, split_document


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: Literal["ok"]
    service: str


class DocumentPreviewRequest(BaseModel):
    """Plain text and chunking settings supplied by a learner."""

    title: str = Field(max_length=200)
    source: str = Field(max_length=2048)
    text: str
    chunk_size_chars: int = Field(default=500, ge=1, le=10_000)
    chunk_overlap_chars: int = Field(default=50, ge=0, le=9_999)

    @field_validator("title", "source", "text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain only whitespace."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Self:
        """Ensure each new window advances through the document."""
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        return self


class ChunkMetadata(BaseModel):
    """Structured fields used for filtering, ordering, and citations."""

    document_id: str
    title: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int


class ChunkPreview(BaseModel):
    """One chunk with content kept separate from structured metadata."""

    id: str
    content: str
    metadata: ChunkMetadata


class DocumentPreviewResponse(BaseModel):
    """Normalized document data that would be sent to a future vector index."""

    document_id: str
    normalized_text: str
    chunk_count: int
    chunks: list[ChunkPreview]


def create_app() -> FastAPI:
    """Create and configure the web application."""
    application = FastAPI(
        title="AI Learning RAG",
        description="A beginner-friendly RAG learning system.",
        version="0.1.0",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Report whether the application is ready to receive requests."""
        return HealthResponse(status="ok", service="ai-learning-rag")

    @application.post(
        "/documents/preview",
        response_model=DocumentPreviewResponse,
        tags=["documents"],
    )
    def preview_document(payload: DocumentPreviewRequest) -> DocumentPreviewResponse:
        """Show normalized text and chunks without persisting any data."""
        document = build_document(
            title=payload.title,
            source=payload.source,
            text=payload.text,
        )
        chunks = split_document(
            document,
            chunk_size_chars=payload.chunk_size_chars,
            chunk_overlap_chars=payload.chunk_overlap_chars,
        )

        return DocumentPreviewResponse(
            document_id=document.id,
            normalized_text=document.text,
            chunk_count=len(chunks),
            chunks=[
                ChunkPreview(
                    id=chunk.id,
                    content=chunk.text,
                    metadata=ChunkMetadata(
                        document_id=chunk.document_id,
                        title=document.title,
                        source=document.source,
                        chunk_index=chunk.chunk_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                    ),
                )
                for chunk in chunks
            ],
        )

    return application


app = create_app()

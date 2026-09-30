"""FastAPI application entry point."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: Literal["ok"]
    service: str


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

    return application


app = create_app()

"""Tests for the application health-check endpoint."""

from fastapi.testclient import TestClient

from ai_learning_rag.main import create_app


def test_health_check_reports_ready_service() -> None:
    """The health endpoint should expose a stable, machine-readable result."""
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ai-learning-rag",
    }

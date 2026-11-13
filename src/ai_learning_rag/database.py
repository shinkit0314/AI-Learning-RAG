"""Explicit PostgreSQL configuration without automatic migrations or secret defaults."""

import os
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url

DATABASE_URL_ENV = "AI_LEARNING_RAG_DATABASE_URL"
TEST_DATABASE_URL_ENV = "AI_LEARNING_RAG_TEST_DATABASE_URL"


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Connection settings loaded only when a PostgreSQL workflow is requested."""

    url: URL
    pool_size: int = 5
    max_overflow: int = 5
    pool_timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not self.url.drivername.startswith("postgresql"):
            raise ValueError("database URL must use PostgreSQL")
        if not self.url.database:
            raise ValueError("database URL must name a database")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be greater than zero")
        if self.max_overflow < 0:
            raise ValueError("max_overflow must not be negative")
        if self.pool_timeout_seconds <= 0:
            raise ValueError("pool_timeout_seconds must be greater than zero")

    @classmethod
    def from_env(cls, variable: str = DATABASE_URL_ENV) -> "DatabaseSettings":
        """Load one required URL without reading or mutating a dotenv file."""

        raw_url = os.getenv(variable, "").strip()
        if not raw_url:
            raise RuntimeError(f"{variable} is required")
        return cls(url=make_url(raw_url))

    def create_engine(self) -> Engine:
        """Create a pre-ping connection pool; schema migration remains a separate command."""

        return create_engine(
            self.url,
            pool_pre_ping=True,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout_seconds,
        )


def require_test_database_url(raw_url: str) -> URL:
    """Fail closed before an integration test can write to a non-test database."""

    url = make_url(raw_url)
    database = (url.database or "").lower()
    if not database.endswith("_test"):
        raise ValueError("integration database name must end with _test")
    if not url.drivername.startswith("postgresql"):
        raise ValueError("integration database URL must use PostgreSQL")
    return url

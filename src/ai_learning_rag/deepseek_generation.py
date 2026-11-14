"""DeepSeek Responses API adapter for the grounded generation boundary."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DOTENV_KEYS = frozenset({DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL_ENV})


@dataclass(frozen=True, slots=True)
class DeepSeekGenerationConfig:
    """Bounded settings for one DeepSeek Responses API request."""

    model: str = "deepseek-flash"
    reasoning_effort: str = "none"
    max_output_tokens: int = 512
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported reasoning effort")
        if self.max_output_tokens < 16:
            raise ValueError("max_output_tokens must be at least 16")
        _validate_base_url(self.base_url)

    @classmethod
    def from_env(
        cls,
        *,
        model: str = "deepseek-flash",
        reasoning_effort: str = "none",
        max_output_tokens: int = 512,
        environ: MutableMapping[str, str] | None = None,
    ) -> DeepSeekGenerationConfig:
        """Build settings with an optional environment-level endpoint override."""

        values = os.environ if environ is None else environ
        return cls(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            base_url=values.get(DEEPSEEK_BASE_URL_ENV, DEFAULT_DEEPSEEK_BASE_URL),
        )


class DeepSeekTextGenerator:
    """Generate text through DeepSeek while keeping provider details out of RAG logic."""

    def __init__(
        self,
        config: DeepSeekGenerationConfig | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or DeepSeekGenerationConfig.from_env()
        self._client = client or self._load_client(api_key)

    def generate(self, prompt: str) -> str:
        """Send one stateless generation request and return its text output."""

        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        response = self._client.responses.create(
            model=self.config.model,
            input=prompt,
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
        )
        output_text = response.output_text.strip()
        if not output_text:
            raise RuntimeError("DeepSeek returned a blank text response")
        return output_text

    def _load_client(self, api_key: str | None) -> Any:
        key = api_key or os.getenv(DEEPSEEK_API_KEY_ENV, "")
        if not key.strip():
            raise RuntimeError(f"{DEEPSEEK_API_KEY_ENV} is required for DeepSeek generation")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "DeepSeek generation dependency is missing. Run scripts/setup.ps1 -WithDeepSeek."
            ) from error
        return OpenAI(api_key=key, base_url=self.config.base_url)


def load_deepseek_dotenv(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Load only DeepSeek variables from one local dotenv file without overriding the process."""

    if not path.is_file():
        return
    values = os.environ if environ is None else environ
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or key not in _DOTENV_KEYS or key in values:
            continue
        values[key] = _dotenv_value(raw_value)


def _dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    comment = value.find(" #")
    return value[:comment].rstrip() if comment >= 0 else value


def _validate_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an absolute HTTP(S) URL without credentials or query")

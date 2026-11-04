"""OpenAI Responses API adapter for the grounded generation boundary."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OpenAIGenerationConfig:
    """Conservative settings for the first paid generation baseline."""

    model: str = "gpt-6-luna"
    reasoning_effort: str = "none"
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")
        if self.max_output_tokens < 16:
            raise ValueError("max_output_tokens must be at least 16")


class OpenAITextGenerator:
    """Generate text through OpenAI while keeping SDK details out of RAG logic."""

    def __init__(
        self,
        config: OpenAIGenerationConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or OpenAIGenerationConfig()
        self._client = client or self._load_client()

    def generate(self, prompt: str) -> str:
        """Send one stateless generation request and return its text output."""
        if not prompt.strip():
            raise ValueError("prompt must not be blank")

        response = self._client.responses.create(
            model=self.config.model,
            input=prompt,
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
            store=False,
        )
        output_text = response.output_text.strip()
        if not output_text:
            raise RuntimeError("OpenAI returned a blank text response")
        return output_text

    @staticmethod
    def _load_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "OpenAI generation dependency is missing. Run scripts/setup.ps1 -WithOpenAI."
            ) from error
        return OpenAI()

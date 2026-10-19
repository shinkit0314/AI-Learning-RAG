"""Tests for the OpenAI generation adapter without making API requests."""

from dataclasses import dataclass
from typing import Any

import pytest

from ai_learning_rag.openai_generation import OpenAIGenerationConfig, OpenAITextGenerator


@dataclass
class FakeResponse:
    """Small response object exposing the SDK convenience property we use."""

    output_text: str


class FakeResponses:
    """Capture Responses API parameters and return controlled text."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(output_text=self.output_text)


class FakeOpenAIClient:
    """Provide the same nested client shape as the official SDK."""

    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)


def test_generator_uses_cost_conscious_stateless_responses_settings() -> None:
    """The first baseline should be explicit, bounded, and not API-stored."""
    client = FakeOpenAIClient("基于证据的回答 [1]。")
    generator = OpenAITextGenerator(client=client)

    answer = generator.generate("证据与问题")

    assert answer == "基于证据的回答 [1]。"
    assert client.responses.calls == [
        {
            "model": "gpt-6-luna",
            "input": "证据与问题",
            "reasoning": {"effort": "none"},
            "max_output_tokens": 512,
            "store": False,
        }
    ]


def test_generator_rejects_blank_prompt_before_an_api_call() -> None:
    """Invalid input should not consume paid API usage."""
    client = FakeOpenAIClient("unused")
    generator = OpenAITextGenerator(client=client)

    with pytest.raises(ValueError, match="prompt must not be blank"):
        generator.generate("  ")
    assert client.responses.calls == []


def test_generator_rejects_blank_api_output() -> None:
    """An empty provider response must fail visibly."""
    generator = OpenAITextGenerator(client=FakeOpenAIClient("   "))

    with pytest.raises(RuntimeError, match="blank text response"):
        generator.generate("valid prompt")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": " "}, "model must not be blank"),
        ({"reasoning_effort": "unsupported"}, "unsupported reasoning effort"),
        ({"max_output_tokens": 15}, "at least 16"),
    ],
)
def test_invalid_generation_config_is_rejected(
    overrides: dict[str, Any],
    message: str,
) -> None:
    """Invalid paid-request settings should fail during local construction."""
    with pytest.raises(ValueError, match=message):
        OpenAIGenerationConfig(**overrides)

"""Tests for the DeepSeek generation adapter without making API requests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ai_learning_rag.deepseek_generation import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DeepSeekGenerationConfig,
    DeepSeekTextGenerator,
    load_deepseek_dotenv,
)


@dataclass
class FakeResponse:
    output_text: str


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(output_text=self.output_text)


class FakeDeepSeekClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)


def test_generator_uses_bounded_deepseek_responses_settings() -> None:
    client = FakeDeepSeekClient('{"status":"answered","answer":"回答 [1]"}')
    generator = DeepSeekTextGenerator(client=client)

    answer = generator.generate("证据与问题")

    assert answer == '{"status":"answered","answer":"回答 [1]"}'
    assert client.responses.calls == [
        {
            "model": "deepseek-flash",
            "input": "证据与问题",
            "reasoning": {"effort": "none"},
            "max_output_tokens": 512,
        }
    ]


def test_generator_rejects_blank_prompt_before_an_api_call() -> None:
    client = FakeDeepSeekClient("unused")
    generator = DeepSeekTextGenerator(client=client)

    with pytest.raises(ValueError, match="prompt must not be blank"):
        generator.generate("  ")
    assert client.responses.calls == []


def test_generator_rejects_blank_api_output() -> None:
    generator = DeepSeekTextGenerator(client=FakeDeepSeekClient("   "))

    with pytest.raises(RuntimeError, match="DeepSeek returned a blank text response"):
        generator.generate("valid prompt")


def test_config_uses_official_base_url_and_allows_environment_override() -> None:
    assert DeepSeekGenerationConfig.from_env(environ={}).base_url == DEFAULT_DEEPSEEK_BASE_URL
    assert (
        DeepSeekGenerationConfig.from_env(
            environ={"DEEPSEEK_BASE_URL": "https://gateway.example/v1"}
        ).base_url
        == "https://gateway.example/v1"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": " "}, "model must not be blank"),
        ({"reasoning_effort": "unsupported"}, "unsupported reasoning effort"),
        ({"max_output_tokens": 15}, "at least 16"),
        ({"base_url": "relative/path"}, "absolute HTTP"),
        ({"base_url": "https://user:secret@example.com"}, "without credentials"),
    ],
)
def test_invalid_generation_config_is_rejected(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DeepSeekGenerationConfig(**overrides)


def test_dotenv_loads_only_deepseek_values_without_overriding_process(
    tmp_path: Path,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secrets\n"
        "DEEPSEEK_API_KEY='file-key'\n"
        "DEEPSEEK_BASE_URL=https://gateway.example/v1 # optional override\n"
        "UNRELATED_SECRET=do-not-load\n",
        encoding="utf-8",
    )
    environ = {"DEEPSEEK_API_KEY": "session-key"}

    load_deepseek_dotenv(dotenv, environ=environ)

    assert environ == {
        "DEEPSEEK_API_KEY": "session-key",
        "DEEPSEEK_BASE_URL": "https://gateway.example/v1",
    }


def test_missing_dotenv_is_optional(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    load_deepseek_dotenv(tmp_path / ".env", environ=environ)

    assert environ == {}

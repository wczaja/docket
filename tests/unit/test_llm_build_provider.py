import pytest

from agent_triage.errors import DetectionError
from agent_triage.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROVIDER_URI,
    AnthropicProvider,
    OpenAIProvider,
    build_provider,
)


def test_build_anthropic_provider() -> None:
    provider = build_provider("anthropic:claude-haiku-4-5-20251001")
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-haiku-4-5-20251001"


def test_build_openai_provider() -> None:
    provider = build_provider("openai:gpt-4o-mini")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"


def test_build_provider_rejects_missing_colon() -> None:
    with pytest.raises(DetectionError, match="must be of the form"):
        build_provider("noslash")


def test_build_provider_rejects_empty_parts() -> None:
    with pytest.raises(DetectionError, match="non-empty"):
        build_provider(":model")
    with pytest.raises(DetectionError, match="non-empty"):
        build_provider("anthropic:")


def test_build_provider_rejects_unknown_provider() -> None:
    with pytest.raises(DetectionError, match="Unknown LLM provider"):
        build_provider("cohere:command")


def test_defaults_exposed() -> None:
    assert f"anthropic:{DEFAULT_ANTHROPIC_MODEL}" == DEFAULT_PROVIDER_URI
    assert DEFAULT_ANTHROPIC_MODEL.startswith("claude-")
    assert DEFAULT_OPENAI_MODEL.startswith("gpt-")

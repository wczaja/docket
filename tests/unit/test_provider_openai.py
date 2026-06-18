import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from docket.errors import CredentialError, DetectionError
from docket.llm._openai import OpenAIProvider


def _mock_client_returning(content: str | None, refusal: str | None = None) -> Any:
    client = MagicMock()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.message.refusal = refusal
    response.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


async def test_structured_complete_returns_parsed_json() -> None:
    payload = {"positive": True, "confidence": 0.7}
    client = _mock_client_returning(json.dumps(payload))
    provider = OpenAIProvider(model="gpt-4o-mini-test", client=client)
    result = await provider.structured_complete(
        system="sys",
        user="usr",
        schema={"type": "object", "properties": {"positive": {"type": "boolean"}}},
    )
    assert result == payload


async def test_structured_complete_passes_schema_as_response_format() -> None:
    schema = {
        "type": "object",
        "required": ["positive"],
        "properties": {"positive": {"type": "boolean"}},
    }
    client = _mock_client_returning(json.dumps({"positive": False}))
    provider = OpenAIProvider(model="m", client=client)
    await provider.structured_complete(system="s", user="u", schema=schema)
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "m"
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["schema"] == schema


async def test_structured_complete_raises_on_none_content() -> None:
    client = _mock_client_returning(None)
    provider = OpenAIProvider(client=client)
    with pytest.raises(DetectionError, match="no content"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_surfaces_refusal_as_detection_error() -> None:
    client = _mock_client_returning(None, refusal="I can't help with that request.")
    provider = OpenAIProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="refused the request"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_raises_on_invalid_json() -> None:
    client = _mock_client_returning("not json at all")
    provider = OpenAIProvider(client=client)
    with pytest.raises(DetectionError, match="non-JSON"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_raises_on_non_object_json() -> None:
    client = _mock_client_returning("[1, 2, 3]")
    provider = OpenAIProvider(client=client)
    with pytest.raises(DetectionError, match="non-object"):
        await provider.structured_complete(system="s", user="u", schema={})


def test_provider_constructs_without_client() -> None:
    provider = OpenAIProvider(model="m", api_key="test-key")
    assert provider.model == "m"


def test_preflight_raises_without_key_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(model="m")
    with pytest.raises(CredentialError, match="OPENAI_API_KEY"):
        provider.preflight()


def test_preflight_passes_with_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAIProvider(model="m", api_key="test-key").preflight()


def test_preflight_passes_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    OpenAIProvider(model="m").preflight()


def test_preflight_passes_with_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAIProvider(model="m", client=MagicMock()).preflight()


async def test_structured_complete_wraps_rate_limit_as_detection_error() -> None:
    import openai

    client = MagicMock()
    response = MagicMock()
    response.status_code = 429
    response.request = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError("rate limit", response=response, body=None)
    )
    provider = OpenAIProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="429"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_wraps_connection_error_as_detection_error() -> None:
    import openai

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )
    provider = OpenAIProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="connection error"):
        await provider.structured_complete(system="s", user="u", schema={})

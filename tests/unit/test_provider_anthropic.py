from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import ToolUseBlock

from docket.errors import CredentialError, DetectionError
from docket.llm._anthropic import AnthropicProvider


def _mock_client_returning(blocks: list[Any]) -> Any:
    client = MagicMock()
    response = MagicMock()
    response.content = blocks
    client.messages.create = AsyncMock(return_value=response)
    return client


def _make_tool_use_block(input_payload: Any, name: str = "submit") -> ToolUseBlock:
    return ToolUseBlock(id="b1", type="tool_use", name=name, input=input_payload)


async def test_structured_complete_returns_tool_input() -> None:
    client = _mock_client_returning([_make_tool_use_block({"positive": True, "confidence": 0.9})])
    provider = AnthropicProvider(model="claude-haiku-4-5-test", client=client)
    result = await provider.structured_complete(
        system="sys",
        user="usr",
        schema={"type": "object", "properties": {"positive": {"type": "boolean"}}},
    )
    assert result == {"positive": True, "confidence": 0.9}


async def test_structured_complete_passes_schema_as_tool_input() -> None:
    schema = {
        "type": "object",
        "required": ["positive"],
        "properties": {"positive": {"type": "boolean"}},
    }
    client = _mock_client_returning([_make_tool_use_block({"positive": False})])
    provider = AnthropicProvider(model="m", client=client)
    await provider.structured_complete(system="s", user="u", schema=schema)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "m"
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "submit"}
    assert call_kwargs["tools"][0]["input_schema"] == schema


async def test_structured_complete_raises_when_no_tool_use() -> None:
    text_block = MagicMock()
    text_block.type = "text"
    client = _mock_client_returning([text_block])
    provider = AnthropicProvider(client=client)
    with pytest.raises(DetectionError, match="did not emit"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_raises_on_non_dict_tool_input() -> None:
    # model_construct bypasses SDK-side validation, simulating a wire payload
    # where `input` is not a JSON object.
    block = ToolUseBlock.model_construct(
        id="b1", type="tool_use", name="submit", input="not an object"
    )
    client = _mock_client_returning([block])
    provider = AnthropicProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="non-object tool input"):
        await provider.structured_complete(system="s", user="u", schema={})


def test_provider_constructs_without_client() -> None:
    provider = AnthropicProvider(model="m", api_key="test-key")
    assert provider.model == "m"


def test_preflight_raises_without_key_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider(model="m")
    with pytest.raises(CredentialError, match="ANTHROPIC_API_KEY"):
        provider.preflight()


def test_preflight_passes_with_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    AnthropicProvider(model="m", api_key="test-key").preflight()


def test_preflight_passes_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    AnthropicProvider(model="m").preflight()


def test_preflight_passes_with_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    AnthropicProvider(model="m", client=MagicMock()).preflight()


async def test_structured_complete_wraps_rate_limit_as_detection_error() -> None:
    import anthropic

    client = MagicMock()
    response = MagicMock()
    response.status_code = 429
    response.request = MagicMock()
    client.messages.create = AsyncMock(
        side_effect=anthropic.RateLimitError("rate limit", response=response, body=None)
    )
    provider = AnthropicProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="429"):
        await provider.structured_complete(system="s", user="u", schema={})


async def test_structured_complete_wraps_connection_error_as_detection_error() -> None:
    import anthropic

    client = MagicMock()
    client.messages.create = AsyncMock(
        side_effect=anthropic.APIConnectionError(request=MagicMock())
    )
    provider = AnthropicProvider(model="m", client=client)
    with pytest.raises(DetectionError, match="connection error"):
        await provider.structured_complete(system="s", user="u", schema={})

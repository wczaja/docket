"""Anthropic provider: structured output via forced tool use.

The provider declares a single synthetic `submit` tool whose `input_schema` is
the rubric's `output_schema`, then forces the model to use it via
`tool_choice={"type": "tool", "name": "submit"}`. The model's `input` payload
on the resulting `tool_use` block IS the structured output.

The SDK client is constructed lazily on first request so an `AnthropicProvider`
can be inspected and registered without requiring credentials.
"""

import os
from typing import Any, cast

import anthropic
from anthropic.types import ToolUseBlock

from agent_triage.errors import CredentialError, DetectionError
from agent_triage.llm.base import ModelProvider

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_SUBMIT_TOOL_NAME = "submit"
_DEFAULT_MAX_RETRIES = 6


class AnthropicProvider(ModelProvider):
    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = client

    def preflight(self) -> None:
        if self._client is not None or self._api_key is not None:
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise CredentialError(
                "Anthropic provider has no API key. Set ANTHROPIC_API_KEY in "
                "your environment (or pass an explicit api_key when "
                "constructing the provider)."
            )

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                max_retries=self._max_retries,
            )
        return self._client

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": _SUBMIT_TOOL_NAME,
                        "description": "Submit structured output matching the schema.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": _SUBMIT_TOOL_NAME},
            )
        except anthropic.APIStatusError as e:
            # The SDK has already done its internal retries (max_retries above)
            # and is honoring Retry-After. Surface what's left as a
            # DetectionError so the classifier's own retry loop can take
            # another swing at it.
            raise DetectionError(f"Anthropic API error (status={e.status_code}): {e}") from e
        except anthropic.APIConnectionError as e:
            raise DetectionError(f"Anthropic connection error: {e}") from e
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == _SUBMIT_TOOL_NAME:
                if not isinstance(block.input, dict):
                    raise DetectionError(
                        f"Anthropic model {self.model!r} returned non-object tool input: "
                        f"{type(block.input).__name__}"
                    )
                return cast(dict[str, Any], block.input)
        raise DetectionError(
            f"Anthropic model {self.model!r} did not emit a `{_SUBMIT_TOOL_NAME}` tool_use block"
        )

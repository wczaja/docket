"""OpenAI provider: structured output via `response_format` json_schema mode.

Uses `strict: False` to keep the rubric DSL unconstrained by OpenAI strict-mode
rules (which would force every property to appear in `required` and disallow
`additionalProperties`). The schema sent here is therefore advisory; actual
schema conformance is enforced by post-validation in the llm_judge detector,
which validates every response against the rubric's `output_schema`.

The SDK client is constructed lazily on first request so an `OpenAIProvider`
can be built and registered without requiring credentials (the SDK rejects a
missing key at construction time).
"""

import json
import os
from typing import Any, cast

import openai

from agent_triage.errors import CredentialError, DetectionError
from agent_triage.llm.base import ModelProvider

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_RETRIES = 6


class OpenAIProvider(ModelProvider):
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = client

    def preflight(self) -> None:
        if self._client is not None or self._api_key is not None:
            return
        if not os.environ.get("OPENAI_API_KEY"):
            raise CredentialError(
                "OpenAI provider has no API key. Set OPENAI_API_KEY in your "
                "environment (or pass an explicit api_key when constructing "
                "the provider)."
            )

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(
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
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verdict",
                        "schema": schema,
                        "strict": False,
                    },
                },
            )
        except openai.APIStatusError as e:
            raise DetectionError(f"OpenAI API error (status={e.status_code}): {e}") from e
        except openai.APIConnectionError as e:
            raise DetectionError(f"OpenAI connection error: {e}") from e
        message = response.choices[0].message
        if message.refusal:
            raise DetectionError(
                f"OpenAI model {self.model!r} refused the request: {message.refusal}"
            )
        content = message.content
        if content is None:
            raise DetectionError(f"OpenAI model {self.model!r} returned no content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise DetectionError(
                f"OpenAI model {self.model!r} returned non-JSON content: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise DetectionError(
                f"OpenAI model {self.model!r} returned non-object JSON: {type(parsed).__name__}"
            )
        return cast(dict[str, Any], parsed)

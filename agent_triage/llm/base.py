"""ModelProvider ABC.

The classifier path is built on this single async method: pass system + user
text and a JSON Schema, get back a dict shaped by the schema. Each provider
requests structured output using its native mechanism (Anthropic forces a
tool call whose `input_schema` is ours; OpenAI uses `response_format` with
`strict: False`). Native mechanisms are best-effort: actual schema
conformance is enforced by post-validation in the llm_judge detector, which
validates every response against the rubric's `output_schema`.

Owning this abstraction in-tree keeps the structured-output contract under our
control. Third-party multi-provider wrappers tend to leak on exactly this
surface, and classifier reliability depends on the schema being enforced
end-to-end.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelProvider(ABC):
    """Async LLM completion with strict structured output."""

    model: str

    def preflight(self) -> None:  # noqa: B027 -- intentional no-op default
        """Validate the provider can run before any expensive caller work.

        Default is a no-op (mirrors `EmbeddingProvider.preflight`). Subclasses
        MAY override to detect missing credentials or malformed config at
        pipeline startup, raising a `CredentialError` (or other
        `AgentTriageError` subclass) so the caller fails fast — before any
        backend I/O — instead of after the full fetch pass.
        """

    @abstractmethod
    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single structured completion.

        Args:
            system: System prompt.
            user: User message text.
            schema: A JSON Schema (draft 2020-12) describing the required
                output shape. Implementations MUST request it via the
                provider-native mechanism, not by re-prompting on failure.
                Conformance is enforced downstream by the llm_judge
                detector's post-validation, so implementations need not
                validate the response against `schema` themselves.

        Returns:
            A dict produced by the model; expected (but not guaranteed) to
            match `schema`.

        Raises:
            DetectionError: if the provider call fails or returns no usable
                structured payload.
        """

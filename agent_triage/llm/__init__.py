"""LLM provider abstraction and factory.

Public surface:
  - `ModelProvider`: the ABC every chat-completion provider implements.
  - `AnthropicProvider`, `OpenAIProvider`: concrete chat providers.
  - `build_provider(uri)`: parse a `provider:model` URI and construct a chat provider.
  - `EmbeddingProvider`: the ABC for clustering embeddings.
  - `OpenAIEmbeddingProvider`: concrete embedding provider.
  - `build_embedding_provider(uri)`: parse a URI and construct an embedding provider.
  - `DEFAULT_PROVIDER_URI`, `DEFAULT_EMBEDDING_URI`: system defaults.
"""

from agent_triage.errors import DetectionError
from agent_triage.llm._anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider
from agent_triage.llm._openai import DEFAULT_OPENAI_MODEL, OpenAIProvider
from agent_triage.llm.base import ModelProvider
from agent_triage.llm.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)

DEFAULT_PROVIDER_URI = f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
DEFAULT_EMBEDDING_URI = f"openai:{DEFAULT_OPENAI_EMBEDDING_MODEL}"


def build_provider(uri: str) -> ModelProvider:
    """Parse a `provider:model` URI and return the matching chat provider.

    Example URIs:
        anthropic:claude-haiku-4-5-20251001
        openai:gpt-4o-mini
    """
    if ":" not in uri:
        raise DetectionError(
            f"Model URI {uri!r} must be of the form 'provider:model' "
            f"(e.g. 'anthropic:{DEFAULT_ANTHROPIC_MODEL}')"
        )
    provider_type, model = uri.split(":", 1)
    if not provider_type or not model:
        raise DetectionError(f"Model URI {uri!r} must have non-empty provider and model parts")
    if provider_type == "anthropic":
        return AnthropicProvider(model=model)
    if provider_type == "openai":
        return OpenAIProvider(model=model)
    raise DetectionError(f"Unknown LLM provider {provider_type!r}; supported: anthropic, openai")


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_EMBEDDING_URI",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_PROVIDER_URI",
    "AnthropicProvider",
    "EmbeddingProvider",
    "ModelProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
    "build_embedding_provider",
    "build_provider",
]

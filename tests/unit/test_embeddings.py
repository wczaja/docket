from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_triage.errors import CredentialError, DetectionError
from agent_triage.llm.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
    OpenAIEmbeddingProvider,
    VoyageEmbeddingProvider,
    build_embedding_provider,
)


def _mock_client_returning(vectors: list[list[float]]) -> Any:
    client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    client.embeddings.create = AsyncMock(return_value=response)
    return client


async def test_embed_returns_vectors() -> None:
    client = _mock_client_returning([[0.1, 0.2], [0.3, 0.4]])
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small-test", client=client)
    result = await provider.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_empty_input_skips_api_call() -> None:
    client = _mock_client_returning([])
    provider = OpenAIEmbeddingProvider(client=client)
    result = await provider.embed([])
    assert result == []
    client.embeddings.create.assert_not_called()


async def test_embed_passes_model_and_texts_to_client() -> None:
    client = _mock_client_returning([[0.0]])
    provider = OpenAIEmbeddingProvider(model="my-embedder", client=client)
    await provider.embed(["hello"])
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "my-embedder"
    assert kwargs["input"] == ["hello"]


async def test_embed_raises_on_count_mismatch() -> None:
    client = _mock_client_returning([[0.0]])  # one vector for two inputs
    provider = OpenAIEmbeddingProvider(client=client)
    with pytest.raises(DetectionError, match="returned 1 vectors for 2 inputs"):
        await provider.embed(["a", "b"])


def test_provider_constructs_without_client() -> None:
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    assert provider.model == DEFAULT_OPENAI_EMBEDDING_MODEL


def test_build_embedding_provider_default() -> None:
    provider = build_embedding_provider("openai:text-embedding-3-small")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "text-embedding-3-small"


def test_build_embedding_provider_rejects_missing_colon() -> None:
    with pytest.raises(DetectionError, match="must be of the form"):
        build_embedding_provider("noslash")


def test_build_embedding_provider_rejects_empty_parts() -> None:
    with pytest.raises(DetectionError, match="non-empty"):
        build_embedding_provider("openai:")
    with pytest.raises(DetectionError, match="non-empty"):
        build_embedding_provider(":model")


def test_build_embedding_provider_unknown_provider() -> None:
    with pytest.raises(DetectionError, match="Unknown embedding provider"):
        build_embedding_provider("cohere:embed-v3")


def test_preflight_noop_when_client_provided() -> None:
    client = _mock_client_returning([])
    provider = OpenAIEmbeddingProvider(client=client)
    provider.preflight()  # must not raise


def test_preflight_noop_when_api_key_provided() -> None:
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    provider.preflight()  # must not raise


def test_preflight_raises_credential_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = build_embedding_provider("openai:text-embedding-3-small")
    with pytest.raises(CredentialError, match="OPENAI_API_KEY"):
        provider.preflight()


def _voyage_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_voyage_embed_returns_vectors_in_index_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    provider = VoyageEmbeddingProvider(api_key="vk-test", client=_voyage_client(handler))
    result = await provider.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


async def test_voyage_embed_sends_model_and_input() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]})

    provider = VoyageEmbeddingProvider(
        model="voyage-test", api_key="vk-test", client=_voyage_client(handler)
    )
    await provider.embed(["hello"])
    assert captured["model"] == "voyage-test"
    assert captured["input"] == ["hello"]


async def test_voyage_embed_empty_input_skips_api_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no API call expected")

    provider = VoyageEmbeddingProvider(api_key="vk-test", client=_voyage_client(handler))
    assert await provider.embed([]) == []


async def test_voyage_embed_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "rate limited"})

    provider = VoyageEmbeddingProvider(api_key="vk-test", client=_voyage_client(handler))
    with pytest.raises(DetectionError, match="HTTP 429"):
        await provider.embed(["a"])


async def test_voyage_embed_raises_on_count_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]})

    provider = VoyageEmbeddingProvider(api_key="vk-test", client=_voyage_client(handler))
    with pytest.raises(DetectionError, match="returned 1 vectors for 2 inputs"):
        await provider.embed(["a", "b"])


def test_voyage_preflight_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    provider = VoyageEmbeddingProvider()
    with pytest.raises(CredentialError, match="VOYAGE_API_KEY"):
        provider.preflight()


def test_voyage_preflight_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "vk-env")
    provider = VoyageEmbeddingProvider()
    provider.preflight()  # must not raise


def test_build_embedding_provider_voyage() -> None:
    provider = build_embedding_provider("voyage:voyage-3.5-lite")
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert provider.model == DEFAULT_VOYAGE_EMBEDDING_MODEL

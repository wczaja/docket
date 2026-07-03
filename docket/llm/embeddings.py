"""Embedding provider abstraction.

The project keeps classification and clustering separate: LLM-as-judge for
classification, embeddings for clustering, never mixed. Embeddings live behind
their own ABC + provider implementations so the classifier (`ModelProvider`)
and the clusterer (`EmbeddingProvider`) stay decoupled. Each can be configured
independently in a rubric.

Three providers ship in-tree: `OpenAIEmbeddingProvider` (default),
`VoyageEmbeddingProvider` (plain HTTP, no extra dependency), and
`LocalEmbeddingProvider` (fastembed/ONNX behind the `local-embeddings`
extra — no API key at all). Anthropic doesn't expose an embeddings API
today, so single-key Anthropic deployments cluster via `local:` or skip
embeddings entirely with `--clustering mode-only`.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import httpx
import openai

from docket.errors import CredentialError, DetectionError

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-3.5-lite"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"

_NO_OPENAI_KEY_REMEDIES = (
    "Set OPENAI_API_KEY in your environment, or pick a no-OpenAI alternative: "
    "`--embedding voyage:voyage-3.5-lite` (needs VOYAGE_API_KEY), "
    "`--embedding local:BAAI/bge-small-en-v1.5` (no key; "
    'pip install "docket-runtime[local-embeddings]"), or '
    "`--clustering mode-only` (no embeddings at all)."
)


class EmbeddingProvider(ABC):
    """Async embeddings interface."""

    model: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` into per-text vector lists.

        Implementations MUST return one vector per input text in the same
        order; an empty input list MUST return an empty list (no API call).
        """

    def preflight(self) -> None:  # noqa: B027 -- intentional no-op default
        """Validate the provider can run before any expensive caller work.

        Default is a no-op. Subclasses MAY override to detect missing
        credentials or malformed config at pipeline startup, raising a
        `CredentialError` (or other `DocketError` subclass) so the
        caller fails fast instead of after the classification pass.
        """


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider.

    Defaults to `text-embedding-3-small` which matches the `agents/v1`
    rubric's `clustering.embedding_model`. Lazy client construction mirrors
    the `ModelProvider` pattern so the adapter is constructible without
    credentials in CI.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        *,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    def preflight(self) -> None:
        try:
            self._get_client()
        except openai.OpenAIError as e:
            raise CredentialError(
                "OpenAI embedding provider could not initialize: clustering "
                "defaults to OpenAI embeddings even when the classifier is "
                f"Anthropic. {_NO_OPENAI_KEY_REMEDIES} Underlying error: {e}"
            ) from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(model=self.model, input=texts)
        if len(response.data) != len(texts):
            raise DetectionError(
                f"OpenAI embeddings returned {len(response.data)} vectors for {len(texts)} inputs"
            )
        return [cast(list[float], list(item.embedding)) for item in response.data]


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embeddings provider over plain HTTP (`httpx`, no SDK).

    Lets Anthropic-only (or OpenAI-less) deployments cluster without buying
    a second LLM vendor account. The API key comes from `VOYAGE_API_KEY`
    unless passed explicitly; it is held in memory only and never logged.
    """

    def __init__(
        self,
        model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key if api_key is not None else os.environ.get("VOYAGE_API_KEY")
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    def preflight(self) -> None:
        if not self._api_key:
            raise CredentialError(
                "Voyage embedding provider has no API key. Set VOYAGE_API_KEY "
                "in your environment (or pass an explicit api_key when "
                "constructing the provider)."
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.preflight()
        client = self._get_client()
        try:
            response = await client.post(
                _VOYAGE_API_URL, json={"model": self.model, "input": texts}
            )
        except httpx.HTTPError as e:
            raise DetectionError(f"Voyage embeddings request failed: {e}") from e
        if response.status_code >= 400:
            raise DetectionError(
                f"Voyage embeddings returned HTTP {response.status_code} for model {self.model!r}"
            )
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            count = len(data) if isinstance(data, list) else "no"
            raise DetectionError(
                f"Voyage embeddings returned {count} vectors for {len(texts)} inputs"
            )
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [cast(list[float], list(item["embedding"])) for item in ordered]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local ONNX embeddings via fastembed — no API key, no per-call cost.

    Behind the `local-embeddings` extra (`pip install
    "docket-runtime[local-embeddings]"`) because the runtime dependency is
    heavyweight; the model file is fetched from Hugging Face on first use
    and cached locally. The go-to for single-key Anthropic deployments and
    air-gapped-ish environments that still want semantic clustering.
    """

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        *,
        engine: Any | None = None,
    ) -> None:
        self.model = model
        self._engine = engine

    def preflight(self) -> None:
        if self._engine is not None:
            return
        try:
            import fastembed  # noqa: F401
        except ImportError as e:
            raise CredentialError(
                "Local embeddings need the fastembed extra: "
                'pip install "docket-runtime[local-embeddings]". '
                "Alternatively use `--embedding openai:...` / `voyage:...` "
                "(API key required) or `--clustering mode-only` (no embeddings)."
            ) from e

    def _get_engine(self) -> Any:
        if self._engine is None:
            self.preflight()
            from fastembed import TextEmbedding

            self._engine = TextEmbedding(model_name=self.model)
        return self._engine

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        engine = await asyncio.to_thread(self._get_engine)  # first call may download the model

        def _run() -> list[list[float]]:
            return [[float(x) for x in vector] for vector in engine.embed(texts)]

        vectors = await asyncio.to_thread(_run)
        if len(vectors) != len(texts):
            raise DetectionError(
                f"Local embeddings returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors


def build_embedding_provider(uri: str) -> EmbeddingProvider:
    """Parse a `provider:model` URI and return the matching embedding provider.

    Example URIs:
        openai:text-embedding-3-small
        openai:text-embedding-3-large
        voyage:voyage-3.5-lite
        local:BAAI/bge-small-en-v1.5
    """
    if ":" not in uri:
        raise DetectionError(
            f"Embedding URI {uri!r} must be of the form 'provider:model' "
            f"(e.g. 'openai:{DEFAULT_OPENAI_EMBEDDING_MODEL}')"
        )
    provider_type, model = uri.split(":", 1)
    if not provider_type or not model:
        raise DetectionError(f"Embedding URI {uri!r} must have non-empty parts")
    if provider_type == "openai":
        return OpenAIEmbeddingProvider(model=model)
    if provider_type == "voyage":
        return VoyageEmbeddingProvider(model=model)
    if provider_type == "local":
        return LocalEmbeddingProvider(model=model)
    raise DetectionError(
        f"Unknown embedding provider {provider_type!r}; supported: openai, voyage, local"
    )

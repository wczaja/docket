"""Unit tests for the Clusterer subagent with a MockEmbeddingProvider.

We control the embeddings directly so HDBSCAN's output is deterministic per
test. Real OpenAI embeddings get exercised in the gated integration test.
"""

import logging

import pytest

from docket.agent.subagents import clusterer as clusterer_mod
from docket.agent.subagents.clusterer import cluster_per_mode
from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Classification
from docket.models.trace import OpenInferenceTrace, Span
from docket.rubric.spec import Clustering, Detection, Mode, Rubric, RubricMetadata


class _MockEmbeddingProvider(EmbeddingProvider):
    """Returns pre-canned vectors per text. Use to drive HDBSCAN deterministically."""

    def __init__(self, by_text: dict[str, list[float]]) -> None:
        self.model = "mock-embed"
        self._by_text = by_text
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            if t not in self._by_text:
                # default: orthogonal unit vector keyed on hash to avoid clustering
                idx = abs(hash(t)) % 32 + 16
                vector = [0.0] * 64
                vector[idx] = 1.0
                out.append(vector)
            else:
                out.append(self._by_text[t])
        return out


def _rubric(modes: list[Mode], min_cluster_size: int = 3, threshold: float = 0.82) -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="cluster-test", version="0.0.1"),
        modes=modes,
        clustering=Clustering(
            embedding_model="mock-embed",
            similarity_threshold=threshold,
            min_cluster_size=min_cluster_size,
        ),
    )


def _classification(
    trace_id: str,
    mode_id: str,
    *,
    excerpt: str | None = None,
    confidence: float | None = None,
) -> Classification:
    extra: dict[str, object] = {}
    if excerpt is not None:
        extra["excerpt"] = excerpt
    if confidence is not None:
        extra["confidence"] = confidence
    return Classification(
        trace_id=trace_id,
        rubric_version="cluster-test@0.0.1",
        mode_id=mode_id,
        positive=True,
        extra=extra,
    )


async def test_cluster_forms_when_min_size_reached() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    similar_vec = [1.0, 0.0]
    classifications = [
        _classification(f"t-{i}", "leak", excerpt=f"similar text {i}", confidence=0.9)
        for i in range(4)
    ]
    embeddings = {f"similar text {i}": similar_vec for i in range(4)}
    provider = _MockEmbeddingProvider(embeddings)

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.mode_id == "leak"
    assert sorted(cluster.member_trace_ids) == sorted([f"t-{i}" for i in range(4)])
    assert cluster.stats.size == 4


async def test_no_cluster_when_below_min_size() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    # Only 2 positives; cluster requires >= 3.
    classifications = [
        _classification("t-1", "leak", excerpt="alpha"),
        _classification("t-2", "leak", excerpt="alpha"),
    ]
    provider = _MockEmbeddingProvider({"alpha": [1.0, 0.0]})
    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert clusters == []


async def test_negative_classifications_are_skipped() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    classifications = [
        Classification(
            trace_id="t-1",
            rubric_version="v",
            mode_id="leak",
            positive=False,
        ),
        Classification(
            trace_id="t-2",
            rubric_version="v",
            mode_id="leak",
            positive=False,
        ),
    ]
    provider = _MockEmbeddingProvider({})
    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert clusters == []
    assert provider.calls == []  # never embedded


async def test_error_classifications_are_skipped() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    classifications = [
        Classification(
            trace_id="t-1",
            rubric_version="v",
            mode_id="leak",
            positive=False,
            error="judge timed out",
        ),
        Classification(
            trace_id="t-2",
            rubric_version="v",
            mode_id="leak",
            positive=False,
            error="judge timed out",
        ),
    ]
    provider = _MockEmbeddingProvider({})
    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert clusters == []


async def test_unknown_mode_ids_dropped() -> None:
    mode = Mode(id="known", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    classifications = [
        _classification("t-1", "ghost", excerpt="phantom"),
        _classification("t-2", "ghost", excerpt="phantom"),
        _classification("t-3", "ghost", excerpt="phantom"),
    ]
    provider = _MockEmbeddingProvider({"phantom": [1.0, 0.0]})
    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert clusters == []


async def test_representative_is_highest_confidence() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    confidences = [0.3, 0.9, 0.5]
    classifications = [
        _classification(f"t-{i}", "leak", excerpt=f"text{i}", confidence=conf)
        for i, conf in enumerate(confidences)
    ]
    same_vec = [1.0, 0.0, 0.0]
    provider = _MockEmbeddingProvider({f"text{i}": same_vec for i in range(3)})

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert len(clusters) == 1
    assert clusters[0].representative_trace_id == "t-1"  # confidence 0.9 was highest
    assert clusters[0].stats.max_confidence == 0.9
    assert clusters[0].stats.min_confidence == 0.3
    assert clusters[0].stats.mean_confidence is not None


async def test_multiple_modes_clustered_independently() -> None:
    modes = [
        Mode(id="a", severity="medium", detection=Detection(type="regex", pattern="x")),
        Mode(id="b", severity="high", detection=Detection(type="regex", pattern="y")),
    ]
    rubric = _rubric(modes, min_cluster_size=3)
    classifications = [_classification(f"a-{i}", "a", excerpt=f"a-text{i}") for i in range(4)] + [
        _classification(f"b-{i}", "b", excerpt=f"b-text{i}") for i in range(4)
    ]
    a_vec = [1.0, 0.0]
    b_vec = [0.0, 1.0]
    provider = _MockEmbeddingProvider(
        {f"a-text{i}": a_vec for i in range(4)} | {f"b-text{i}": b_vec for i in range(4)},
    )
    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    cluster_modes = {c.mode_id for c in clusters}
    assert cluster_modes == {"a", "b"}
    for cluster in clusters:
        assert cluster.stats.size == 4


async def test_excerpts_redacted_before_embedding_and_in_cluster() -> None:
    """PII in classification evidence is scrubbed before the external
    embeddings API sees it, and `representative_excerpt` stores the redacted
    text (so the drafter/report/queue/evals inherit it)."""
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    raw = "User jane.doe@example.com reported the failure"
    redacted = "User [REDACTED_EMAIL] reported the failure"
    classifications = [
        _classification(f"t-{i}", "leak", excerpt=raw, confidence=0.9) for i in range(3)
    ]
    provider = _MockEmbeddingProvider({redacted: [1.0, 0.0]})

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    # Texts sent to the embedding provider were already redacted.
    assert provider.calls == [[redacted] * 3]
    for call in provider.calls:
        assert all("jane.doe@example.com" not in text for text in call)
    assert len(clusters) == 1
    assert clusters[0].representative_excerpt == redacted


async def test_final_response_fallback_is_redacted() -> None:
    """The `trace.get_final_response()` fallback path is redacted too."""
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    redacted = "Call us at [REDACTED_PHONE] for help"
    traces_by_id = {
        f"t-{i}": OpenInferenceTrace(
            trace_id=f"t-{i}",
            spans=[
                Span(
                    span_id=f"s-{i}",
                    trace_id=f"t-{i}",
                    name="llm",
                    start_time_unix_nano=0,
                    end_time_unix_nano=1,
                    attributes={
                        "openinference.span.kind": "LLM",
                        "llm.output_messages.0.message.role": "assistant",
                        "llm.output_messages.0.message.content": (
                            "Call us at (555) 123-4567 for help"
                        ),
                    },
                )
            ],
        )
        for i in range(3)
    }
    # No excerpt in extra -> clusterer falls back to the trace's final response.
    classifications = [_classification(f"t-{i}", "leak", confidence=0.9) for i in range(3)]
    provider = _MockEmbeddingProvider({redacted: [1.0, 0.0]})

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
        traces_by_id=traces_by_id,
    )
    assert provider.calls == [[redacted] * 3]
    assert len(clusters) == 1
    assert clusters[0].representative_excerpt == redacted


class _MismatchEmbeddingProvider(EmbeddingProvider):
    """Always returns a single vector regardless of input length."""

    def __init__(self) -> None:
        self.model = "mismatch-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]


async def test_embedding_count_mismatch_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    classifications = [_classification(f"t-{i}", "leak", excerpt=f"text{i}") for i in range(3)]
    provider = _MismatchEmbeddingProvider()

    with caplog.at_level(logging.WARNING, logger="docket.agent.subagents.clusterer"):
        clusters = await cluster_per_mode(
            classifications,
            rubric=rubric,
            embedding_provider=provider,
        )
    assert clusters == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "leak" in message
    assert "1" in message
    assert "3" in message


class _RaisingHDBSCAN:
    """Stand-in for sklearn.cluster.HDBSCAN that raises during fit_predict.

    Used to simulate the known sklearn bug where `allow_single_cluster=True`
    plus `cluster_selection_epsilon > 0` crashes inside `traverse_upwards`
    when the EOM step picks the root as the only cluster.
    """

    def __init__(self, exc: BaseException, **_: object) -> None:
        self._exc = exc

    def fit_predict(self, _distances: object) -> object:
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("only 0-dimensional arrays can be converted to Python scalars"),
        IndexError("index 0 is out of bounds for axis 0 with size 0"),
    ],
)
async def test_falls_back_to_single_cluster_on_known_hdbscan_bug(
    monkeypatch: pytest.MonkeyPatch,
    exc: BaseException,
) -> None:
    def _factory(**kwargs: object) -> _RaisingHDBSCAN:
        return _RaisingHDBSCAN(exc, **kwargs)

    monkeypatch.setattr(clusterer_mod, "HDBSCAN", _factory)

    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    classifications = [
        _classification(f"t-{i}", "leak", excerpt=f"similar text {i}", confidence=0.5 + 0.1 * i)
        for i in range(5)
    ]
    # Vectors don't matter here — HDBSCAN is stubbed — but the embedding
    # provider still needs to return one vector per excerpt.
    provider = _MockEmbeddingProvider({f"similar text {i}": [1.0, 0.0] for i in range(5)})

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=provider,
    )
    assert len(clusters) == 1
    assert clusters[0].stats.size == 5
    assert sorted(clusters[0].member_trace_ids) == sorted(f"t-{i}" for i in range(5))


async def test_unrelated_typeerror_from_hdbscan_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TypeError that isn't the known sklearn bug should not be swallowed."""

    def _factory(**kwargs: object) -> _RaisingHDBSCAN:
        return _RaisingHDBSCAN(TypeError("unexpected keyword argument 'foo'"), **kwargs)

    monkeypatch.setattr(clusterer_mod, "HDBSCAN", _factory)

    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    classifications = [_classification(f"t-{i}", "leak", excerpt=f"text-{i}") for i in range(3)]
    provider = _MockEmbeddingProvider({f"text-{i}": [1.0, 0.0] for i in range(3)})

    with pytest.raises(TypeError, match="unexpected keyword"):
        await cluster_per_mode(
            classifications,
            rubric=rubric,
            embedding_provider=provider,
        )


# --- cluster_mode_only (the --clustering mode-only fallback) -----------------


def test_mode_only_groups_all_positives_per_mode() -> None:
    """Dissimilar excerpts that embeddings would separate still land in one
    cluster per mode — that's the documented lossy trade."""
    leak = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    halluc = Mode(id="halluc", severity="critical", detection=Detection(type="regex", pattern="y"))
    rubric = _rubric([leak, halluc], min_cluster_size=3)
    classifications = [
        _classification(f"l-{i}", "leak", excerpt=f"completely different text {i}")
        for i in range(3)
    ] + [_classification(f"h-{i}", "halluc", excerpt=f"unrelated falsehood {i}") for i in range(4)]

    clusters = clusterer_mod.cluster_mode_only(classifications, rubric=rubric)

    assert {(c.mode_id, c.stats.size) for c in clusters} == {("leak", 3), ("halluc", 4)}


def test_mode_only_respects_min_cluster_size() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=3)
    classifications = [_classification(f"t-{i}", "leak", excerpt="a") for i in range(2)]
    assert clusterer_mod.cluster_mode_only(classifications, rubric=rubric) == []


def test_mode_only_skips_negatives_errors_and_unknown_modes() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    ok = [_classification(f"t-{i}", "leak", excerpt="a") for i in range(2)]
    negative = Classification(
        trace_id="t-neg", rubric_version="cluster-test@0.0.1", mode_id="leak", positive=False
    )
    errored = Classification(
        trace_id="t-err",
        rubric_version="cluster-test@0.0.1",
        mode_id="leak",
        positive=True,
        error="boom",
    )
    unknown = _classification("t-unk", "not-in-rubric", excerpt="a")

    clusters = clusterer_mod.cluster_mode_only([*ok, negative, errored, unknown], rubric=rubric)

    assert len(clusters) == 1
    assert sorted(clusters[0].member_trace_ids) == ["t-0", "t-1"]


def test_mode_only_representative_is_highest_confidence() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    classifications = [
        _classification("t-low", "leak", excerpt="low", confidence=0.4),
        _classification("t-high", "leak", excerpt="high", confidence=0.95),
    ]
    clusters = clusterer_mod.cluster_mode_only(classifications, rubric=rubric)
    assert clusters[0].representative_trace_id == "t-high"


def test_mode_only_redacts_excerpts() -> None:
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = _rubric([mode], min_cluster_size=2)
    classifications = [
        _classification(f"t-{i}", "leak", excerpt="reach me at user@example.com", confidence=0.9)
        for i in range(2)
    ]
    clusters = clusterer_mod.cluster_mode_only(classifications, rubric=rubric)
    assert "user@example.com" not in clusters[0].representative_excerpt

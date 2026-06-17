"""§7 acceptance: classifier failure on 3 traces does not abort the run.

We wrap the real classifier path with a Detector that throws on a fixed set
of trace IDs. The Phase 5 retry budget exhausts after `max_retries`, the
Classification gets `error` set, the run continues to clustering + drafting
+ report. The error inventory lists the failed (trace_id, mode_id) pairs.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_triage.adapters.base import TraceBackend
from agent_triage.agent.triage import run_triage_pipeline
from agent_triage.detectors.base import Detector
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.llm.embeddings import EmbeddingProvider
from agent_triage.models.classification import Annotation
from agent_triage.models.trace import OpenInferenceTrace, Span, TraceLike, Verdict
from agent_triage.rubric.spec import Detection, Mode, Rubric, RubricMetadata


class _FakeBackend(TraceBackend):
    def __init__(self, traces: dict[str, OpenInferenceTrace]) -> None:
        self.traces = traces
        self.annotations: list[Annotation] = []

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        return list(self.traces.keys())

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        return self.traces[trace_id]

    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        self.annotations.append(annotation)

    async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
        pass

    async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
        return set()


class _MockLLMProvider(ModelProvider):
    def __init__(self) -> None:
        self.model = "mock"

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {"title": "Test issue", "body": "Body" * 20}


class _MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = "mock-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _FailOnSpecificTraces(Detector):
    """Wraps a real detector to throw DetectionError for a fixed set of trace IDs."""

    def __init__(self, fail_trace_ids: set[str]) -> None:
        self._fail = fail_trace_ids

    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        if trace.trace_id in self._fail:
            msg = f"injected failure for trace {trace.trace_id!r}"
            raise DetectionError(msg)
        return Verdict(positive=True, extra={})


def _trace(trace_id: str) -> OpenInferenceTrace:
    return OpenInferenceTrace(
        trace_id=trace_id,
        spans=[
            Span(
                span_id="s",
                trace_id=trace_id,
                name="x",
                start_time_unix_nano=0,
                end_time_unix_nano=1_000_000,
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "say something",
                },
            ),
        ],
    )


def _rubric() -> Rubric:
    return Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="failure-injection", version="0.1.0"),
        modes=[
            Mode(
                id="m1",
                severity="critical",
                detection=Detection(type="regex", pattern="x"),
            ),
        ],
    )


async def test_classifier_failure_on_three_traces_does_not_abort_run(
    tmp_path: Path,
) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}") for i in range(10)})
    rubric = _rubric()

    # Inject a failing detector for the regex slot. The Classifier caches by
    # detection type, so the FIRST classify_trace call materializes the regex
    # detector via get_detector(); we patch by pre-seeding the cache.

    fail_ids = {"t-2", "t-5", "t-7"}

    # The Classifier's _detectors cache is keyed on detection type. Inject our
    # failing detector before any classify_trace runs by monkey-patching the
    # internal cache when the Classifier is constructed inside run_triage_pipeline.
    # We do this by wrapping run_triage_pipeline's Classifier instantiation
    # via patching get_detector to return our failing detector for type=regex.
    failing = _FailOnSpecificTraces(fail_ids)
    from agent_triage import detectors as detectors_module

    real_get = detectors_module.get_detector

    def fake_get(
        detection_type: str, llm_provider: ModelProvider | None = None, batch_size: int = 1
    ) -> Detector:  # noqa: ARG001
        if detection_type == "regex":
            return failing
        return real_get(detection_type, llm_provider=llm_provider, batch_size=batch_size)

    from unittest.mock import patch

    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)

    with patch(
        "agent_triage.agent.subagents.classifier.get_detector",
        side_effect=fake_get,
    ):
        result = await run_triage_pipeline(
            backend=backend,
            rubric=rubric,
            since=since,
            until=until,
            llm_provider=_MockLLMProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
        )

    # Run did NOT abort.
    assert result.run_report.trace_count == 10
    # 3 traces have errors on the m1 mode.
    failed_results = [r for r in result.run_report.trace_results if r.error_modes]
    failed_ids = {r.trace_id for r in failed_results}
    assert failed_ids == fail_ids
    # The other 7 are positive.
    positive_results = [r for r in result.run_report.trace_results if r.positive_modes]
    assert len(positive_results) == 7

    # The mode stats reflect both the errors and the positives.
    m1_stats = next(ms for ms in result.run_report.mode_stats if ms.mode_id == "m1")
    assert m1_stats.error_count == 3
    assert m1_stats.positive_count == 7

    # The report.md surfaces the errors.
    assert "## Detector errors" in result.report_markdown
    for tid in fail_ids:
        assert tid in result.report_markdown


# Used so pytest-asyncio picks up the test (the conftest configures asyncio_mode = "auto").
def test_failure_injection_uses_the_real_classifier() -> None:
    """Static sanity: import the Classifier and confirm its retry knobs are set
    to a small enough number that the unit test doesn't time out under default
    backoff. The integration knob the test cares about is `max_retries=3` at
    Classifier construction time, which is the design §4.4 default."""
    from agent_triage.agent.subagents.classifier import Classifier

    classifier = Classifier(_MockLLMProvider())
    # The pipeline constructs the Classifier with default kwargs; verify
    # max_retries matches design §4.4.
    assert classifier._max_retries == 3  # type: ignore[attr-defined]  # noqa: SLF001

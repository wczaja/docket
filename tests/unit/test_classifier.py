"""Tests for the Classifier subagent: concurrency + retry + mode iteration."""

from typing import Any

import pytest

from agent_triage.agent.subagents.classifier import Classifier, flatten_classifications
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import OpenInferenceTrace, Span
from agent_triage.rubric.spec import Detection, Mode, Rubric, RubricMetadata


class _MockLLMProvider(ModelProvider):
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.model = "mock"
        self._response = response or {"positive": False}

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        return self._response


def _rubric_regex(pattern: str = "x") -> Rubric:
    return Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[
            Mode(id="m1", severity="medium", detection=Detection(type="regex", pattern=pattern)),
        ],
    )


def _trace(trace_id: str, text: str) -> OpenInferenceTrace:
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
                    "llm.output_messages.0.message.content": text,
                },
            ),
        ],
    )


def test_classifier_rejects_zero_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        Classifier(_MockLLMProvider(), concurrency=0)


def test_classifier_rejects_zero_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        Classifier(_MockLLMProvider(), max_retries=0)


async def test_classify_trace_returns_per_mode_results() -> None:
    classifier = Classifier(_MockLLMProvider(), concurrency=1)
    rubric = _rubric_regex("hello")
    classifications = await classifier.classify_trace("t-1", _trace("t-1", "hello world"), rubric)
    assert len(classifications) == 1
    assert classifications[0].positive is True
    assert classifications[0].mode_id == "m1"


async def test_classify_all_runs_concurrently() -> None:
    classifier = Classifier(_MockLLMProvider(), concurrency=4)
    rubric = _rubric_regex("hello")
    traces = [(f"t-{i}", _trace(f"t-{i}", "hello world")) for i in range(8)]
    results = await classifier.classify_all(traces, rubric)
    assert len(results) == 8
    for items in results.values():
        assert len(items) == 1
        assert items[0].positive is True


async def test_classifier_retries_then_records_error() -> None:
    """Detector that always errors -> Classification with `error` set after retries."""

    rubric = Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[
            # metric_threshold against a metric that doesn't exist on the trace -> DetectionError
            Mode(
                id="missing-metric",
                severity="medium",
                detection=Detection(
                    type="metric_threshold",
                    metric="absent",
                    threshold=1.0,
                    operator=">",
                ),
            ),
        ],
    )
    classifier = Classifier(
        _MockLLMProvider(),
        max_retries=2,
        backoff_base_s=0.0,
        backoff_cap_s=0.0,
    )
    classifications = await classifier.classify_trace("t-1", _trace("t-1", "hi"), rubric)
    assert len(classifications) == 1
    assert classifications[0].error is not None
    assert "after 2 attempts" in classifications[0].error


async def test_progress_callback_called_per_trace() -> None:
    classifier = Classifier(_MockLLMProvider(), concurrency=1)
    rubric = _rubric_regex("x")
    traces = [(f"t-{i}", _trace(f"t-{i}", "x")) for i in range(3)]
    progress_log: list[tuple[str, int, int]] = []

    async def _progress(trace_id: str, current: int, total: int) -> None:
        progress_log.append((trace_id, current, total))

    await classifier.classify_all(traces, rubric, on_progress=_progress)
    assert len(progress_log) == 3
    for _trace_id, _, total in progress_log:
        assert total == 3


def test_flatten_classifications_preserves_dict_order() -> None:
    from agent_triage.models.classification import Classification

    a = Classification(trace_id="a", rubric_version="v", mode_id="m", positive=False)
    b = Classification(trace_id="b", rubric_version="v", mode_id="m", positive=False)
    flat = flatten_classifications({"first": [a], "second": [b]})
    assert flat == [a, b]


async def test_classifier_succeeds_after_transient_failure() -> None:
    """If the detector fails once then succeeds, the final Classification has no error."""

    from agent_triage.detectors.base import Detector
    from agent_triage.models.trace import TraceLike, Verdict

    class _FlakyDetector(Detector):
        def __init__(self) -> None:
            self.attempts = 0

        async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
            self.attempts += 1
            if self.attempts < 2:
                raise DetectionError("transient")
            return Verdict(positive=True, extra={})

    flaky = _FlakyDetector()
    classifier = Classifier(
        _MockLLMProvider(),
        max_retries=3,
        backoff_base_s=0.0,
    )
    # Inject the flaky detector for one detection type.
    classifier._detectors["regex"] = flaky  # type: ignore[attr-defined]  # noqa: SLF001
    rubric = _rubric_regex("doesn't matter")
    classifications = await classifier.classify_trace("t-1", _trace("t-1", "anything"), rubric)
    assert flaky.attempts == 2
    assert classifications[0].error is None
    assert classifications[0].positive is True


async def test_classify_all_settles_siblings_before_reraising_unexpected_error() -> None:
    """A non-DetectionError from one trace must not abandon in-flight siblings."""
    from agent_triage.detectors.base import Detector
    from agent_triage.models.trace import TraceLike, Verdict

    evaluated: list[str] = []

    class _ExplodingDetector(Detector):
        async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
            evaluated.append(trace.trace_id)
            if trace.trace_id == "t-boom":
                raise RuntimeError("unexpected")
            return Verdict(positive=False)

    classifier = Classifier(_MockLLMProvider(), concurrency=4)
    classifier._detectors["regex"] = _ExplodingDetector()  # type: ignore[attr-defined]  # noqa: SLF001
    rubric = _rubric_regex()
    traces = [(tid, _trace(tid, "text")) for tid in ("t-boom", "t-2", "t-3")]
    with pytest.raises(RuntimeError, match="unexpected"):
        await classifier.classify_all(traces, rubric)
    # All siblings ran to completion before the failure was re-raised.
    assert set(evaluated) == {"t-boom", "t-2", "t-3"}

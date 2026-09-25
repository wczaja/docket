"""Tests for the Classifier subagent: concurrency + retry + mode iteration."""

import asyncio
import re
from collections.abc import Callable
from typing import Any

import pytest

from docket.agent.subagents import classifier as classifier_module
from docket.agent.subagents.classifier import Classifier, flatten_classifications
from docket.cost import DEFAULT_INPUT_TOKENS_PER_CALL, estimate_cost
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.classification import Classification
from docket.models.trace import OpenInferenceTrace, Span
from docket.rubric.loader import load_rubric
from docket.rubric.spec import Detection, Mode, Rubric, RubricMetadata


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
        apiVersion="docket.dev/v1",
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
        apiVersion="docket.dev/v1",
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
    from docket.models.classification import Classification

    a = Classification(trace_id="a", rubric_version="v", mode_id="m", positive=False)
    b = Classification(trace_id="b", rubric_version="v", mode_id="m", positive=False)
    flat = flatten_classifications({"first": [a], "second": [b]})
    assert flat == [a, b]


async def test_classifier_succeeds_after_transient_failure() -> None:
    """If the detector fails once then succeeds, the final Classification has no error."""

    from docket.detectors.base import Detector
    from docket.models.trace import TraceLike, Verdict

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
    from docket.detectors.base import Detector
    from docket.models.trace import TraceLike, Verdict

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


# --- Budget mode: batched llm_judge calls ----------------------------------------

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["positive"],
    "properties": {"positive": {"type": "boolean"}},
}


class _JudgeProvider(ModelProvider):
    """Answers single and batched judge calls; records how many traces each judged.

    A trace is positive iff its text contains "<<pos>>". Batched prompts are
    split on the detector's `=== Trace N ===` headers; subclasses override
    `batched` to answer them badly.
    """

    model = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        wrapper = schema.get("properties", {}).get("verdicts")
        if wrapper is None:
            self.calls.append(1)
            return self.single(user)
        self.calls.append(wrapper["minItems"])
        return self.batched(re.split(r"=== Trace \d+ ===\n", user)[1:])

    def single(self, text: str) -> dict[str, Any]:
        return {"positive": "<<pos>>" in text, "confidence": 0.9}

    def batched(self, blocks: list[str]) -> dict[str, Any]:
        return {"verdicts": [self.single(b) for b in blocks]}


def _agent_trace(trace_id: str, text: str) -> OpenInferenceTrace:
    """An assistant reply, then a handoff tool call (so handoff gates pass)."""
    return OpenInferenceTrace(
        trace_id=trace_id,
        spans=[
            Span(
                span_id="llm",
                trace_id=trace_id,
                name="llm",
                start_time_unix_nano=0,
                end_time_unix_nano=1_000_000,
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": text,
                },
            ),
            Span(
                span_id="tool",
                trace_id=trace_id,
                name="handoff_to_specialist",
                start_time_unix_nano=1_000_000,
                end_time_unix_nano=2_000_000,
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": "handoff_to_specialist",
                },
            ),
        ],
    )


def _agent_traces(n: int, text: Callable[[int], str] = str) -> list[tuple[str, OpenInferenceTrace]]:
    return [(f"t-{i}", _agent_trace(f"t-{i}", f"reply {text(i)}")) for i in range(n)]


def _judge(prompt: str) -> Detection:
    return Detection(type="llm_judge", prompt=prompt, output_schema=_VERDICT_SCHEMA)


def _judge_rubric() -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[Mode(id="judge", severity="high", detection=_judge("Is the reply wrong?"))],
    )


def _mixed_rubric() -> Rubric:
    """Two batchable judges, a regex, and a composite whose judge runs on every
    `_agent_trace` (its tool_call gate always matches)."""
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[
            Mode(id="judge-a", severity="high", detection=_judge("Is failure A present?")),
            Mode(id="says-hi", severity="low", detection=Detection(type="regex", pattern="hello")),
            Mode(
                id="gated-judge",
                severity="high",
                detection=Detection(
                    type="composite",
                    operator="and",
                    operands=[
                        Detection(type="tool_call", tool_calls=["handoff_to_specialist"]),
                        _judge("Did the handoff fail?"),
                    ],
                ),
            ),
            Mode(id="judge-b", severity="medium", detection=_judge("Is failure B present?")),
        ],
    )


def test_classifier_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Classifier(_MockLLMProvider(), batch_size=0)


@pytest.mark.parametrize("rubric_source", ["mixed", "docket.dev/builtin/agents/v1"])
@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 3, 8, 32])
async def test_provider_calls_equal_estimated_calls(
    batch_size: int, concurrency: int, rubric_source: str
) -> None:
    """The --dry-run / budget-gate estimate counts exactly the calls a run makes."""
    rubric = _mixed_rubric() if rubric_source == "mixed" else load_rubric(rubric_source)
    traces = _agent_traces(20)
    provider = _JudgeProvider()
    results = await Classifier(
        provider, batch_size=batch_size, concurrency=concurrency
    ).classify_all(traces, rubric)

    estimate = estimate_cost(
        trace_count=len(traces), rubric=rubric, model=provider.model, batch_size=batch_size
    )
    assert len(provider.calls) == estimate.total_calls
    # Tokens are priced per trace judged, and batching judges every trace all the same.
    assert sum(provider.calls) * DEFAULT_INPUT_TOKENS_PER_CALL == estimate.total_input_tokens
    assert max(provider.calls) == min(batch_size, len(traces))
    assert all(c.error is None for items in results.values() for c in items)


async def test_budget_mode_batches_plain_judges_but_not_composites() -> None:
    provider = _JudgeProvider()
    await Classifier(provider, batch_size=8).classify_all(_agent_traces(20), _mixed_rubric())
    # judge-a and judge-b: 8 + 8 + 4 each; gated-judge: one call per trace.
    assert sorted(provider.calls) == sorted([8, 8, 4] * 2 + [1] * 20)


async def test_batched_verdicts_match_unbatched_verdicts() -> None:
    traces = _agent_traces(10, lambda i: f"{i} <<pos>>" if i % 3 == 0 else str(i))

    def _verdicts(results: dict[str, list[Classification]]) -> list[tuple[Any, ...]]:
        return [
            (tid, c.mode_id, c.positive, c.extra, c.error)
            for tid, items in results.items()
            for c in items
        ]

    unbatched = await Classifier(_JudgeProvider()).classify_all(traces, _mixed_rubric())
    batched = await Classifier(_JudgeProvider(), batch_size=4).classify_all(traces, _mixed_rubric())
    assert _verdicts(batched) == _verdicts(unbatched)
    assert batched["t-3"][0].positive is True
    assert batched["t-4"][0].positive is False


class _SchemaViolatingEntry(_JudgeProvider):
    """Batched answers for traces saying "<<bad>>" violate the schema."""

    def batched(self, blocks: list[str]) -> dict[str, Any]:
        return {
            "verdicts": [{"positive": "yes"} if "<<bad>>" in b else self.single(b) for b in blocks]
        }


async def test_bad_batched_verdict_is_rejudged_alone_and_spares_the_rest() -> None:
    traces = _agent_traces(4, lambda i: "<<bad>> <<pos>>" if i == 2 else f"{i}")
    provider = _SchemaViolatingEntry()
    results = await Classifier(provider, batch_size=4, backoff_base_s=0.0).classify_all(
        traces, _judge_rubric()
    )
    # One batched call; only t-2 needs a second, single-trace attempt.
    assert provider.calls == [4, 1]
    assert [items[0].positive for items in results.values()] == [False, False, True, False]
    assert all(items[0].error is None for items in results.values())


class _OneVerdictShort(_JudgeProvider):
    def batched(self, blocks: list[str]) -> dict[str, Any]:
        return {"verdicts": [self.single(b) for b in blocks[:-1]]}


async def test_unsplittable_batched_response_rejudges_each_trace_alone() -> None:
    provider = _OneVerdictShort()
    results = await Classifier(provider, batch_size=4, backoff_base_s=0.0).classify_all(
        _agent_traces(5), _judge_rubric()
    )
    # Chunk t-0..t-3 can't be attributed, so each trace retries alone; t-4's
    # chunk of one was a single-trace call to begin with.
    assert provider.calls[0] == 4
    assert sorted(provider.calls[1:]) == [1] * 5
    assert all(items[0].error is None for items in results.values())


class _Down(_JudgeProvider):
    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        await super().structured_complete(system, user, schema)
        raise DetectionError("provider down")


@pytest.mark.parametrize("batch_size", [1, 3])
async def test_batching_keeps_per_trace_retries_and_backoff(
    batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace gets the same attempts, backoff and error with or without batching."""
    delays: list[float] = []

    async def _no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(classifier_module.asyncio, "sleep", _no_sleep)
    provider = _Down()
    results = await Classifier(provider, batch_size=batch_size, max_retries=3).classify_all(
        _agent_traces(3), _judge_rubric()
    )
    for items in results.values():
        assert items[0].error == "after 3 attempts: provider down"
        assert items[0].positive is False
    assert sorted(delays) == [1.0] * 3 + [2.0] * 3
    if batch_size == 1:
        assert provider.calls == [1] * 9
    else:
        # The shared batched call was each trace's first attempt.
        assert provider.calls == [3] + [1] * 6


async def test_failed_batched_attempt_is_final_when_max_retries_is_one() -> None:
    provider = _OneVerdictShort()
    results = await Classifier(provider, batch_size=4, max_retries=1).classify_all(
        _agent_traces(4), _judge_rubric()
    )
    assert provider.calls == [4]
    for items in results.values():
        error = items[0].error
        assert error is not None
        assert error.startswith("after 1 attempts: Mode 'judge': expected 4 verdicts")


class _InFlightJudge(_JudgeProvider):
    def __init__(self, *, short_batches: bool = False) -> None:
        super().__init__()
        self.short_batches = short_batches
        self.in_flight = 0
        self.max_in_flight = 0

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.001)
            return await super().structured_complete(system, user, schema)
        finally:
            self.in_flight -= 1

    def batched(self, blocks: list[str]) -> dict[str, Any]:
        if self.short_batches:
            return {"verdicts": []}
        return super().batched(blocks)


@pytest.mark.parametrize(("batch_size", "short_batches"), [(1, False), (4, False), (4, True)])
async def test_concurrency_bounds_in_flight_calls_including_retries(
    batch_size: int, short_batches: bool
) -> None:
    provider = _InFlightJudge(short_batches=short_batches)
    await Classifier(
        provider, batch_size=batch_size, concurrency=3, backoff_base_s=0.0
    ).classify_all(_agent_traces(20), _mixed_rubric())
    assert provider.max_in_flight == 3


class _EarlyTracesSlow(_JudgeProvider):
    """Answers for lower-numbered traces take more event-loop turns, so work
    finishes in reverse input order (deterministically, no wall clock)."""

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        match = re.search(r"reply (\d+)", user)
        assert match is not None
        for _ in range(10 * (10 - int(match.group(1)))):
            await asyncio.sleep(0)
        return await super().structured_complete(system, user, schema)


@pytest.mark.parametrize("batch_size", [1, 4])
async def test_results_follow_input_order_not_completion_order(batch_size: int) -> None:
    rubric = _mixed_rubric()
    traces = _agent_traces(10)
    completed: list[str] = []

    async def _progress(trace_id: str, _done: int, _total: int) -> None:
        completed.append(trace_id)

    results = await Classifier(
        _EarlyTracesSlow(), batch_size=batch_size, concurrency=10
    ).classify_all(traces, rubric, on_progress=_progress)

    ids = [tid for tid, _ in traces]
    assert list(results) == ids
    for items in results.values():
        assert [c.mode_id for c in items] == [m.id for m in rubric.modes]
    assert sorted(completed) == sorted(ids)
    if batch_size == 1:
        assert completed == ids[::-1]


async def test_progress_reports_each_trace_once_in_budget_mode() -> None:
    progress: list[tuple[str, int, int]] = []

    async def _progress(trace_id: str, done: int, total: int) -> None:
        progress.append((trace_id, done, total))

    traces = _agent_traces(10)
    await Classifier(_JudgeProvider(), batch_size=4, concurrency=2).classify_all(
        traces, _mixed_rubric(), on_progress=_progress
    )
    assert sorted(tid for tid, _, _ in progress) == sorted(tid for tid, _ in traces)
    assert [done for _, done, _ in progress] == list(range(1, 11))
    assert {total for _, _, total in progress} == {10}


class _ExplodesOnTraceZero(_JudgeProvider):
    def batched(self, blocks: list[str]) -> dict[str, Any]:
        if any(re.search(r"\breply 0\b", b) for b in blocks):
            raise RuntimeError("unexpected")
        return super().batched(blocks)


async def test_unexpected_error_in_batched_call_settles_siblings_then_raises() -> None:
    provider = _ExplodesOnTraceZero()
    with pytest.raises(RuntimeError, match="unexpected"):
        await Classifier(provider, batch_size=2, concurrency=4).classify_all(
            _agent_traces(6), _judge_rubric()
        )
    # Every chunk's call still ran before the failure surfaced.
    assert provider.calls == [2, 2, 2]


async def test_rubric_without_modes_still_reports_every_trace() -> None:
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[],
    )
    progress: list[str] = []

    async def _progress(trace_id: str, _done: int, _total: int) -> None:
        progress.append(trace_id)

    results = await Classifier(_JudgeProvider(), batch_size=4).classify_all(
        _agent_traces(3), rubric, on_progress=_progress
    )
    assert results == {"t-0": [], "t-1": [], "t-2": []}
    assert sorted(progress) == ["t-0", "t-1", "t-2"]

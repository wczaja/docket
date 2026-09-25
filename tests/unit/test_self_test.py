import re
from typing import Any

from docket.llm.base import ModelProvider
from docket.rubric.spec import (
    Detection,
    Example,
    Mode,
    Rubric,
    RubricMetadata,
)
from docket.self_test import run_self_test


class _StubProvider(ModelProvider):
    def __init__(self, responses_by_trace: dict[str, dict[str, Any]]) -> None:
        self.model = "stub:1"
        self._by_trace = responses_by_trace
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        for marker, response in self._by_trace.items():
            if marker in user:
                return response
        return {"positive": False}


def _rubric_with_examples() -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="self-test-fixture", version="1.0.0"),
        modes=[
            Mode(
                id="hallucination",
                severity="critical",
                detection=Detection(
                    type="llm_judge",
                    prompt="Detect hallucination.",
                    output_schema={
                        "type": "object",
                        "required": ["positive"],
                        "properties": {"positive": {"type": "boolean"}},
                    },
                ),
                examples=[
                    Example(trace_excerpt="POS_EXAMPLE", expected="positive"),
                    Example(trace_excerpt="NEG_EXAMPLE", expected="negative"),
                ],
            ),
            Mode(
                id="regex-mode",
                severity="low",
                detection=Detection(type="regex", pattern="x"),
                examples=[Example(trace_excerpt="x", expected="positive")],
            ),
        ],
    )


async def test_self_test_pass_when_provider_classifies_correctly() -> None:
    provider = _StubProvider(
        {"POS_EXAMPLE": {"positive": True}, "NEG_EXAMPLE": {"positive": False}}
    )
    results = await run_self_test(_rubric_with_examples(), provider)
    judge_results = [r for r in results if r.mode_id == "hallucination"]
    assert len(judge_results) == 2
    assert all(r.passed and not r.skipped for r in judge_results)


async def test_self_test_fails_when_provider_misclassifies() -> None:
    provider = _StubProvider({"POS_EXAMPLE": {"positive": False}})
    results = await run_self_test(_rubric_with_examples(), provider)
    pos_result = next(r for r in results if r.mode_id == "hallucination" and r.example_index == 0)
    assert not pos_result.passed
    assert "expected positive, got negative" in pos_result.message


async def test_self_test_exercises_regex_examples() -> None:
    """Regex modes are no longer skipped: each example runs through the
    regex detector without touching the LLM provider."""
    provider = _StubProvider({})
    results = await run_self_test(_rubric_with_examples(), provider)
    regex_result = next(r for r in results if r.mode_id == "regex-mode")
    assert not regex_result.skipped
    assert regex_result.passed
    assert regex_result.example_index == 0


async def test_self_test_regex_positive_and_negative_examples() -> None:
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="x", version="1"),
        modes=[
            Mode(
                id="timeout-regex",
                severity="low",
                detection=Detection(type="regex", pattern="(?i)request timed out"),
                examples=[
                    Example(trace_excerpt="ERROR: request timed out", expected="positive"),
                    Example(trace_excerpt="request completed in 12ms", expected="negative"),
                ],
            )
        ],
    )
    results = await run_self_test(rubric, _StubProvider({}))
    assert len(results) == 2
    assert all(r.passed and not r.skipped for r in results)


async def test_self_test_reports_failing_regex_example() -> None:
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="x", version="1"),
        modes=[
            Mode(
                id="timeout-regex",
                severity="low",
                detection=Detection(type="regex", pattern="(?i)request timed out"),
                examples=[
                    Example(trace_excerpt="request completed in 12ms", expected="positive"),
                ],
            )
        ],
    )
    results = await run_self_test(rubric, _StubProvider({}))
    assert len(results) == 1
    assert not results[0].passed
    assert not results[0].skipped
    assert "expected positive, got negative" in results[0].message


async def test_self_test_skips_tool_call_and_metric_threshold_with_reason() -> None:
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="x", version="1"),
        modes=[
            Mode(
                id="tool-mode",
                severity="low",
                detection=Detection(type="tool_call", tool_calls=["delete_record"]),
                examples=[Example(trace_excerpt="x", expected="positive")],
            ),
            Mode(
                id="metric-mode",
                severity="low",
                detection=Detection(
                    type="metric_threshold", metric="latency_ms", threshold=1.0, operator=">"
                ),
                examples=[Example(trace_excerpt="x", expected="positive")],
            ),
        ],
    )
    results = await run_self_test(rubric, _StubProvider({}))
    tool_result = next(r for r in results if r.mode_id == "tool-mode")
    metric_result = next(r for r in results if r.mode_id == "metric-mode")
    assert tool_result.skipped
    assert "structured tool-call records" in tool_result.message
    assert metric_result.skipped
    assert "trace metrics" in metric_result.message


async def test_self_test_modes_without_examples_are_ignored() -> None:
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="x", version="1"),
        modes=[
            Mode(
                id="no-examples",
                severity="low",
                detection=Detection(type="regex", pattern="x"),
            )
        ],
    )
    provider = _StubProvider({})
    results = await run_self_test(rubric, provider)
    assert results == []


class _BatchAwareStub(ModelProvider):
    """Judges examples by marker in single and batched calls; records traces per call.

    Examples containing `bad` get a schema-violating verdict in batched calls
    (and in single calls too when `bad_alone` is set).
    """

    def __init__(self, bad: str | None = None, *, bad_alone: bool = False) -> None:
        self.model = "stub:1"
        self.calls: list[int] = []
        self._bad = bad
        self._bad_alone = bad_alone

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        wrapper = schema.get("properties", {}).get("verdicts")
        if wrapper is None:
            self.calls.append(1)
            return self._verdict(user, batched=False)
        blocks = re.split(r"=== Trace \d+ ===\n", user)[1:]
        self.calls.append(len(blocks))
        return {"verdicts": [self._verdict(b, batched=True) for b in blocks]}

    def _verdict(self, text: str, *, batched: bool) -> dict[str, Any]:
        if self._bad is not None and self._bad in text and (batched or self._bad_alone):
            return {"positive": "maybe"}
        return {"positive": "POS_EXAMPLE" in text}


def _judge_rubric(n_examples: int) -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="batched-self-test", version="1.0.0"),
        modes=[
            Mode(
                id="judge",
                severity="high",
                detection=Detection(
                    type="llm_judge",
                    prompt="Detect the failure.",
                    output_schema={
                        "type": "object",
                        "required": ["positive"],
                        "properties": {"positive": {"type": "boolean"}},
                    },
                ),
                examples=[
                    Example(trace_excerpt=f"POS_EXAMPLE-{i}", expected="positive")
                    if i % 2 == 0
                    else Example(trace_excerpt=f"NEG_EXAMPLE-{i}", expected="negative")
                    for i in range(n_examples)
                ],
            )
        ],
    )


async def test_self_test_batches_llm_judge_examples() -> None:
    provider = _BatchAwareStub()
    results = await run_self_test(_judge_rubric(5), provider, batch_size=2)
    assert provider.calls == [2, 2, 1]
    assert [r.example_index for r in results] == [0, 1, 2, 3, 4]
    assert all(r.passed and not r.skipped for r in results)
    assert all("judged alone" not in r.message for r in results)


async def test_self_test_rejudges_unusable_batched_verdict_alone() -> None:
    provider = _BatchAwareStub(bad="NEG_EXAMPLE-1")
    results = await run_self_test(_judge_rubric(3), provider, batch_size=8)
    assert provider.calls == [3, 1]
    assert all(r.passed for r in results)
    assert "judged alone after its batched verdict failed" in results[1].message
    assert "violates `output_schema`" in results[1].message
    assert "judged alone" not in results[0].message + results[2].message


async def test_self_test_reports_error_when_solo_attempt_also_fails() -> None:
    provider = _BatchAwareStub(bad="NEG_EXAMPLE-1", bad_alone=True)
    results = await run_self_test(_judge_rubric(3), provider, batch_size=8)
    assert [r.passed for r in results] == [True, False, True]
    assert results[1].message.startswith("error: ")
    assert "judged alone after its batched verdict failed" in results[1].message


async def test_self_test_batch_size_one_makes_one_call_per_example() -> None:
    provider = _BatchAwareStub(bad="NEG_EXAMPLE-1")
    results = await run_self_test(_judge_rubric(3), provider, batch_size=1)
    assert provider.calls == [1, 1, 1]
    # Unbatched, `bad` only breaks batched verdicts, so every example passes.
    assert all(r.passed for r in results)


async def test_self_test_regex_examples_ignore_batch_size() -> None:
    provider = _StubProvider(
        {"POS_EXAMPLE": {"positive": True}, "NEG_EXAMPLE": {"positive": False}}
    )
    results = await run_self_test(_rubric_with_examples(), provider, batch_size=8)
    regex_result = next(r for r in results if r.mode_id == "regex-mode")
    assert regex_result.passed
    assert not regex_result.skipped

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

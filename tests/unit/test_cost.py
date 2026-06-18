"""Unit tests for `docket.cost`."""

import pytest

from docket.cost import (
    DEFAULT_INPUT_TOKENS_PER_CALL,
    DEFAULT_OUTPUT_TOKENS_PER_CALL,
    estimate_cost,
    known_models,
    llm_judge_modes,
)
from docket.rubric.spec import Detection, Mode, Rubric, RubricMetadata

_JUDGE_SCHEMA = {"type": "object", "properties": {"positive": {"type": "boolean"}}}


def _judge_mode(mode_id: str, *, model: str | None = None) -> Mode:
    return Mode(
        id=mode_id,
        severity="medium",
        detection=Detection(
            type="llm_judge",
            prompt="Does the agent hallucinate?",
            output_schema=_JUDGE_SCHEMA,
            model=model,
        ),
    )


def _regex_mode(mode_id: str) -> Mode:
    return Mode(
        id=mode_id,
        severity="low",
        detection=Detection(type="regex", pattern="hello"),
    )


def _composite_judge_mode(mode_id: str, *, model: str | None = None) -> Mode:
    return Mode(
        id=mode_id,
        severity="high",
        detection=Detection(
            type="composite",
            operator="and",
            operands=[
                Detection(type="tool_call", tool_calls=["handoff"]),
                Detection(
                    type="llm_judge",
                    prompt="Was the handoff context complete?",
                    output_schema=_JUDGE_SCHEMA,
                    model=model,
                ),
            ],
        ),
    )


def _rubric(modes: list[Mode]) -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="testbench", version="0.1.0"),
        modes=modes,
    )


def test_estimate_with_known_haiku_model() -> None:
    est = estimate_cost(
        trace_count=100,
        rubric=_rubric([_judge_mode(f"judge-{i}") for i in range(5)]),
        model="claude-haiku-4-5-20251001",
    )
    assert est.total_calls == 500
    assert est.total_input_tokens == 500 * DEFAULT_INPUT_TOKENS_PER_CALL
    assert est.total_output_tokens == 500 * DEFAULT_OUTPUT_TOKENS_PER_CALL
    # Haiku: $1/M input + $5/M output.
    # 500 calls × 1500 in = 750k in → $0.75
    # 500 calls × 200 out = 100k out → $0.50
    # Total: $1.25
    assert est.estimated_usd == pytest.approx(1.25, rel=0.01)


def test_estimate_counts_only_llm_judge_modes() -> None:
    rubric = _rubric(
        [
            _judge_mode("judge-a"),
            _regex_mode("regex-a"),
            Mode(
                id="metric-a",
                severity="low",
                detection=Detection(type="metric_threshold", metric="span_count", threshold=3),
            ),
            _composite_judge_mode("composite-judge"),
            Mode(
                id="composite-deterministic",
                severity="low",
                detection=Detection(
                    type="composite",
                    operator="or",
                    operands=[
                        Detection(type="regex", pattern="x"),
                        Detection(type="tool_call", tool_calls=["t"]),
                    ],
                ),
            ),
        ]
    )
    assert [m.id for m in llm_judge_modes(rubric)] == ["judge-a", "composite-judge"]
    est = estimate_cost(trace_count=10, rubric=rubric, model="gpt-4o-mini")
    assert est.mode_count == 5
    assert est.judge_mode_count == 2
    assert est.total_calls == 20  # 2 judge modes × 10 traces, batch=1


def test_deterministic_only_rubric_estimates_zero() -> None:
    rubric = _rubric([_regex_mode("regex-a"), _regex_mode("regex-b")])
    est = estimate_cost(trace_count=1000, rubric=rubric, model="gpt-4o-mini")
    assert est.judge_mode_count == 0
    assert est.total_calls == 0
    assert est.estimated_usd == 0.0


def test_deterministic_only_rubric_ignores_unknown_default_model() -> None:
    # No judge mode ever needs the default model's rate, so an unknown
    # default model is not an error for a $0 run.
    rubric = _rubric([_regex_mode("regex-a")])
    est = estimate_cost(trace_count=10, rubric=rubric, model="claude-future-edition")
    assert est.estimated_usd == 0.0


def test_batching_divides_call_count() -> None:
    rubric = _rubric([_judge_mode("judge-a")])
    unbatched = estimate_cost(trace_count=100, rubric=rubric, model="gpt-4o-mini")
    batched = estimate_cost(trace_count=100, rubric=rubric, model="gpt-4o-mini", batch_size=8)
    assert unbatched.total_calls == 100
    assert batched.total_calls == 13  # ceil(100 / 8)
    assert batched.estimated_usd == pytest.approx(unbatched.estimated_usd * 13 / 100, rel=1e-6)


def test_per_mode_model_override_changes_estimate() -> None:
    base = estimate_cost(
        trace_count=100,
        rubric=_rubric([_judge_mode("judge-a")]),
        model="claude-haiku-4-5-20251001",
    )
    overridden = estimate_cost(
        trace_count=100,
        rubric=_rubric([_judge_mode("judge-a", model="claude-opus-4-7")]),
        model="claude-haiku-4-5-20251001",
    )
    # Opus is 15× Haiku on both input and output rates.
    assert overridden.estimated_usd == pytest.approx(15 * base.estimated_usd, rel=1e-6)


def test_per_mode_override_in_composite_priced_with_own_model() -> None:
    base = estimate_cost(
        trace_count=10,
        rubric=_rubric([_composite_judge_mode("c")]),
        model="claude-haiku-4-5-20251001",
    )
    overridden = estimate_cost(
        trace_count=10,
        rubric=_rubric([_composite_judge_mode("c", model="claude-sonnet-4-6")]),
        model="claude-haiku-4-5-20251001",
    )
    assert overridden.estimated_usd > base.estimated_usd


def test_unknown_per_mode_override_falls_back_to_default_model() -> None:
    base = estimate_cost(
        trace_count=10,
        rubric=_rubric([_judge_mode("judge-a")]),
        model="claude-haiku-4-5-20251001",
    )
    fallback = estimate_cost(
        trace_count=10,
        rubric=_rubric([_judge_mode("judge-a", model="some-unpriced-model")]),
        model="claude-haiku-4-5-20251001",
    )
    assert fallback.estimated_usd == pytest.approx(base.estimated_usd, rel=1e-9)


def test_estimate_unknown_model_raises_with_hint() -> None:
    rubric = _rubric([_judge_mode("judge-a")])
    with pytest.raises(ValueError, match="no pricing entry"):
        estimate_cost(trace_count=10, rubric=rubric, model="claude-future-edition")


def test_estimate_render_includes_key_numbers_and_variance() -> None:
    est = estimate_cost(
        trace_count=42,
        rubric=_rubric([_judge_mode(f"j-{i}") for i in range(3)]),
        model="gpt-4o-mini",
    )
    rendered = est.render()
    assert "42 traces × 3 modes" in rendered
    assert "126 LLM calls" in rendered
    assert "gpt-4o-mini" in rendered
    assert "USD" in rendered
    assert "±100%" in rendered


def test_known_models_includes_defaults() -> None:
    models = known_models()
    assert "claude-haiku-4-5-20251001" in models
    assert "gpt-4o-mini" in models


def test_estimate_with_custom_token_shape() -> None:
    est = estimate_cost(
        trace_count=10,
        rubric=_rubric([_judge_mode("judge-a")]),
        model="gpt-4o-mini",
        input_tokens_per_call=10_000,
        output_tokens_per_call=500,
    )
    # gpt-4o-mini: $0.15/M in + $0.60/M out
    # 10 calls × 10k in = 100k → $0.015
    # 10 calls × 500 out = 5k → $0.003
    assert est.estimated_usd == pytest.approx(0.018, rel=0.01)

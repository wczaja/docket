"""Cost estimation for `docket run --dry-run` and the per-run gate.

Only modes whose detection involves an LLM judge cost money: `llm_judge`
itself, or a `composite` containing one (recursively). Deterministic
detectors (regex, tool_call, metric_threshold) are free. Calls are counted
the way `Classifier.classify_all` makes them:

    llm_judge mode:  ceil(trace_count / batch_size) calls
    composite mode:  trace_count calls per `llm_judge` operand (composites
                     are never batched; an upper bound, since a
                     deterministic operand that settles the verdict first
                     skips the judge)

Tokens and dollars are priced per trace judged, not per call: a batched
call still sends every trace's text and returns every trace's verdict, so
`--batch` lowers the call count but not the estimate. The estimate leaves
out two effects that pull in opposite directions: the system prompt, judge
instructions and schema that a batched call sends once instead of per trace
(a real saving, so batched estimates err high), and the extra calls from
retries or from re-judging a batched verdict alone after it fails
validation.

Each judge is priced at its own `model:` override (a `provider:model` URI,
looked up by its model part) when that model is in the price table,
falling back to the run's default model otherwise.

Token shape per trace judged, i.e. per unbatched call (measured
empirically against the 60-trace fixture):
- Input: ~1500 tokens (system + serialized TraceLike + rubric metadata
  for one mode). Larger traces drift up; this is a reasonable mean.
- Output: ~200 tokens (structured verdict with evidence excerpt).

Honest variance (design Phase 11): trace sizes vary by ~30× across
deployment shapes, so the estimate carries a ±100% disclaimer — it is a
wide estimate, not a tight one.

Price table is in USD per 1M tokens. Updating it is a single-file edit;
keep it in sync with provider price pages. Overridable in config under
`pricing.<provider>.<model>` (not implemented in v1.1 — the baked table
is the single source of truth until users complain).
"""

import math
from dataclasses import dataclass
from typing import Final

from docket.rubric.spec import Detection, Mode, Rubric

DEFAULT_INPUT_TOKENS_PER_CALL: Final[int] = 1500
DEFAULT_OUTPUT_TOKENS_PER_CALL: Final[int] = 200


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens, input and output."""

    input_per_million: float
    output_per_million: float


# Source: provider pricing pages, current as of v1.1 release.
# Anthropic: https://www.anthropic.com/pricing
# OpenAI:    https://openai.com/api/pricing/
_PRICE_TABLE: dict[str, ModelPricing] = {
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-opus-4-7": ModelPricing(15.0, 75.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o": ModelPricing(2.50, 10.00),
}


def _judge_detections(detection: Detection) -> list[Detection]:
    """The `llm_judge` detections in `detection`, recursing into composites."""
    if detection.type == "llm_judge":
        return [detection]
    if detection.type == "composite" and detection.operands:
        return [judge for op in detection.operands for judge in _judge_detections(op)]
    return []


def _model_name(model_uri: str) -> str:
    """The price-table key for a `model:` override: `anthropic:claude-x` -> `claude-x`."""
    return model_uri.split(":", 1)[-1]


def llm_judge_modes(rubric: Rubric) -> list[Mode]:
    """Modes whose detection involves an LLM judge (directly or via composite)."""
    return [m for m in rubric.modes if _judge_detections(m.detection)]


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of an `docket run` invocation.

    `total_calls` is what the classifier will issue; the token fields and
    `estimated_usd` count one `*_tokens_per_call` per trace judged, which is
    one call when unbatched.
    """

    trace_count: int
    mode_count: int
    judge_mode_count: int
    batch_size: int
    model: str
    input_tokens_per_call: int
    output_tokens_per_call: int
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_usd: float

    def render(self) -> str:
        batch_note = (
            "(Batching cuts calls, not this estimate: a batched call is still "
            "billed for every trace's text and verdict.)\n"
            if self.batch_size > 1
            else ""
        )
        return (
            f"Would classify {self.trace_count} traces × {self.mode_count} modes "
            f"({self.judge_mode_count} LLM-judge mode(s), batch={self.batch_size}) "
            f"= {self.total_calls} LLM calls\n"
            f"{batch_note}"
            f"Default model: {self.model}\n"
            f"Estimated tokens: {self.total_input_tokens:,} input + "
            f"{self.total_output_tokens:,} output\n"
            f"Estimated cost: ${self.estimated_usd:.4f} USD (±100%)\n"
            "(Variance: trace sizes vary ~30× across deployment shapes, so "
            "actual cost may differ by up to ±100% from this estimate.)"
        )


def estimate_cost(
    *,
    trace_count: int,
    rubric: Rubric,
    model: str,
    batch_size: int = 1,
    input_tokens_per_call: int = DEFAULT_INPUT_TOKENS_PER_CALL,
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_CALL,
) -> CostEstimate:
    """Estimate the dollar cost of a run before it executes.

    Only LLM-judge-involving modes are priced (deterministic detectors are
    free). Calls are counted as the classifier makes them: ceil(trace_count
    / batch_size) per `llm_judge` mode and trace_count per `llm_judge`
    operand of a composite. Tokens and dollars are counted per trace judged,
    so `batch_size` changes `total_calls` but not `estimated_usd`. Per-judge
    `model:` overrides use their own rates when present in the price table
    and fall back to the default `model`'s rates otherwise.

    Raises `ValueError` if `batch_size` < 1 or a needed model isn't in the
    price table; callers should surface this as a config error pointing the
    user at `pricing.<provider>.<model>` overrides (v1.2).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
    judge_modes = llm_judge_modes(rubric)
    total_calls = 0
    judgments = 0
    cost_usd = 0.0
    default_pricing: ModelPricing | None = None
    for mode in judge_modes:
        judges = _judge_detections(mode.detection)
        # Same rule as Classifier._batches: only plain llm_judge modes batch.
        if mode.detection.type == "llm_judge":
            total_calls += math.ceil(trace_count / batch_size)
        else:
            total_calls += trace_count * len(judges)
        for judge in judges:
            pricing = (
                _PRICE_TABLE.get(_model_name(judge.model)) if judge.model is not None else None
            )
            if pricing is None:
                if default_pricing is None:
                    default_pricing = _PRICE_TABLE.get(model)
                    if default_pricing is None:
                        known = ", ".join(sorted(_PRICE_TABLE.keys()))
                        raise ValueError(
                            f"no pricing entry for model {model!r}; known models: {known}. "
                            f"Update docket.cost._PRICE_TABLE or pass --model to a "
                            f"known one."
                        )
                pricing = default_pricing
            judgments += trace_count
            cost_usd += trace_count * (
                input_tokens_per_call / 1_000_000 * pricing.input_per_million
                + output_tokens_per_call / 1_000_000 * pricing.output_per_million
            )
    return CostEstimate(
        trace_count=trace_count,
        mode_count=len(rubric.modes),
        judge_mode_count=len(judge_modes),
        batch_size=batch_size,
        model=model,
        input_tokens_per_call=input_tokens_per_call,
        output_tokens_per_call=output_tokens_per_call,
        total_calls=total_calls,
        total_input_tokens=judgments * input_tokens_per_call,
        total_output_tokens=judgments * output_tokens_per_call,
        estimated_usd=cost_usd,
    )


def known_models() -> list[str]:
    return sorted(_PRICE_TABLE.keys())

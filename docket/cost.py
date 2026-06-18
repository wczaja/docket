"""Cost estimation for `docket run --dry-run` and the per-run gate.

Only modes whose detection involves an LLM judge cost money: `llm_judge`
itself, or a `composite` containing one (recursively). Deterministic
detectors (regex, tool_call, metric_threshold) are free. The classifier
batches traces per judge mode, so:

    LLM calls = ceil(trace_count / batch_size) per judge mode

Each judge mode is priced at its own `model:` override when that model is
in the price table, falling back to the run's default model otherwise.

Token shape (measured empirically against the 60-trace fixture):
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


def _detection_uses_llm_judge(detection: Detection) -> bool:
    if detection.type == "llm_judge":
        return True
    if detection.type == "composite" and detection.operands:
        return any(_detection_uses_llm_judge(op) for op in detection.operands)
    return False


def _judge_model_override(detection: Detection) -> str | None:
    """First explicit `model:` on an llm_judge detection, recursing composites."""
    if detection.type == "llm_judge":
        return detection.model
    if detection.type == "composite" and detection.operands:
        for op in detection.operands:
            override = _judge_model_override(op)
            if override is not None:
                return override
    return None


def llm_judge_modes(rubric: Rubric) -> list[Mode]:
    """Modes whose detection involves an LLM judge (directly or via composite)."""
    return [m for m in rubric.modes if _detection_uses_llm_judge(m.detection)]


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost of an `docket run` invocation."""

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
        return (
            f"Would classify {self.trace_count} traces × {self.mode_count} modes "
            f"({self.judge_mode_count} LLM-judge mode(s), batch={self.batch_size}) "
            f"= {self.total_calls} LLM calls\n"
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
    free), each at ceil(trace_count / batch_size) calls. Per-mode `model:`
    overrides use their own rates when present in the price table and fall
    back to the default `model`'s rates otherwise.

    Raises `ValueError` if a needed model isn't in the price table; callers
    should surface this as a config error pointing the user at
    `pricing.<provider>.<model>` overrides (v1.2).
    """
    judge_modes = llm_judge_modes(rubric)
    calls_per_mode = math.ceil(trace_count / batch_size) if trace_count else 0
    total_calls = 0
    cost_usd = 0.0
    default_pricing: ModelPricing | None = None
    for mode in judge_modes:
        override = _judge_model_override(mode.detection)
        pricing = _PRICE_TABLE.get(override) if override is not None else None
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
        total_calls += calls_per_mode
        cost_usd += calls_per_mode * (
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
        total_input_tokens=total_calls * input_tokens_per_call,
        total_output_tokens=total_calls * output_tokens_per_call,
        estimated_usd=cost_usd,
    )


def known_models() -> list[str]:
    return sorted(_PRICE_TABLE.keys())

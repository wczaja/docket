"""Self-test runner: exercise rubric examples against detectors.

Exercises `llm_judge` and `regex` modes: each example's `trace_excerpt` is
wrapped in a `TraceLike` and run through the mode's detector, and the verdict
is checked against `expected`. `tool_call` and `metric_threshold` (and
`composite`) modes are reported as skipped with an explicit reason — they
need structured traces (tool-call records, metrics) that the `examples:`
shape cannot express.

With `batch_size > 1`, each `llm_judge` mode's examples are judged
`batch_size` per provider call. An example whose batched verdict is
unusable is judged again on its own, the one attempt an unbatched run makes,
and its message says so.
"""

from dataclasses import dataclass

from docket.detectors import get_detector
from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.trace import TraceLike, Verdict
from docket.rubric.spec import Example, Mode, Rubric

_SKIP_REASONS: dict[str, str] = {
    "tool_call": (
        "skipped: tool_call examples need structured tool-call records, "
        "which `examples:` cannot express"
    ),
    "metric_threshold": (
        "skipped: metric_threshold examples need trace metrics, which `examples:` cannot express"
    ),
    "composite": (
        "skipped: composite operands may need structured traces, which `examples:` cannot express"
    ),
}


@dataclass(frozen=True)
class SelfTestResult:
    mode_id: str
    example_index: int
    passed: bool
    skipped: bool
    message: str


async def run_self_test(
    rubric: Rubric,
    default_provider: ModelProvider,
    batch_size: int = 1,
) -> list[SelfTestResult]:
    """Run each mode's examples through the appropriate detector.

    Returns one `SelfTestResult` per example (or one with `skipped=True` per
    non-exercisable mode that has examples).
    """
    results: list[SelfTestResult] = []
    for mode in rubric.modes:
        if not mode.examples:
            continue
        detection_type = mode.detection.type
        if detection_type == "llm_judge":
            detector: Detector = get_detector(
                "llm_judge", llm_provider=default_provider, batch_size=batch_size
            )
            examples_per_call = batch_size
        elif detection_type == "regex":
            detector = get_detector("regex")
            examples_per_call = 1
        else:
            results.append(
                SelfTestResult(
                    mode_id=mode.id,
                    example_index=-1,
                    passed=True,
                    skipped=True,
                    message=_SKIP_REASONS.get(
                        detection_type,
                        f"skipped: self-test does not exercise {detection_type} modes",
                    ),
                )
            )
            continue
        results.extend(await _run_mode_examples(mode, detector, examples_per_call))
    return results


async def _run_mode_examples(
    mode: Mode,
    detector: Detector,
    examples_per_call: int,
) -> list[SelfTestResult]:
    results: list[SelfTestResult] = []
    examples = list(enumerate(mode.examples))
    for start in range(0, len(examples), examples_per_call):
        chunk = examples[start : start + examples_per_call]
        traces = [TraceLike(full_text=ex.trace_excerpt, context=ex.context) for _, ex in chunk]
        outcomes = await detector.evaluate_batch_isolated(mode, traces)
        for (i, example), trace, outcome in zip(chunk, traces, outcomes, strict=True):
            note = ""
            if isinstance(outcome, DetectionError) and len(chunk) > 1:
                note = f" (judged alone after its batched verdict failed: {outcome})"
                (outcome,) = await detector.evaluate_batch_isolated(mode, [trace])
            results.append(_example_result(mode, i, example, outcome, note))
    return results


def _example_result(
    mode: Mode,
    index: int,
    example: Example,
    outcome: Verdict | DetectionError,
    note: str,
) -> SelfTestResult:
    if isinstance(outcome, DetectionError):
        return SelfTestResult(
            mode_id=mode.id,
            example_index=index,
            passed=False,
            skipped=False,
            message=f"error: {outcome}{note}",
        )
    expected_positive = example.expected == "positive"
    return SelfTestResult(
        mode_id=mode.id,
        example_index=index,
        passed=outcome.positive == expected_positive,
        skipped=False,
        message=(
            f"expected {example.expected}, "
            f"got {'positive' if outcome.positive else 'negative'}{note}"
        ),
    )

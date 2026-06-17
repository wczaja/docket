"""Self-test runner: exercise rubric examples against detectors.

Exercises `llm_judge` and `regex` modes: each example's `trace_excerpt` is
wrapped in a `TraceLike` and run through the mode's detector, and the verdict
is checked against `expected`. `tool_call` and `metric_threshold` (and
`composite`) modes are reported as skipped with an explicit reason — they
need structured traces (tool-call records, metrics) that the `examples:`
shape cannot express.
"""

from dataclasses import dataclass

from agent_triage.detectors import get_detector
from agent_triage.detectors.base import Detector
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import TraceLike
from agent_triage.rubric.spec import Mode, Rubric

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
        elif detection_type == "regex":
            detector = get_detector("regex")
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
        results.extend(await _run_mode_examples(mode, detector))
    return results


async def _run_mode_examples(mode: Mode, detector: Detector) -> list[SelfTestResult]:
    results: list[SelfTestResult] = []
    for i, example in enumerate(mode.examples):
        trace = TraceLike(full_text=example.trace_excerpt, context=example.context)
        try:
            verdict = await detector.evaluate(mode, trace)
        except DetectionError as e:
            results.append(
                SelfTestResult(
                    mode_id=mode.id,
                    example_index=i,
                    passed=False,
                    skipped=False,
                    message=f"error: {e}",
                )
            )
            continue
        expected_positive = example.expected == "positive"
        passed = verdict.positive == expected_positive
        results.append(
            SelfTestResult(
                mode_id=mode.id,
                example_index=i,
                passed=passed,
                skipped=False,
                message=(
                    f"expected {example.expected}, "
                    f"got {'positive' if verdict.positive else 'negative'}"
                ),
            )
        )
    return results

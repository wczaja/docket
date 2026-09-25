"""Composite detector: AND/OR over sub-detector verdicts.

Sub-detectors are resolved through a factory injected at construction time so
the composite stays decoupled from the registry and can be exercised with
mocks in tests.

Operands are evaluated sequentially with short-circuiting: AND stops at the
first negative, OR stops at the first positive. Cheap deterministic operands
(regex, tool_call, metric_threshold, and composites built only from those)
are evaluated before operands that involve an LLM judge — stable among equals
— so a cheap gate can spare the expensive judge call. The boolean verdict is
identical to evaluating every operand; `sub_verdicts` evidence covers the
operands actually evaluated.

Clustering and annotations read `confidence`, `excerpt` and `match` from the
top level of `extra`, so a positive composite lifts them from the operands that
fired: `confidence` is the lowest one reported (an AND is no surer than its
least sure operand; an OR fires on a single operand), and `excerpt` (with its
`excerpt_verified` flag) and `match` come from the first firing operand, in
evaluation order, that has one. A nested composite has already lifted its own
operands' keys, so this recurses.
"""

from collections.abc import Callable
from typing import Any

from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.models.trace import TraceLike, Verdict
from docket.rubric.spec import Detection, Mode


def _involves_llm_judge(detection: Detection) -> bool:
    if detection.type == "llm_judge":
        return True
    return any(_involves_llm_judge(op) for op in detection.operands or [])


class CompositeDetector(Detector):
    def __init__(self, get_sub_detector: Callable[[str], Detector]) -> None:
        self._get_sub_detector = get_sub_detector

    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        d = mode.detection
        if not d.operands:
            raise DetectionError(f"Mode {mode.id!r}: composite requires `operands`")
        if d.operator not in ("and", "or"):
            raise DetectionError(
                f"Mode {mode.id!r}: composite operator must be 'and' or 'or' (got {d.operator!r})"
            )
        # Deterministic (non-LLM) operands first; stable sort preserves the
        # rubric's order among equals.
        ordered = sorted(d.operands, key=_involves_llm_judge)
        sub_verdicts: list[Verdict] = []
        for sub_detection in ordered:
            sub_mode = mode.model_copy(update={"detection": sub_detection})
            sub_detector = self._get_sub_detector(sub_detection.type)
            verdict = await sub_detector.evaluate(sub_mode, trace)
            sub_verdicts.append(verdict)
            if d.operator == "and" and not verdict.positive:
                break
            if d.operator == "or" and verdict.positive:
                break
        if d.operator == "and":
            positive = all(v.positive for v in sub_verdicts)
        else:
            positive = any(v.positive for v in sub_verdicts)
        extra: dict[str, Any] = {
            "operator": d.operator,
            "sub_verdicts": [v.model_dump() for v in sub_verdicts],
        }
        if positive:
            extra.update(_lift_evidence(sub_verdicts))
        return Verdict(positive=positive, extra=extra)


def _lift_evidence(sub_verdicts: list[Verdict]) -> dict[str, Any]:
    """The firing operands' confidence and evidence, keyed as on a leaf verdict."""
    fired = [v.extra for v in sub_verdicts if v.positive]
    lifted: dict[str, Any] = {}
    confidences = [
        float(c)
        for c in (e.get("confidence") for e in fired)
        if isinstance(c, (int, float)) and not isinstance(c, bool)
    ]
    if confidences:
        lifted["confidence"] = min(confidences)
    for e in fired:
        if _is_text(e.get("excerpt")):
            lifted["excerpt"] = e["excerpt"]
            # The flag describes this excerpt only, so it travels with it.
            if isinstance(e.get("excerpt_verified"), bool):
                lifted["excerpt_verified"] = e["excerpt_verified"]
            break
    for e in fired:
        if _is_text(e.get("match")):
            lifted["match"] = e["match"]
            break
    return lifted


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())

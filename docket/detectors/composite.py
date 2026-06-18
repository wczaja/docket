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
"""

from collections.abc import Callable

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
        return Verdict(
            positive=positive,
            extra={"operator": d.operator, "sub_verdicts": [v.model_dump() for v in sub_verdicts]},
        )

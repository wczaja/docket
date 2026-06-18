"""Metric-threshold detector: compares a numeric trace metric against a constant."""

import operator
from collections.abc import Callable
from typing import Final

from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.models.trace import TraceLike, Verdict
from docket.rubric.spec import Mode

_COMPARATORS: Final[dict[str, Callable[[float, float], bool]]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class MetricThresholdDetector(Detector):
    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        d = mode.detection
        if d.metric is None or d.threshold is None or d.operator is None:
            raise DetectionError(
                f"Mode {mode.id!r}: metric_threshold requires metric, threshold, operator"
            )
        if d.operator not in _COMPARATORS:
            raise DetectionError(
                f"Mode {mode.id!r}: invalid metric_threshold operator {d.operator!r}; "
                f"must be one of {sorted(_COMPARATORS)}"
            )
        if d.metric not in trace.metrics:
            raise DetectionError(f"Mode {mode.id!r}: metric {d.metric!r} not present on trace")
        value = trace.metrics[d.metric]
        positive = _COMPARATORS[d.operator](value, d.threshold)
        return Verdict(
            positive=positive,
            extra={"value": value, "threshold": d.threshold, "operator": d.operator},
        )

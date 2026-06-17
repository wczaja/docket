"""Regex detector: positive iff the rubric's pattern matches the trace text."""

import re
from functools import lru_cache

from agent_triage.detectors.base import Detector
from agent_triage.errors import DetectionError
from agent_triage.models.trace import TraceLike, Verdict
from agent_triage.rubric.spec import Mode


@lru_cache(maxsize=256)
def _compile(pattern_str: str) -> re.Pattern[str]:
    # A run evaluates the same rubric pattern against every trace; compiling
    # once per pattern instead of once per (mode, trace) pair matters at
    # production trace volumes.
    return re.compile(pattern_str)


class RegexDetector(Detector):
    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        pattern_str = mode.detection.pattern
        if pattern_str is None:
            raise DetectionError(f"Mode {mode.id!r}: regex detection requires `pattern`")
        try:
            pattern = _compile(pattern_str)
        except re.error as e:
            raise DetectionError(
                f"Mode {mode.id!r}: invalid regex pattern {pattern_str!r}: {e}"
            ) from e
        text = trace.full_text or trace.final_response
        match = pattern.search(text)
        return Verdict(
            positive=match is not None,
            extra={"match": match.group(0) if match else None},
        )

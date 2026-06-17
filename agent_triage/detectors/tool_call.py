"""Tool-call detector: positive iff any of the rubric's tool names was called.

v1 semantic is `any-of`; sequence and absence variants are out of scope for
Phase 2. The DSL can be extended later with a `mode: any|all|sequence` knob
without breaking existing rubrics (default stays `any`).
"""

from agent_triage.detectors.base import Detector
from agent_triage.errors import DetectionError
from agent_triage.models.trace import TraceLike, Verdict
from agent_triage.rubric.spec import Mode


class ToolCallDetector(Detector):
    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        wanted = mode.detection.tool_calls
        if not wanted:
            raise DetectionError(f"Mode {mode.id!r}: tool_call detection requires `tool_calls`")
        called = {tc.name for tc in trace.tool_calls}
        intersection = sorted(set(wanted) & called)
        return Verdict(
            positive=bool(intersection),
            extra={"matched_tools": intersection},
        )

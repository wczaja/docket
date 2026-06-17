"""Detector registry / factory.

`get_detector(detection_type, llm_provider=None)` returns the appropriate
detector. `llm_judge` requires a `ModelProvider`; composite uses the same
factory recursively for its operands.
"""

from agent_triage.detectors.base import Detector
from agent_triage.detectors.composite import CompositeDetector
from agent_triage.detectors.llm_judge import LLMJudgeDetector
from agent_triage.detectors.metric_threshold import MetricThresholdDetector
from agent_triage.detectors.regex import RegexDetector
from agent_triage.detectors.tool_call import ToolCallDetector
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider


def get_detector(
    detection_type: str,
    llm_provider: ModelProvider | None = None,
    batch_size: int = 1,
) -> Detector:
    if detection_type == "regex":
        return RegexDetector()
    if detection_type == "tool_call":
        return ToolCallDetector()
    if detection_type == "metric_threshold":
        return MetricThresholdDetector()
    if detection_type == "composite":
        return CompositeDetector(
            lambda t: get_detector(t, llm_provider=llm_provider, batch_size=batch_size)
        )
    if detection_type == "llm_judge":
        if llm_provider is None:
            raise DetectionError("llm_judge detector requires a ModelProvider; pass `llm_provider`")
        return LLMJudgeDetector(llm_provider, batch_size=batch_size)
    raise DetectionError(f"Unknown detection type: {detection_type!r}")


__all__ = [
    "CompositeDetector",
    "Detector",
    "LLMJudgeDetector",
    "MetricThresholdDetector",
    "RegexDetector",
    "ToolCallDetector",
    "get_detector",
]

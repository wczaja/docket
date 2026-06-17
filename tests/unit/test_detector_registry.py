import pytest

from agent_triage.detectors import get_detector
from agent_triage.detectors.composite import CompositeDetector
from agent_triage.detectors.metric_threshold import MetricThresholdDetector
from agent_triage.detectors.regex import RegexDetector
from agent_triage.detectors.tool_call import ToolCallDetector
from agent_triage.errors import DetectionError


def test_get_detector_returns_correct_types() -> None:
    assert isinstance(get_detector("regex"), RegexDetector)
    assert isinstance(get_detector("tool_call"), ToolCallDetector)
    assert isinstance(get_detector("metric_threshold"), MetricThresholdDetector)
    assert isinstance(get_detector("composite"), CompositeDetector)


def test_get_detector_unknown_type_raises() -> None:
    with pytest.raises(DetectionError, match="Unknown detection type"):
        get_detector("not-a-real-type")


def test_get_detector_llm_judge_without_provider_raises() -> None:
    with pytest.raises(DetectionError, match="requires a ModelProvider"):
        get_detector("llm_judge")


def test_get_detector_llm_judge_with_provider() -> None:
    from agent_triage.detectors.llm_judge import LLMJudgeDetector
    from agent_triage.llm.base import ModelProvider

    class _DummyProvider(ModelProvider):
        model = "dummy:1"

        async def structured_complete(  # noqa: D401
            self,
            system: str,
            user: str,
            schema: dict[str, object],
        ) -> dict[str, object]:
            return {}

    detector = get_detector("llm_judge", llm_provider=_DummyProvider())
    assert isinstance(detector, LLMJudgeDetector)

import pytest

from agent_triage.detectors.regex import RegexDetector
from agent_triage.errors import DetectionError
from agent_triage.models.trace import TraceLike
from agent_triage.rubric.spec import Detection, Mode


def _mode(**detection_kwargs: object) -> Mode:
    return Mode(
        id="m",
        severity="low",
        detection=Detection(type="regex", **detection_kwargs),  # type: ignore[arg-type]
    )


async def test_regex_positive_match() -> None:
    verdict = await RegexDetector().evaluate(
        _mode(pattern="hello"),
        TraceLike(full_text="please say hello world"),
    )
    assert verdict.positive
    assert verdict.extra["match"] == "hello"


async def test_regex_negative_match() -> None:
    verdict = await RegexDetector().evaluate(
        _mode(pattern="hello"),
        TraceLike(full_text="goodbye"),
    )
    assert not verdict.positive
    assert verdict.extra["match"] is None


async def test_regex_uses_final_response_when_full_text_empty() -> None:
    verdict = await RegexDetector().evaluate(
        _mode(pattern="bye"),
        TraceLike(full_text="", final_response="goodbye"),
    )
    assert verdict.positive


async def test_regex_missing_pattern_raises() -> None:
    with pytest.raises(DetectionError, match="requires `pattern`"):
        await RegexDetector().evaluate(_mode(), TraceLike(full_text="x"))


async def test_regex_invalid_pattern_raises() -> None:
    with pytest.raises(DetectionError, match="invalid regex"):
        await RegexDetector().evaluate(_mode(pattern="[invalid"), TraceLike(full_text="x"))


async def test_regex_batch_delegates_to_evaluate() -> None:
    detector = RegexDetector()
    mode = _mode(pattern="hi")
    verdicts = await detector.evaluate_batch(
        mode, [TraceLike(full_text="hi"), TraceLike(full_text="bye")]
    )
    assert [v.positive for v in verdicts] == [True, False]

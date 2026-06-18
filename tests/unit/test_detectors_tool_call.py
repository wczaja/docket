import pytest

from docket.detectors.tool_call import ToolCallDetector
from docket.errors import DetectionError
from docket.models.trace import ToolCall, TraceLike
from docket.rubric.spec import Detection, Mode


def _mode(tool_calls: list[str] | None = None) -> Mode:
    detection_kwargs: dict[str, object] = {"type": "tool_call"}
    if tool_calls is not None:
        detection_kwargs["tool_calls"] = tool_calls
    return Mode(id="m", severity="high", detection=Detection(**detection_kwargs))  # type: ignore[arg-type]


async def test_tool_call_positive() -> None:
    verdict = await ToolCallDetector().evaluate(
        _mode(tool_calls=["delete", "drop"]),
        TraceLike(full_text="", tool_calls=[ToolCall(name="delete")]),
    )
    assert verdict.positive
    assert verdict.extra["matched_tools"] == ["delete"]


async def test_tool_call_no_match() -> None:
    verdict = await ToolCallDetector().evaluate(
        _mode(tool_calls=["delete"]),
        TraceLike(full_text="", tool_calls=[ToolCall(name="read")]),
    )
    assert not verdict.positive
    assert verdict.extra["matched_tools"] == []


async def test_tool_call_matches_multiple() -> None:
    verdict = await ToolCallDetector().evaluate(
        _mode(tool_calls=["delete", "drop"]),
        TraceLike(
            full_text="",
            tool_calls=[ToolCall(name="delete"), ToolCall(name="drop"), ToolCall(name="read")],
        ),
    )
    assert verdict.positive
    assert verdict.extra["matched_tools"] == ["delete", "drop"]


async def test_tool_call_missing_list_raises() -> None:
    with pytest.raises(DetectionError, match="requires `tool_calls`"):
        await ToolCallDetector().evaluate(_mode(), TraceLike(full_text=""))


async def test_tool_call_empty_trace_calls() -> None:
    verdict = await ToolCallDetector().evaluate(
        _mode(tool_calls=["delete"]),
        TraceLike(full_text=""),
    )
    assert not verdict.positive

import pytest

from docket.detectors.metric_threshold import MetricThresholdDetector
from docket.errors import DetectionError
from docket.models.trace import TraceLike
from docket.rubric.spec import Detection, Mode


def _mode(**detection_kwargs: object) -> Mode:
    return Mode(
        id="m",
        severity="medium",
        detection=Detection(type="metric_threshold", **detection_kwargs),  # type: ignore[arg-type]
    )


async def test_metric_threshold_gt_positive() -> None:
    verdict = await MetricThresholdDetector().evaluate(
        _mode(metric="latency_ms", threshold=5000, operator=">"),
        TraceLike(full_text="", metrics={"latency_ms": 6000}),
    )
    assert verdict.positive
    assert verdict.extra["value"] == 6000


async def test_metric_threshold_gt_negative() -> None:
    verdict = await MetricThresholdDetector().evaluate(
        _mode(metric="latency_ms", threshold=5000, operator=">"),
        TraceLike(full_text="", metrics={"latency_ms": 1000}),
    )
    assert not verdict.positive


@pytest.mark.parametrize(
    ("op", "value", "threshold", "expected"),
    [
        ("==", 5.0, 5.0, True),
        ("==", 5.0, 6.0, False),
        ("!=", 5.0, 6.0, True),
        ("!=", 5.0, 5.0, False),
        ("<", 4.0, 5.0, True),
        ("<", 5.0, 5.0, False),
        ("<=", 5.0, 5.0, True),
        ("<=", 6.0, 5.0, False),
        (">=", 5.0, 5.0, True),
        (">=", 4.0, 5.0, False),
    ],
)
async def test_metric_threshold_operators(
    op: str, value: float, threshold: float, expected: bool
) -> None:
    verdict = await MetricThresholdDetector().evaluate(
        _mode(metric="x", threshold=threshold, operator=op),
        TraceLike(full_text="", metrics={"x": value}),
    )
    assert verdict.positive is expected


async def test_metric_threshold_missing_field_raises() -> None:
    with pytest.raises(DetectionError, match="requires metric"):
        await MetricThresholdDetector().evaluate(
            _mode(metric="x", threshold=1.0),
            TraceLike(full_text="", metrics={"x": 1.0}),
        )


async def test_metric_threshold_metric_not_on_trace_raises() -> None:
    with pytest.raises(DetectionError, match="not present on trace"):
        await MetricThresholdDetector().evaluate(
            _mode(metric="missing", threshold=1.0, operator=">"),
            TraceLike(full_text=""),
        )


async def test_metric_threshold_logical_operator_rejected() -> None:
    with pytest.raises(DetectionError, match="invalid metric_threshold operator"):
        await MetricThresholdDetector().evaluate(
            _mode(metric="x", threshold=1.0, operator="and"),
            TraceLike(full_text="", metrics={"x": 1.0}),
        )

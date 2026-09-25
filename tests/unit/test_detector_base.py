"""Tests for the `Detector` ABC's default batch implementations."""

from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.models.trace import TraceLike, Verdict
from docket.rubric.spec import Detection, Mode


class _Picky(Detector):
    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        if trace.full_text == "bad":
            raise DetectionError("bad trace")
        return Verdict(positive=trace.full_text == "yes")


_MODE = Mode(id="m", severity="low", detection=Detection(type="regex", pattern="x"))


async def test_default_evaluate_batch_isolated_keeps_failures_per_trace() -> None:
    outcomes = await _Picky().evaluate_batch_isolated(
        _MODE, [TraceLike(full_text=t) for t in ("yes", "bad", "no")]
    )
    yes, bad, no = outcomes
    assert isinstance(yes, Verdict)
    assert yes.positive
    assert isinstance(bad, DetectionError)
    assert str(bad) == "bad trace"
    assert isinstance(no, Verdict)
    assert not no.positive

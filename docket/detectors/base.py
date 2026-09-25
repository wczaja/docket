"""Detector ABC.

Each detector implements `evaluate(mode, trace) -> Verdict`. Detectors that
benefit from batched API calls (notably `LLMJudgeDetector`) override
`evaluate_batch` and `evaluate_batch_isolated` to group calls; the default
implementations fall back to serial evaluation.
"""

from abc import ABC, abstractmethod

from docket.errors import DetectionError
from docket.models.trace import TraceLike, Verdict
from docket.rubric.spec import Mode


class Detector(ABC):
    @abstractmethod
    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict: ...

    async def evaluate_batch(
        self,
        mode: Mode,
        traces: list[TraceLike],
    ) -> list[Verdict]:
        return [await self.evaluate(mode, t) for t in traces]

    async def evaluate_batch_isolated(
        self,
        mode: Mode,
        traces: list[TraceLike],
    ) -> list[Verdict | DetectionError]:
        """Like `evaluate_batch`, but a failure stays with its own trace.

        Returns one entry per trace, in order: its `Verdict`, or the
        `DetectionError` it failed with. One trace failing never discards
        another trace's verdict.
        """
        out: list[Verdict | DetectionError] = []
        for t in traces:
            try:
                out.append(await self.evaluate(mode, t))
            except DetectionError as e:
                out.append(e)
        return out

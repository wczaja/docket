"""Detector ABC.

Each detector implements `evaluate(mode, trace) -> Verdict`. Detectors that
benefit from batched API calls (notably `LLMJudgeDetector`) override
`evaluate_batch` to group calls; the default implementation falls back to
serial evaluation.
"""

from abc import ABC, abstractmethod

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

"""Classifier subagent: run every mode's detector against every trace.

Per design §4.3:
  - For each mode in the rubric, evaluate its detection. For `llm_judge` this
    is a structured LLM call; for the deterministic detectors it's pure
    Python.
  - Traces are classified in parallel (default concurrency 8).

Per design §4.4:
  - Classifier failure: up to 3 attempts total (i.e. 2 retries) with
    exponential backoff — `max_retries` counts attempts. After the final
    failure, classify as `unprocessed` (a Classification with `error` set)
    and continue — the run does NOT abort.
"""

import asyncio
from collections.abc import Awaitable, Callable

from docket.detectors import get_detector
from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.classification import Classification
from docket.models.trace import OpenInferenceTrace, TraceLike
from docket.rubric.spec import Mode, Rubric


class Classifier:
    """Async classifier with concurrency + retry-with-backoff.

    Stateless except for a detector cache keyed on detection type so the
    LLMJudgeDetector + its provider are reused across traces.
    """

    def __init__(
        self,
        llm_provider: ModelProvider,
        *,
        batch_size: int = 1,
        concurrency: int = 8,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 16.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._llm_provider = llm_provider
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._detectors: dict[str, Detector] = {}

    def _detector_for(self, detection_type: str) -> Detector:
        if detection_type not in self._detectors:
            self._detectors[detection_type] = get_detector(
                detection_type,
                llm_provider=self._llm_provider,
                batch_size=self._batch_size,
            )
        return self._detectors[detection_type]

    async def classify_trace(
        self,
        trace_id: str,
        trace: OpenInferenceTrace,
        rubric: Rubric,
    ) -> list[Classification]:
        """Run every mode's detector against `trace`. Returns one Classification per mode."""
        trace_like = trace.to_trace_like()
        rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
        out: list[Classification] = []
        for mode in rubric.modes:
            classification = await self._classify_mode(
                trace_id=trace_id,
                trace_like=trace_like,
                mode=mode,
                rubric_version=rubric_version,
            )
            out.append(classification)
        return out

    async def classify_all(
        self,
        traces: list[tuple[str, OpenInferenceTrace]],
        rubric: Rubric,
        *,
        on_progress: Callable[[str, int, int], Awaitable[None]] | None = None,
    ) -> dict[str, list[Classification]]:
        """Classify many traces concurrently. Returns {trace_id: [Classification]}."""
        sem = asyncio.Semaphore(self._concurrency)
        results: dict[str, list[Classification]] = {}
        total = len(traces)
        done_count = 0
        done_lock = asyncio.Lock()

        async def _one(trace_id: str, trace: OpenInferenceTrace) -> None:
            nonlocal done_count
            async with sem:
                results[trace_id] = await self.classify_trace(trace_id, trace, rubric)
            if on_progress is not None:
                async with done_lock:
                    done_count += 1
                    completed = done_count
                await on_progress(trace_id, completed, total)

        # return_exceptions=True so an unexpected (non-DetectionError) failure
        # in one task doesn't abandon in-flight siblings mid-await; the first
        # such failure is re-raised after every task has settled.
        outcomes = await asyncio.gather(
            *(_one(tid, t) for tid, t in traces),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return results

    async def _classify_mode(
        self,
        *,
        trace_id: str,
        trace_like: TraceLike,
        mode: Mode,
        rubric_version: str,
    ) -> Classification:
        detector = self._detector_for(mode.detection.type)
        start = asyncio.get_event_loop().time()
        last_error: DetectionError | None = None
        for attempt in range(self._max_retries):
            try:
                verdict = await detector.evaluate(mode, trace_like)
                elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000.0
                return Classification(
                    trace_id=trace_id,
                    rubric_version=rubric_version,
                    mode_id=mode.id,
                    positive=verdict.positive,
                    extra=verdict.extra,
                    duration_ms=elapsed_ms,
                )
            except DetectionError as e:
                last_error = e
                if attempt == self._max_retries - 1:
                    break
                delay = min(self._backoff_base_s * (2**attempt), self._backoff_cap_s)
                await asyncio.sleep(delay)
        elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000.0
        error_message = (
            f"after {self._max_retries} attempts: {last_error}"
            if last_error is not None
            else f"after {self._max_retries} attempts"
        )
        return Classification(
            trace_id=trace_id,
            rubric_version=rubric_version,
            mode_id=mode.id,
            positive=False,
            duration_ms=elapsed_ms,
            error=error_message,
        )


def flatten_classifications(
    by_trace: dict[str, list[Classification]],
) -> list[Classification]:
    """Flatten the dict-of-lists into a single list (stable order: input trace order)."""
    out: list[Classification] = []
    for _trace_id, items in by_trace.items():
        out.extend(items)
    return out

"""Classifier subagent: run every mode's detector against every trace.

Per design §4.3:
  - For each mode in the rubric, evaluate its detection. For `llm_judge` this
    is a structured LLM call; for the deterministic detectors it's pure
    Python.
  - Traces are classified in parallel (default concurrency 8).
  - Budget mode (`batch_size > 1`): each `llm_judge` mode judges up to
    `batch_size` traces per provider call. `composite` modes stay per trace
    even when an operand is an `llm_judge`: their deterministic operands
    decide, trace by trace, whether the judge runs at all.

Per design §4.4:
  - Classifier failure: up to 3 attempts total (i.e. 2 retries) with
    exponential backoff — `max_retries` counts attempts. After the final
    failure, classify as `unprocessed` (a Classification with `error` set)
    and continue — the run does NOT abort.
  - In budget mode a trace's first attempt is the batched call. If the call
    fails, or the trace's own verdict in it fails validation, that trace
    continues from its second attempt with single-trace calls on the same
    backoff schedule; valid verdicts from the same batch are kept.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from docket.detectors import get_detector
from docket.detectors.base import Detector
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.classification import Classification
from docket.models.trace import OpenInferenceTrace, TraceLike, Verdict
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
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
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

    def _batches(self, mode: Mode) -> bool:
        """Whether `classify_all` judges `mode` several traces per provider call.

        `docket.cost.estimate_cost` counts calls on the same rule; keep the two
        in step.
        """
        return self._batch_size > 1 and mode.detection.type == "llm_judge"

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
        """Classify many traces concurrently. Returns {trace_id: [Classification]}.

        Keys follow input order and each list follows rubric mode order,
        however the work interleaves. With `batch_size > 1`, each `llm_judge`
        mode is judged `batch_size` traces per provider call; every other mode
        runs per trace, as in `classify_trace`. At most `concurrency` units of
        work run at once, each making one provider call at a time.
        `on_progress` fires once per trace, when its last mode is classified.
        """
        rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
        modes = rubric.modes
        batched = [m for m, mode in enumerate(modes) if self._batches(mode)]
        per_trace = [m for m, mode in enumerate(modes) if not self._batches(mode)]
        trace_likes = [trace.to_trace_like() for _, trace in traces]
        loop = asyncio.get_running_loop()
        sem = asyncio.Semaphore(self._concurrency)
        # found[t][m]: trace t's Classification for modes[m], once known.
        found: list[list[Classification | None]] = [[None] * len(modes) for _ in traces]
        missing = [len(modes)] * len(traces)
        done_count = 0

        async def _record(t: int, classified: list[tuple[int, Classification]]) -> None:
            nonlocal done_count
            for m, classification in classified:
                found[t][m] = classification
            missing[t] -= len(classified)
            if missing[t] == 0 and on_progress is not None:
                done_count += 1
                await on_progress(traces[t][0], done_count, len(traces))

        async def _classify_per_trace_modes(t: int) -> None:
            async with sem:
                classified = [
                    (
                        m,
                        await self._classify_mode(
                            trace_id=traces[t][0],
                            trace_like=trace_likes[t],
                            mode=modes[m],
                            rubric_version=rubric_version,
                        ),
                    )
                    for m in per_trace
                ]
            await _record(t, classified)

        async def _judge_chunk(m: int, chunk: range) -> None:
            mode = modes[m]
            detector = self._detector_for(mode.detection.type)
            async with sem:
                start = loop.time()
                outcomes = await detector.evaluate_batch_isolated(
                    mode, [trace_likes[t] for t in chunk]
                )
                batched_s = loop.time() - start
            # Pair up first, so a detector returning the wrong number of
            # outcomes raises before any follow-up coroutine is created.
            paired = list(zip(chunk, outcomes, strict=True))
            follow_ups: list[Awaitable[None]] = []
            for t, outcome in paired:
                if isinstance(outcome, DetectionError):
                    follow_ups.append(_retry_alone(m, t, outcome, batched_s))
                else:
                    classification = _classification(
                        traces[t][0], rubric_version, mode, outcome, batched_s
                    )
                    follow_ups.append(_record(t, [(m, classification)]))
            await _settle(follow_ups)

        async def _retry_alone(m: int, t: int, error: DetectionError, batched_s: float) -> None:
            async with sem:
                classification = await self._classify_mode(
                    trace_id=traces[t][0],
                    trace_like=trace_likes[t],
                    mode=modes[m],
                    rubric_version=rubric_version,
                    batched_error=error,
                    batched_s=batched_s,
                )
            await _record(t, [(m, classification)])

        # Queue work chunk by chunk in input order so traces finish roughly in
        # order. With batch_size 1 this is one per-trace unit per trace.
        units: list[Awaitable[None]] = []
        for first in range(0, len(traces), self._batch_size):
            chunk = range(first, min(first + self._batch_size, len(traces)))
            # A rubric without modes still reports progress for every trace.
            if per_trace or not modes:
                units.extend(_classify_per_trace_modes(t) for t in chunk)
            units.extend(_judge_chunk(m, chunk) for m in batched)
        await _settle(units)
        return {
            trace_id: [c for c in found[t] if c is not None]
            for t, (trace_id, _) in enumerate(traces)
        }

    async def _classify_mode(
        self,
        *,
        trace_id: str,
        trace_like: TraceLike,
        mode: Mode,
        rubric_version: str,
        batched_error: DetectionError | None = None,
        batched_s: float = 0.0,
    ) -> Classification:
        """Classify one (trace, mode) pair: up to `max_retries` attempts with backoff.

        `batched_error` means the first attempt was this trace's share of a
        batched call (which took `batched_s`) and failed with that error; the
        remaining attempts are single-trace calls on the usual schedule.
        """
        detector = self._detector_for(mode.detection.type)
        loop = asyncio.get_running_loop()
        start = loop.time() - batched_s
        last_error = batched_error
        first_attempt = 0 if batched_error is None else 1
        for attempt in range(first_attempt, self._max_retries):
            if attempt:
                delay = min(self._backoff_base_s * (2 ** (attempt - 1)), self._backoff_cap_s)
                await asyncio.sleep(delay)
            try:
                verdict = await detector.evaluate(mode, trace_like)
            except DetectionError as e:
                last_error = e
                continue
            return _classification(trace_id, rubric_version, mode, verdict, loop.time() - start)
        elapsed_ms = (loop.time() - start) * 1000.0
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


def _classification(
    trace_id: str,
    rubric_version: str,
    mode: Mode,
    verdict: Verdict,
    elapsed_s: float,
) -> Classification:
    return Classification(
        trace_id=trace_id,
        rubric_version=rubric_version,
        mode_id=mode.id,
        positive=verdict.positive,
        extra=verdict.extra,
        duration_ms=elapsed_s * 1000.0,
    )


async def _settle(aws: Iterable[Awaitable[None]]) -> None:
    """Await every awaitable concurrently, then re-raise the first failure.

    Letting every sibling settle first means an unexpected (non-DetectionError)
    failure in one unit of work doesn't abandon the others mid-await.
    """
    outcomes = await asyncio.gather(*aws, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


def flatten_classifications(
    by_trace: dict[str, list[Classification]],
) -> list[Classification]:
    """Flatten the dict-of-lists into a single list (stable order: input trace order)."""
    out: list[Classification] = []
    for _trace_id, items in by_trace.items():
        out.extend(items)
    return out

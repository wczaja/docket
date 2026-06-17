"""Annotator subagent: write classifications back to the trace backend.

Per design §4.3:
  - Invoke `backend.annotate_trace` with `(trace_id, mode_id, confidence,
    evidence, run_id, rubric_version)`.
  - Re-running with the same `(run_id, rubric_version)` MUST overwrite, not
    duplicate.

Per design §4.4:
  - Backend write failure: retry up to 5 times. If still failing, abort the
    run with a clear error. We MUST NOT have a partial-write run.
"""

import asyncio
from typing import Any

from agent_triage.adapters.base import TraceBackend
from agent_triage.errors import BackendError
from agent_triage.models.classification import Annotation, Classification
from agent_triage.rubric.spec import Mode, Rubric


class Annotator:
    def __init__(
        self,
        backend: TraceBackend,
        *,
        run_id: str,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 16.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._backend = backend
        self._run_id = run_id
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s

    async def annotate_positive(
        self,
        classifications: list[Classification],
        rubric: Rubric,
    ) -> int:
        """Write annotations for every positive, non-error classification.

        Returns the number of annotations written. Raises `BackendError` on
        persistent failure (after the per-annotation retry budget is
        exhausted) — per design §4.4, the run must NOT proceed with partial
        writes.
        """
        modes_by_id = {m.id: m for m in rubric.modes}
        written = 0
        for c in classifications:
            if c.error is not None or not c.positive:
                continue
            mode = modes_by_id.get(c.mode_id)
            if mode is None:
                continue
            annotation = self._to_annotation(c, mode)
            await self._annotate_with_retry(c.trace_id, annotation)
            written += 1
        return written

    def _to_annotation(self, c: Classification, mode: Mode) -> Annotation:
        extra = c.extra or {}
        return Annotation(
            trace_id=c.trace_id,
            run_id=self._run_id,
            rubric_version=c.rubric_version,
            mode_id=c.mode_id,
            positive=c.positive,
            severity=mode.severity,
            confidence=_extract_confidence(extra),
            excerpt=_extract_excerpt(extra),
            notes=extra,
        )

    async def _annotate_with_retry(
        self,
        trace_id: str,
        annotation: Annotation,
    ) -> None:
        last_error: BackendError | None = None
        for attempt in range(self._max_retries):
            try:
                await self._backend.annotate_trace(trace_id, annotation)
                return
            except BackendError as e:
                last_error = e
                if attempt == self._max_retries - 1:
                    break
                delay = min(self._backoff_base_s * (2**attempt), self._backoff_cap_s)
                await asyncio.sleep(delay)
        msg = (
            f"Annotation write failed for trace {trace_id!r} after "
            f"{self._max_retries} attempts: {last_error}"
        )
        raise BackendError(msg)


def _extract_confidence(extra: dict[str, Any]) -> float | None:
    v = extra.get("confidence")
    return float(v) if isinstance(v, (int, float)) else None


def _extract_excerpt(extra: dict[str, Any]) -> str | None:
    v = extra.get("excerpt") or extra.get("match")
    return str(v) if isinstance(v, str) else None

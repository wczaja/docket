"""Annotation and classification value types.

`Classification` is the output of a single detector run on a single trace; it
stays local to the runtime. `Annotation` is the trace-backend-bound view that
gets posted via `TraceBackend.annotate_trace`.

The annotation key — `(trace_id, run_id, rubric_version, mode_id)` — is what
the design uses to make annotation writes idempotent across re-runs.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["critical", "high", "medium", "low"]


class Classification(BaseModel):
    """One detector's run on one trace, in-process.

    `error` carries the detector's failure message when classification could
    not complete (e.g. metric_threshold against a trace missing the metric).
    A classification with `error` set MUST have `positive=False` and SHOULD
    NOT be turned into an annotation.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    rubric_version: str
    mode_id: str
    positive: bool
    extra: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None


class Annotation(BaseModel):
    """A classification ready to write back to the trace backend.

    `run_id` is the agent-triage run that produced this; together with
    `(trace_id, rubric_version, mode_id)` it forms the idempotency key the
    backend should use to upsert.
    """

    trace_id: str
    run_id: str
    rubric_version: str
    mode_id: str
    positive: bool
    severity: Severity
    confidence: float | None = None
    excerpt: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def idempotency_key(self) -> str:
        return f"{self.trace_id}|{self.run_id}|{self.rubric_version}|{self.mode_id}"

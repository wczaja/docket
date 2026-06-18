"""Trace-listing result types (proposal 001 Specs B + C).

`TraceListing` is what `TraceBackend.list_traces_v2` returns: per-trace
summaries plus a loud `truncated` flag so a backend pagination ceiling can
never silently masquerade as "the whole window". `TraceSummary` carries only
fields derivable from the listing response alone — building summaries via
per-trace `get_trace` calls is forbidden (it would erase the sampling
savings). `trace_id`-only summaries are valid; that is what the base-class
default produces for adapters that only implement the legacy `list_traces`.

`TraceFilter` is the typed view of the reserved keys inside the existing
`filter: dict` parameter on `list_traces` / `list_traces_v2`. Adapters MUST
honor reserved keys or raise `BackendError`; backend-specific passthrough
keys remain MAY-ignore (see `docket.adapters.base`).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from docket.errors import BackendError

TraceStatus = Literal["ok", "error"]

# Reserved keys in the `filter` dict that adapters MUST honor or reject.
RESERVED_FILTER_KEYS: frozenset[str] = frozenset({"status"})


class TraceSummary(BaseModel):
    """Per-trace metadata available at list time.

    `status` reflects the *root* run/span only — traces where just a child
    span errored are not "error" here, because the child tree is unknown at
    list time. `tags` carry listing-level key/value metadata (e.g. tenant
    key, deployment label) where the backend exposes it.
    """

    trace_id: str
    start_time: datetime | None = None
    status: TraceStatus | None = None
    latency_ms: float | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class TraceListing(BaseModel):
    """Result of `TraceBackend.list_traces_v2`.

    `truncated` means the adapter stopped paginating at its page ceiling
    with a full last page, so `summaries` is a lower bound on the window —
    and any sample drawn from it samples the truncated frame, not the
    window population.
    """

    summaries: list[TraceSummary]
    truncated: bool = False
    page_limit: int | None = None

    @property
    def trace_ids(self) -> list[str]:
        return [s.trace_id for s in self.summaries]


class TraceFilter(BaseModel):
    """Typed view of the reserved `filter` keys.

    Serializes into the existing `filter: dict` parameter — no adapter
    signature change. `status="error"` means "only traces whose root
    run/span ended in error".
    """

    status: TraceStatus | None = None

    def to_filter_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def parse_status_filter(filter: dict[str, Any] | None) -> TraceStatus | None:
    """Extract and validate the reserved `status` key from a filter dict.

    Raises `BackendError` on an unsupported value, so a typo'd filter fails
    loudly instead of silently listing the wrong population. Non-reserved
    keys are left for the adapter to interpret (or ignore).
    """
    if not filter or "status" not in filter:
        return None
    value = filter["status"]
    if value not in ("ok", "error"):
        raise BackendError(
            f"unsupported value for reserved filter key 'status': {value!r} "
            "(expected 'ok' or 'error')"
        )
    return value  # type: ignore[no-any-return]

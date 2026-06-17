"""Run report types.

`RunReport` is the output of `runtime.run_triage()` — a per-mode + per-trace
summary of one classification pass. Phase 4 prints this as a plain-text
table; Phase 5 layers a markdown `report.md` on top with clustering and
severity rollups.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent_triage.models.classification import Classification, Severity

if TYPE_CHECKING:
    pass


class TraceResult(BaseModel):
    trace_id: str
    classifications: list[Classification] = Field(default_factory=list)

    @property
    def positive_modes(self) -> list[str]:
        return [c.mode_id for c in self.classifications if c.positive and c.error is None]

    @property
    def error_modes(self) -> list[str]:
        return [c.mode_id for c in self.classifications if c.error is not None]


class ModeStats(BaseModel):
    mode_id: str
    severity: Severity
    positive_count: int = 0
    negative_count: int = 0
    error_count: int = 0


class TrackerFailure(BaseModel):
    """One draft whose tracker writeback failed (design §4.4: report, don't abort).

    `reason` MUST already be redacted by the caller — it may quote a tracker
    error message that embeds user data.
    """

    cluster_id: str
    mode_id: str
    reason: str


class FetchFailure(BaseModel):
    """One trace that could not be fetched (design §4.4: log, skip, report).

    `reason` MUST already be redacted by the caller — it may quote a backend
    error message that embeds user data.
    """

    trace_id: str
    reason: str


class RunReport(BaseModel):
    run_id: str
    rubric_name: str
    rubric_version: str
    since: datetime
    until: datetime
    started_at: datetime
    finished_at: datetime
    trace_count: int
    trace_results: list[TraceResult] = Field(default_factory=list)
    mode_stats: list[ModeStats] = Field(default_factory=list)
    annotations_written: int = 0
    tracker_failures: list[TrackerFailure] = Field(default_factory=list)
    fetch_failures: list[FetchFailure] = Field(default_factory=list)

    def render_table(self) -> str:
        lines: list[str] = []
        lines.append(f"agent-triage run {self.run_id}")
        lines.append(f"  rubric: {self.rubric_name} v{self.rubric_version}")
        lines.append(f"  window: {self.since.isoformat()} -> {self.until.isoformat()}")
        duration = (self.finished_at - self.started_at).total_seconds()
        lines.append(f"  traces: {self.trace_count}   elapsed: {duration:.1f}s")
        if self.fetch_failures:
            lines.append(
                f"  processed: {self.trace_count - len(self.fetch_failures)}   "
                f"skipped (fetch failures): {len(self.fetch_failures)}"
            )
        if self.annotations_written:
            lines.append(f"  annotations written: {self.annotations_written}")
        lines.append("")
        lines.append("Mode summary:")
        lines.append(f"  {'mode':<32}{'severity':<10}{'pos':>6}{'neg':>6}{'err':>6}")
        lines.append(f"  {'-' * 60}")
        for ms in self.mode_stats:
            lines.append(
                f"  {ms.mode_id:<32}{ms.severity:<10}"
                f"{ms.positive_count:>6}{ms.negative_count:>6}{ms.error_count:>6}"
            )
        positives = [r for r in self.trace_results if r.positive_modes]
        if positives:
            lines.append("")
            lines.append("Positive classifications:")
            for r in positives:
                lines.append(f"  {r.trace_id}: {', '.join(r.positive_modes)}")
        errors = [r for r in self.trace_results if r.error_modes]
        if errors:
            lines.append("")
            lines.append("Detector errors (trace not aborted):")
            for r in errors:
                lines.append(f"  {r.trace_id}: {', '.join(r.error_modes)}")
        if self.tracker_failures:
            lines.append("")
            lines.append("Tracker failures (drafts remain queued; run not aborted):")
            for tf in self.tracker_failures:
                lines.append(f"  {tf.cluster_id} ({tf.mode_id}): {tf.reason}")
        return "\n".join(lines)

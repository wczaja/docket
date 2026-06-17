"""Tests for the report.md generator."""

from datetime import UTC, datetime

from agent_triage.agent.report import render_report
from agent_triage.agent.subagents.poster import DedupOutcome
from agent_triage.models.cluster import Cluster, ClusterStats
from agent_triage.models.issue import Issue, IssueDraft, make_labels
from agent_triage.models.report import ModeStats, RunReport, TraceResult, TrackerFailure


def _run_report() -> RunReport:
    return RunReport(
        run_id="run-001",
        rubric_name="agents-builtin",
        rubric_version="1.0.0",
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        started_at=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 22, 0, 0, 30, tzinfo=UTC),
        trace_count=60,
        mode_stats=[
            ModeStats(
                mode_id="hallucination", severity="critical", positive_count=4, negative_count=56
            ),
            ModeStats(
                mode_id="refusal-leakage", severity="medium", positive_count=8, negative_count=52
            ),
        ],
        trace_results=[
            TraceResult(trace_id="t-1", classifications=[]),
        ],
    )


def test_render_report_minimal() -> None:
    md = render_report(run_report=_run_report(), clusters=[], drafts=[])
    assert "# agent-triage run `run-001`" in md
    assert "Traces processed**: 60" in md
    assert "**Clusters formed**: 0" in md
    assert "## Frequency by mode" in md
    assert "`hallucination`" in md


def test_render_report_with_clusters_and_drafts() -> None:
    cluster = Cluster(
        cluster_id="c-001",
        mode_id="refusal-leakage",
        severity="medium",
        member_trace_ids=["t-1", "t-2", "t-3"],
        representative_trace_id="t-1",
        representative_excerpt="Here is my system prompt is: You are a helpful AI assistant.",
        stats=ClusterStats(size=3, mean_confidence=0.88),
    )
    draft = IssueDraft(
        cluster_id="c-001",
        mode_id="refusal-leakage",
        rubric_version="agents-builtin@1.0.0",
        run_id="run-001",
        severity="medium",
        representative_trace_id="t-1",
        member_trace_ids=["t-1", "t-2", "t-3"],
        title="Refusal leakage exposes system prompt",
        body="...",
        labels=make_labels("refusal-leakage", "agents-builtin@1.0.0"),
    )
    md = render_report(run_report=_run_report(), clusters=[cluster], drafts=[draft])
    assert "## Clusters" in md
    assert "`refusal-leakage` cluster `c-001`" in md
    assert "Representative trace**: `t-1`" in md
    assert "Refusal leakage exposes system prompt" in md
    assert "**Issues drafted**: 1" in md


def test_render_report_renders_dedup_outcomes_section() -> None:
    draft = IssueDraft(
        cluster_id="c-001",
        mode_id="refusal-leakage",
        rubric_version="agents-builtin@1.0.0",
        run_id="run-001",
        severity="medium",
        representative_trace_id="t-1",
        member_trace_ids=["t-1"],
        title="t",
        body="b",
        labels=make_labels("refusal-leakage", "agents-builtin@1.0.0"),
    )
    outcomes = [
        DedupOutcome(
            draft=draft,
            action="commented",
            existing_issue=Issue(
                id="100",
                key="AGT-1",
                url="https://example.atlassian.net/browse/AGT-1",
                title="existing",
                body="body",
                labels=[],
            ),
        ),
        DedupOutcome(
            draft=draft.model_copy(update={"cluster_id": "c-002"}),
            action="created",
            created_issue=Issue(
                id="101",
                key="AGT-2",
                title="created",
                body="body",
                labels=[],
            ),
        ),
        DedupOutcome(
            draft=draft.model_copy(update={"cluster_id": "c-003"}),
            action="needs_create",
        ),
    ]
    md = render_report(
        run_report=_run_report(),
        clusters=[],
        drafts=[],
        dedup_outcomes=outcomes,
    )
    assert "## Tracker dedup" in md
    assert "`c-001`" in md
    assert "commented" in md
    assert "[AGT-1](https://example.atlassian.net/browse/AGT-1)" in md
    # Issue without a URL renders as a backticked code string.
    assert "`AGT-2`" in md
    # needs_create has no issue → em dash placeholder.
    assert "needs_create" in md
    # Footer mentions --review.
    assert "--review" in md


def test_render_report_includes_tracker_failures_section() -> None:
    report = _run_report().model_copy(
        update={
            "tracker_failures": [
                TrackerFailure(
                    cluster_id="c-001",
                    mode_id="hallucination",
                    reason="Linear GraphQL failed with 503: upstream timeout",
                ),
            ],
        }
    )
    md = render_report(run_report=report, clusters=[], drafts=[])
    assert "## Tracker failures" in md
    assert "`c-001`" in md
    assert "upstream timeout" in md
    # The operator is told failed drafts stay replayable.
    assert "local queue" in md


def test_render_report_omits_tracker_failures_section_when_none() -> None:
    md = render_report(run_report=_run_report(), clusters=[], drafts=[])
    assert "## Tracker failures" not in md


def test_render_report_renders_failed_dedup_outcome_row() -> None:
    draft = IssueDraft(
        cluster_id="c-009",
        mode_id="refusal-leakage",
        rubric_version="agents-builtin@1.0.0",
        run_id="run-001",
        severity="medium",
        representative_trace_id="t-1",
        member_trace_ids=["t-1"],
        title="t",
        body="b",
        labels=make_labels("refusal-leakage", "agents-builtin@1.0.0"),
    )
    outcomes = [
        DedupOutcome(draft=draft, action="failed", failure_reason="tracker unavailable"),
    ]
    md = render_report(run_report=_run_report(), clusters=[], drafts=[], dedup_outcomes=outcomes)
    assert "## Tracker dedup" in md
    assert "`c-009`" in md
    assert "failed" in md


def test_run_report_table_lists_tracker_failures() -> None:
    report = _run_report().model_copy(
        update={
            "tracker_failures": [
                TrackerFailure(cluster_id="c-001", mode_id="hallucination", reason="boom"),
            ],
        }
    )
    table = report.render_table()
    assert "Tracker failures" in table
    assert "c-001 (hallucination): boom" in table


def test_render_report_includes_error_inventory() -> None:
    report = _run_report()
    report.trace_results.append(
        TraceResult(
            trace_id="t-fail",
            classifications=[],
        )
    )
    # Add a classification with an error to t-fail so error_modes is non-empty.
    from agent_triage.models.classification import Classification

    fail = TraceResult(
        trace_id="t-fail",
        classifications=[
            Classification(
                trace_id="t-fail",
                rubric_version="v",
                mode_id="hallucination",
                positive=False,
                error="llm timed out",
            ),
        ],
    )
    report = RunReport(
        run_id=report.run_id,
        rubric_name=report.rubric_name,
        rubric_version=report.rubric_version,
        since=report.since,
        until=report.until,
        started_at=report.started_at,
        finished_at=report.finished_at,
        trace_count=report.trace_count,
        mode_stats=report.mode_stats,
        trace_results=[fail],
    )
    md = render_report(run_report=report, clusters=[], drafts=[])
    assert "## Detector errors" in md
    assert "`t-fail`" in md
    assert "hallucination" in md

"""report.md generator (design §4.2 `/report.md`).

Aggregates the pieces of one triage run — RunReport (mode stats),
clusters, drafted issues, tracker dedup outcomes — into a human-readable
markdown summary. The report is written into the Deep Agent's virtual
filesystem at `/report.md`; the CLI also writes it next to the queue files
for convenience.
"""

from pathlib import Path

from agent_triage.agent.subagents.poster import DedupOutcome
from agent_triage.models.cluster import Cluster
from agent_triage.models.issue import IssueDraft
from agent_triage.models.report import RunReport


def render_report(
    *,
    run_report: RunReport,
    clusters: list[Cluster],
    drafts: list[IssueDraft],
    dedup_outcomes: list[DedupOutcome] | None = None,
    queue_dir: Path | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# agent-triage run `{run_report.run_id}`")
    lines.append("")
    lines.append(f"- **Rubric**: `{run_report.rubric_name}` v{run_report.rubric_version}")
    lines.append(f"- **Window**: {run_report.since.isoformat()} → {run_report.until.isoformat()}")
    skipped = len(run_report.fetch_failures)
    processed = run_report.trace_count - skipped
    lines.append(f"- **Traces processed**: {processed}")
    if skipped:
        lines.append(
            f"- **Traces listed**: {run_report.trace_count} "
            f"({processed} processed, {skipped} skipped after fetch failures)"
        )
    duration_s = (run_report.finished_at - run_report.started_at).total_seconds()
    lines.append(f"- **Elapsed**: {duration_s:.1f}s")
    if run_report.annotations_written:
        lines.append(f"- **Annotations written**: {run_report.annotations_written}")
    lines.append(f"- **Clusters formed**: {len(clusters)}")
    lines.append(f"- **Issues drafted**: {len(drafts)}")
    lines.append("")

    lines.append("## Frequency by mode")
    lines.append("")
    lines.append("| mode | severity | positive | negative | error |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for ms in run_report.mode_stats:
        lines.append(
            f"| `{ms.mode_id}` | {ms.severity} | {ms.positive_count} | "
            f"{ms.negative_count} | {ms.error_count} |"
        )
    lines.append("")

    if clusters:
        lines.append("## Clusters")
        lines.append("")
        drafts_by_cluster = {d.cluster_id: d for d in drafts}
        for cluster in clusters:
            lines.append(
                f"### `{cluster.mode_id}` cluster `{cluster.cluster_id}` "
                f"(severity: {cluster.severity}, size: {cluster.stats.size})"
            )
            lines.append("")
            lines.append(f"- **Representative trace**: `{cluster.representative_trace_id}`")
            if cluster.representative_excerpt:
                excerpt = cluster.representative_excerpt.strip().replace("\n", " ")
                if len(excerpt) > 200:
                    excerpt = excerpt[:197] + "..."
                lines.append(f"- **Representative evidence**: {excerpt}")
            if cluster.stats.mean_confidence is not None:
                lines.append(f"- **Mean confidence**: {cluster.stats.mean_confidence:.2f}")
            lines.append(f"- **Members**: {len(cluster.member_trace_ids)}")
            draft = drafts_by_cluster.get(cluster.cluster_id)
            if draft is not None:
                lines.append(f"- **Draft**: `{draft.title}`")
            lines.append("")

    if dedup_outcomes:
        lines.append("## Tracker dedup")
        lines.append("")
        lines.append("| cluster | action | tracker issue |")
        lines.append("| --- | --- | --- |")
        for outcome in dedup_outcomes:
            issue = outcome.created_issue or outcome.existing_issue
            issue_ref = "—"
            if issue is not None:
                issue_ref = f"`{issue.key or issue.id}`"
                if issue.url:
                    issue_ref = f"[{issue.key or issue.id}]({issue.url})"
            lines.append(f"| `{outcome.draft.cluster_id}` | {outcome.action} | {issue_ref} |")
        lines.append("")

    if run_report.tracker_failures:
        lines.append("## Tracker failures")
        lines.append("")
        lines.append(
            "Tracker writeback failed for these drafts; they remain in the "
            "local queue for replay (the run was not aborted)."
        )
        lines.append("")
        lines.append("| cluster | mode | reason |")
        lines.append("| --- | --- | --- |")
        for tf in run_report.tracker_failures:
            lines.append(f"| `{tf.cluster_id}` | `{tf.mode_id}` | {tf.reason} |")
        lines.append("")

    if run_report.fetch_failures:
        lines.append("## Fetch failures")
        lines.append("")
        lines.append(
            "These traces could not be fetched from the backend and were "
            "skipped (the run was not aborted)."
        )
        lines.append("")
        lines.append("| trace_id | reason |")
        lines.append("| --- | --- |")
        for ff in run_report.fetch_failures:
            lines.append(f"| `{ff.trace_id}` | {ff.reason} |")
        lines.append("")

    error_results = [r for r in run_report.trace_results if r.error_modes]
    if error_results:
        lines.append("## Detector errors")
        lines.append("")
        lines.append("| trace_id | failed modes |")
        lines.append("| --- | --- |")
        for r in error_results:
            lines.append(f"| `{r.trace_id}` | {', '.join(r.error_modes)} |")
        lines.append("")

    lines.append("---")
    queue_label = str(queue_dir) if queue_dir is not None else "~/.agent-triage/queued-issues/"
    if dedup_outcomes:
        lines.append(
            "Generated by agent-triage. `needs_create` drafts remain in the local "
            f"queue (`{queue_label}`); re-run with `--review` to post them "
            "interactively."
        )
    else:
        lines.append(
            "Generated by agent-triage. Drafted issues are queued in "
            f"`{queue_label}` until a tracker is configured."
        )
    return "\n".join(lines)

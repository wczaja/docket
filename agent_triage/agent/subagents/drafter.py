"""Drafter subagent: structured-output LLM call that turns a Cluster into an IssueDraft.

Per design §5.2 every draft carries:
  - An HTML-comment provenance block at the end of the body
  - The label set `agent-triage`, `mode:<id>`, `rubric:<rubric-id>@<version>`

Phase 5 writes drafts to a local-file queue at `~/.agent-triage/queued-issues/`
(JSON + Markdown side-by-side). Tracker integration arrives in Phase 8; the
queue is what the design calls "the configured work directory" default for
tracker-write fallback. The drafter doesn't do tracker dedup yet (no
`list_open_issues` MCP call) — that lands with the trackers.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.models.cluster import Cluster
from agent_triage.models.issue import (
    IssueDraft,
    IssueProvenance,
    make_labels,
)
from agent_triage.rubric.spec import Mode, Rubric

DEFAULT_QUEUE_DIR = Path.home() / ".agent-triage" / "queued-issues"

_SYSTEM_PROMPT = (
    "You are an experienced site-reliability engineer drafting a tracker issue "
    "for an LLM agent failure-mode cluster. You write clear, actionable issue "
    "titles and bodies. Output strictly the JSON object described by the schema."
)

_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["title", "body"],
    "properties": {
        "title": {
            "type": "string",
            "minLength": 5,
            "maxLength": 120,
            "description": "Concise issue title (<= 120 chars).",
        },
        "body": {
            "type": "string",
            "minLength": 30,
            "description": (
                "Markdown body covering: 1) what's happening, 2) representative "
                "evidence, 3) frequency / blast radius, 4) suggested next step."
            ),
        },
    },
}


async def draft_issues(
    clusters: list[Cluster],
    *,
    rubric: Rubric,
    llm_provider: ModelProvider,
    run_id: str,
    output_dir: Path | None = None,
) -> list[IssueDraft]:
    """Draft one issue per Cluster and write the queue files.

    Returns the in-memory drafts; the JSON + markdown files are written to
    `output_dir` (default: `~/.agent-triage/queued-issues/`).
    """
    queue_dir = output_dir if output_dir is not None else DEFAULT_QUEUE_DIR
    queue_dir.mkdir(parents=True, exist_ok=True)
    modes_by_id = {m.id: m for m in rubric.modes}
    rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
    severity_to_priority = (
        rubric.triage.default_severity_to_tracker if rubric.triage is not None else {}
    )

    drafts: list[IssueDraft] = []
    for cluster in clusters:
        mode = modes_by_id.get(cluster.mode_id)
        if mode is None:
            continue
        draft = await _draft_one(
            cluster=cluster,
            mode=mode,
            llm_provider=llm_provider,
            rubric_version=rubric_version,
            run_id=run_id,
            priority=severity_to_priority.get(cluster.severity),
        )
        # File writes go to a worker thread; the event loop may be running
        # concurrent classifier/embedding calls.
        await asyncio.to_thread(_write_draft, draft, queue_dir)
        drafts.append(draft)
    return drafts


async def _draft_one(
    *,
    cluster: Cluster,
    mode: Mode,
    llm_provider: ModelProvider,
    rubric_version: str,
    run_id: str,
    priority: str | None = None,
) -> IssueDraft:
    user_prompt = _build_user_prompt(cluster, mode)
    result = await llm_provider.structured_complete(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        schema=_DRAFT_SCHEMA,
    )
    title = result.get("title")
    body = result.get("body")
    if not isinstance(title, str) or not isinstance(body, str):
        raise DetectionError(
            f"Drafter LLM did not return title/body strings for cluster {cluster.cluster_id!r}"
        )
    provenance = IssueProvenance(
        rubric_version=rubric_version,
        mode_id=cluster.mode_id,
        cluster_id=cluster.cluster_id,
        representative_trace_id=cluster.representative_trace_id,
        run_id=run_id,
        member_trace_ids=list(cluster.member_trace_ids),
    )
    full_body = f"{body}\n\n{provenance.to_html_comment()}"
    labels = make_labels(cluster.mode_id, rubric_version)
    return IssueDraft(
        cluster_id=cluster.cluster_id,
        mode_id=cluster.mode_id,
        rubric_version=rubric_version,
        run_id=run_id,
        severity=cluster.severity,
        representative_trace_id=cluster.representative_trace_id,
        member_trace_ids=list(cluster.member_trace_ids),
        title=title.strip(),
        body=full_body,
        labels=labels,
        priority=priority,
    )


def _build_user_prompt(cluster: Cluster, mode: Mode) -> str:
    parts: list[str] = []
    parts.append(f"Failure mode: {mode.id}")
    if mode.name:
        parts.append(f"Mode name: {mode.name}")
    if mode.description:
        parts.append(f"Mode description:\n{mode.description.strip()}")
    parts.append(f"Severity: {mode.severity}")
    parts.append(f"Cluster size: {cluster.stats.size}")
    if cluster.stats.mean_confidence is not None:
        parts.append(f"Mean confidence across cluster: {cluster.stats.mean_confidence:.2f}")
    parts.append(f"Representative trace ID: {cluster.representative_trace_id}")
    if cluster.representative_excerpt:
        parts.append(f"Representative evidence:\n---\n{cluster.representative_excerpt}\n---")
    parts.append(
        "Write an issue draft. The title should describe the failure mode "
        "concretely (not just the mode name). The body should explain what's "
        "happening, cite the representative evidence, name the frequency, and "
        "suggest a concrete next step. Return JSON {title, body}."
    )
    return "\n\n".join(parts)


def _write_draft(draft: IssueDraft, queue_dir: Path) -> None:
    json_path = queue_dir / f"{draft.cluster_id}.json"
    md_path = queue_dir / f"{draft.cluster_id}.md"
    json_path.write_text(json.dumps(draft.to_json_record(), indent=2, sort_keys=True))
    md_path.write_text(draft.to_markdown())

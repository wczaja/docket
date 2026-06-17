"""Local draft-queue inspection and replay.

The drafter writes every issue draft to the queue directory
(`~/.agent-triage/queued-issues/` by default) as `{cluster_id}.json` +
`{cluster_id}.md`. This module is the read side: list queued drafts, post
them to a tracker, and retire posted files into a `posted/` subdirectory so
a replay can never double-post. It is deliberately filesystem-only — the
queue is the §4.4 fallback for tracker write failures, so it must not
depend on the tracker being reachable.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent_triage.agent.subagents.drafter import DEFAULT_QUEUE_DIR
from agent_triage.errors import ConfigError
from agent_triage.models.issue import IssueDraft

log = logging.getLogger(__name__)

POSTED_SUBDIR = "posted"

__all__ = [
    "DEFAULT_QUEUE_DIR",
    "POSTED_SUBDIR",
    "QueuedDraft",
    "clear_queue",
    "list_queued_drafts",
    "mark_posted",
]


@dataclass(frozen=True)
class QueuedDraft:
    """One on-disk draft: the parsed model plus its backing files."""

    draft: IssueDraft
    json_path: Path
    md_path: Path | None


def list_queued_drafts(queue_dir: Path | None = None) -> list[QueuedDraft]:
    """Load every parseable draft in `queue_dir`, sorted by file name.

    Unparseable JSON files are skipped with a warning rather than failing
    the listing — one corrupt draft must not block replay of the rest.
    """
    base = queue_dir if queue_dir is not None else DEFAULT_QUEUE_DIR
    if not base.is_dir():
        return []
    out: list[QueuedDraft] = []
    for json_path in sorted(base.glob("*.json")):
        try:
            record = json.loads(json_path.read_text())
            draft = IssueDraft.model_validate(record)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("skipping unparseable queued draft %s: %s", json_path.name, e)
            continue
        md_path = json_path.with_suffix(".md")
        out.append(
            QueuedDraft(
                draft=draft,
                json_path=json_path,
                md_path=md_path if md_path.exists() else None,
            )
        )
    return out


def mark_posted(queued: QueuedDraft, *, issue_url: str | None = None) -> Path:
    """Move a posted draft's files into `posted/` so replay can't double-post.

    Returns the new JSON path. The issue URL, when known, is recorded in the
    moved JSON record under `posted_issue_url`.
    """
    posted_dir = queued.json_path.parent / POSTED_SUBDIR
    posted_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads(queued.json_path.read_text())
    if issue_url:
        record["posted_issue_url"] = issue_url
    target = posted_dir / queued.json_path.name
    target.write_text(json.dumps(record, indent=2, sort_keys=True))
    queued.json_path.unlink()
    if queued.md_path is not None and queued.md_path.exists():
        queued.md_path.rename(posted_dir / queued.md_path.name)
    return target


def clear_queue(queue_dir: Path | None = None) -> int:
    """Delete all queued (non-posted) draft files; returns the count removed."""
    base = queue_dir if queue_dir is not None else DEFAULT_QUEUE_DIR
    if not base.is_dir():
        return 0
    if base == Path("/"):  # defense in depth; never sweep a filesystem root
        raise ConfigError("Refusing to clear queue at filesystem root")
    removed = 0
    for json_path in sorted(base.glob("*.json")):
        md_path = json_path.with_suffix(".md")
        json_path.unlink()
        if md_path.exists():
            md_path.unlink()
        removed += 1
    return removed

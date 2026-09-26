"""Local draft-queue inspection and replay.

The drafter writes every issue draft to the queue directory
(`~/.docket/queued-issues/` by default) as `{cluster_id}.json` +
`{cluster_id}.md`, before any tracker call. Once the tracker represents a
draft — dedup created its issue, or found an open issue already tracking
its cluster — its files retire into a `posted/` subdirectory, so a replay
can never double-post. What stays queued is what still needs a tracker
write: every draft when no tracker is configured, `needs_create` drafts
below `auto_post_threshold`, and drafts whose tracker write failed.

This module is the filesystem side: list queued drafts, retire handled
ones, clear the rest. It is deliberately filesystem-only — the queue is
the §4.4 fallback for tracker write failures, so it must not depend on the
tracker being reachable.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from docket.agent.subagents.drafter import DEFAULT_QUEUE_DIR
from docket.errors import ConfigError
from docket.models.issue import Issue, IssueDraft

log = logging.getLogger(__name__)

POSTED_SUBDIR = "posted"

__all__ = [
    "DEFAULT_QUEUE_DIR",
    "POSTED_SUBDIR",
    "TRACKED_ACTIONS",
    "QueuedDraft",
    "clear_queue",
    "list_queued_drafts",
    "mark_posted",
    "retire_drafts",
]

# Dedup actions after which the tracker represents a draft: its issue was
# filed (`created`), or an open issue already tracks its cluster
# (`commented`, `skipped`). Only these retire a draft to `posted/`;
# `needs_create` and `failed` drafts stay queued for replay.
TRACKED_ACTIONS: frozenset[str] = frozenset({"created", "commented", "skipped"})


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
    return _retire(queued.json_path, queued.md_path, issue_url=issue_url)


def retire_drafts(
    issues: Mapping[str, Issue | None],
    *,
    queue_dir: Path | None = None,
) -> int:
    """Retire the queued drafts the tracker now represents; best-effort.

    `issues` maps each handled draft's cluster ID to the tracker issue that
    now represents it (created, or already open); its URL is recorded as
    `posted_issue_url`, like `mark_posted`. Cluster IDs with no queued
    draft are ignored. A failed retirement is logged, not raised: the
    tracker write it follows already succeeded, and the leftover draft is
    deduped again on replay. Returns how many drafts moved to `posted/`.
    """
    base = queue_dir if queue_dir is not None else DEFAULT_QUEUE_DIR
    retired = 0
    for cluster_id, issue in issues.items():
        json_path = base / f"{cluster_id}.json"
        try:
            if not json_path.is_file():
                continue
            _retire(
                json_path,
                json_path.with_suffix(".md"),
                issue_url=issue.url if issue is not None else None,
            )
        except (OSError, ValueError) as e:
            log.warning("could not retire queued draft %s: %s", cluster_id, e)
            continue
        retired += 1
    return retired


def _retire(json_path: Path, md_path: Path | None, *, issue_url: str | None) -> Path:
    posted_dir = json_path.parent / POSTED_SUBDIR
    posted_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads(json_path.read_text())
    if issue_url:
        record["posted_issue_url"] = issue_url
    target = posted_dir / json_path.name
    target.write_text(json.dumps(record, indent=2, sort_keys=True))
    if md_path is not None and md_path.exists():
        # `replace`, not `rename`: re-running a window retires the same
        # cluster again, and only `replace` overwrites on every platform.
        md_path.replace(posted_dir / md_path.name)
    # Unlinking the queued JSON is the commit point: until then the draft
    # still lists as queued, so an interrupted retire is simply redone.
    json_path.unlink()
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

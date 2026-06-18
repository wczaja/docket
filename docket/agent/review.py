"""Interactive review mode (design §7 Phase 8 PR 3).

Walks the operator through each `needs_create` outcome from the dedup loop,
opens the draft in `$EDITOR` so they can refine title / body, then prompts
accept-or-reject. Accepted drafts are posted via `tracker.create_issue`;
rejected drafts stay in the local queue.

When `$EDITOR` is unset (CI containers, minimal images), the review falls
back to printing the draft to stdout and prompting y/n without an editor
step — the operator can still accept or reject, but can't edit inline.

The implementation is deliberately stdlib-only (`subprocess`, `tempfile`)
so the review surface works in headless / minimal environments without
extra dependencies.
"""

import asyncio
import os
import re
import shlex
import subprocess  # noqa: S404  -- invoking the operator's own $EDITOR is the goal
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click

from docket.adapters.base import Tracker
from docket.agent.subagents.poster import DedupOutcome
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssueProvenance

# Confirm prompt is injected so tests can drive the flow without stdin.
ConfirmFn = Callable[[str], bool]


@dataclass(frozen=True)
class ReviewOutcome:
    """Result of `--review`-driven posting for one draft."""

    draft: IssueDraft
    action: str  # "posted" | "rejected" | "edit_cancelled" | "post_failed"
    posted_issue: Issue | None = None


async def review_and_post(
    outcomes: list[DedupOutcome],
    *,
    tracker: Tracker,
    editor: str | None = None,
    confirm: ConfirmFn | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> list[ReviewOutcome]:
    """Loop `needs_create` outcomes through editor + accept/reject + post.

    `outcomes` from any other action (`skipped`, `commented`, `created`)
    are ignored — they've already been handled by the dedup loop.

    Returns one `ReviewOutcome` per draft considered.
    """
    confirm_fn: ConfirmFn = confirm or _default_confirm
    print_fn = print_fn or click.echo
    editor_cmd = editor if editor is not None else os.environ.get("EDITOR", "")
    results: list[ReviewOutcome] = []
    needs_create = [o for o in outcomes if o.action == "needs_create"]
    if not needs_create:
        return results
    for outcome in needs_create:
        draft = outcome.draft
        # The editor session blocks on operator input; run it in a worker
        # thread so the event loop (and any background tasks) stay live.
        edited = await asyncio.to_thread(
            _edit_draft, draft, editor_cmd=editor_cmd, print_fn=print_fn
        )
        if edited is None:
            results.append(ReviewOutcome(draft=draft, action="edit_cancelled"))
            continue
        print_fn(
            f"\n=== Draft for cluster {draft.cluster_id} "
            f"(severity={draft.severity}, mode={draft.mode_id}) ===",
        )
        print_fn(f"Title: {edited.title}")
        print_fn("Body preview:\n" + _shorten(edited.body, 500))
        if confirm_fn(f"Post draft for cluster {draft.cluster_id}?"):
            try:
                issue = await tracker.create_issue(edited)
            except TrackerError as e:
                print_fn(f"  ERROR posting cluster {draft.cluster_id}: {e}")
                # A tracker failure is not an operator rejection; record it
                # distinctly so the outcome summary doesn't conflate the two.
                results.append(ReviewOutcome(draft=edited, action="post_failed"))
                continue
            results.append(
                ReviewOutcome(draft=edited, action="posted", posted_issue=issue),
            )
        else:
            results.append(ReviewOutcome(draft=edited, action="rejected"))
    return results


def _edit_draft(
    draft: IssueDraft,
    *,
    editor_cmd: str,
    print_fn: Callable[[str], None],
) -> IssueDraft | None:
    """Open the draft in `$EDITOR` (if set) and re-parse the edited content.

    Returns the (possibly-edited) draft, or None if the operator killed the
    editor session (non-zero exit). When `editor_cmd` is empty we skip the
    editor step and return the draft unchanged so the operator can still
    accept-or-reject via the confirm prompt.
    """
    if not editor_cmd:
        return draft
    rendered = draft.to_markdown()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"-{draft.cluster_id}.md",
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(rendered)
        tmp_path = Path(tf.name)
    try:
        argv = shlex.split(editor_cmd) + [str(tmp_path)]
        try:
            result = subprocess.run(argv, check=False)  # noqa: S603  -- operator-chosen editor
        except FileNotFoundError:
            print_fn(
                f"  $EDITOR {editor_cmd!r} not found on PATH; using the draft as-is.",
            )
            return draft
        if result.returncode != 0:
            print_fn(
                f"  Editor exited with status {result.returncode}; skipping this draft.",
            )
            return None
        edited_text = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
    return _apply_markdown_edits(draft, edited_text)


_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_DESCRIPTION_RE = re.compile(
    r"##\s+Description\s*\n+(.+?)(?=\n##\s+|\Z)",
    re.DOTALL,
)


def _apply_markdown_edits(draft: IssueDraft, text: str) -> IssueDraft:
    """Re-parse the edited markdown and update title + body on the draft.

    The drafter writes `IssueDraft.to_markdown()` which has predictable
    structure: a `# <title>` heading, a metadata table, and a `##
    Description` section that holds the body. We extract those two; the
    provenance HTML comment is preserved (re-appended on the body if the
    operator removed it).
    """
    title_match = _TITLE_RE.search(text)
    description_match = _DESCRIPTION_RE.search(text)
    new_title = title_match.group(1).strip() if title_match else draft.title
    new_body = description_match.group(1).strip() if description_match else draft.body
    # Re-attach provenance if it was stripped by the operator.
    if IssueProvenance.parse_from_body(new_body) is None:
        prov = IssueProvenance.parse_from_body(draft.body)
        if prov is not None:
            new_body = f"{new_body}\n\n{prov.to_html_comment()}"
    return draft.model_copy(update={"title": new_title, "body": new_body})


def summarize_review_outcomes(outcomes: list[ReviewOutcome]) -> str:
    """Markdown summary of a review pass, appended to the printed report.

    The run report is rendered before `--review` runs, so without this the
    printed output would never reflect what the operator actually posted.
    """
    if not outcomes:
        return ""
    lines = ["", "## Review outcomes", ""]
    for o in outcomes:
        detail = ""
        if o.action == "posted" and o.posted_issue is not None:
            detail = f" -> {o.posted_issue.url or o.posted_issue.id}"
        lines.append(f"- `{o.draft.cluster_id}` [{o.draft.severity}] {o.action}{detail}")
    posted = sum(1 for o in outcomes if o.action == "posted")
    failed = sum(1 for o in outcomes if o.action == "post_failed")
    lines.append("")
    lines.append(
        f"{posted} posted, {failed} failed, {len(outcomes) - posted - failed} "
        "rejected/cancelled (rejected drafts remain in the local queue)."
    )
    return "\n".join(lines)


def _default_confirm(prompt: str) -> bool:
    return click.confirm(prompt, default=False)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated {len(text) - limit} chars)"

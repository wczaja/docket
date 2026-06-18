"""Poster subagent: dedup-aware tracker writeback for drafted issues.

Per design §5.2 / §7 Phase 8, after the drafter produces an `IssueDraft` per
cluster, the poster decides one of five actions per draft:

  - `skipped`: existing issue's provenance already lists every current
    cluster member (full idempotency — re-running with no new traces posts
    nothing).
  - `commented`: existing issue exists for the same `(mode_id,
    rubric_version, cluster_id)` triple — or, when the cluster grew and its
    hash-derived `cluster_id` changed, for the same cluster lineage matched
    by provenance member-trace overlap — and new member trace IDs appeared;
    a comment listing only the diff is posted.
  - `created`: no existing issue and the cluster's severity meets the
    operator-configured `auto_post_threshold`; the draft is posted via
    `tracker.create_issue`.
  - `needs_create`: no existing issue and severity is below the threshold;
    the draft stays in the local queue for interactive `--review` or for
    the operator to inspect manually (`docket` does not auto-post
    below threshold).
  - `failed`: a tracker call (list/search/create/comment) raised
    `TrackerError` for this draft. Per design §4.4 the run is NOT aborted —
    the draft already sits in the local queue (the drafter writes it before
    posting), so it stays replayable; the redacted error is surfaced in the
    run report instead.

The dedup query uses the standard label set
(`docket`, `mode:<id>`, `rubric:<id>@<version>`); the cluster_id
disambiguation comes from the parsed HTML provenance comment.
"""

from dataclasses import dataclass, field
from typing import Literal

from docket.adapters.base import Tracker
from docket.errors import TrackerError
from docket.models.classification import Severity
from docket.models.issue import (
    Issue,
    IssueDraft,
    IssueProvenance,
    make_labels,
)
from docket.observability import redact

AutoPostThreshold = Literal["critical", "high", "medium", "low", "never"]

_SEVERITY_RANK: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

# Cap on trace-ID bullets in a "new members" comment — keeps comments under
# tracker size limits; the overflow is summarized as "…and N more".
_COMMENT_MEMBER_CAP = 50


def meets_auto_post_threshold(severity: Severity, threshold: AutoPostThreshold) -> bool:
    """Return True iff `severity` is at or above `threshold` (design §1.5).

    `threshold='never'` always returns False — the operator opts in to
    auto-post explicitly.
    """
    if threshold == "never":
        return False
    return _SEVERITY_RANK.get(severity, 0) >= _SEVERITY_RANK.get(threshold, 0)


DedupAction = Literal["skipped", "commented", "created", "needs_create", "failed"]


@dataclass(frozen=True)
class DedupOutcome:
    """Per-draft result of the dedup + threshold loop.

    `action` is one of: `skipped`, `commented`, `created`, `needs_create`,
    `failed`. `created_issue` is the post-create `Issue` from the tracker
    (only set when action=='created'); `existing_issue` is the open issue
    found by dedup (set when action=='skipped' or 'commented');
    `failure_reason` is a redacted error summary (only set when
    action=='failed').
    """

    draft: IssueDraft
    action: DedupAction
    existing_issue: Issue | None = None
    created_issue: Issue | None = None
    new_member_trace_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None


async def dedup_drafts(
    drafts: list[IssueDraft],
    *,
    tracker: Tracker,
    auto_post_threshold: AutoPostThreshold = "never",
) -> list[DedupOutcome]:
    """Run the dedup + threshold loop for every draft.

    For each draft: query tracker by labels, match a candidate by
    `cluster_id` from the parsed provenance (falling back to provenance
    member-trace overlap for grown clusters whose hash-derived id changed),
    then:

      - existing match + no new members → `skipped`;
      - existing match + new members → comment, action `commented`;
      - no match + severity meets threshold → create, action `created`;
      - no match + below threshold → `needs_create` (queued for `--review`
        or manual operator action).

    A `TrackerError` from any tracker call is contained per draft (design
    §4.4: tracker write failure must not fail the whole run): the draft's
    outcome records `action='failed'` with a redacted reason and the loop
    continues. An unavailable tracker (e.g. `list_open_issues` failing)
    therefore degrades to every draft being recorded as `failed` — the
    drafts remain in the local queue for replay.

    Returns one `DedupOutcome` per input draft, in input order.
    """
    outcomes: list[DedupOutcome] = []
    for draft in drafts:
        try:
            outcome = await _dedup_one(
                draft,
                tracker=tracker,
                auto_post_threshold=auto_post_threshold,
            )
        except TrackerError as e:
            outcome = DedupOutcome(
                draft=draft,
                action="failed",
                failure_reason=redact(str(e)),
            )
        outcomes.append(outcome)
    return outcomes


async def _dedup_one(
    draft: IssueDraft,
    *,
    tracker: Tracker,
    auto_post_threshold: AutoPostThreshold,
) -> DedupOutcome:
    labels = make_labels(draft.mode_id, draft.rubric_version)
    candidates = await tracker.list_open_issues(filter={"labels": labels})
    match = _find_match_by_cluster_id(candidates, cluster_id=draft.cluster_id)
    if match is None:
        # `cluster_id` is a hash of the member trace IDs, so a grown cluster
        # gets a NEW id. Fall back to matching the cluster *lineage* by
        # provenance member-trace overlap so growth comments instead of
        # creating a duplicate issue.
        match = _find_match_by_member_overlap(candidates, draft=draft)
    if match is None:
        if meets_auto_post_threshold(draft.severity, auto_post_threshold):
            created = await tracker.create_issue(draft)
            return DedupOutcome(draft=draft, action="created", created_issue=created)
        return DedupOutcome(draft=draft, action="needs_create")
    existing_prov = IssueProvenance.parse_from_body(match.body)
    existing_members: set[str] = set(existing_prov.member_trace_ids) if existing_prov else set()
    current_members = set(draft.member_trace_ids)
    new_members = sorted(current_members - existing_members)
    if not new_members:
        return DedupOutcome(draft=draft, action="skipped", existing_issue=match)
    comment = _format_new_members_comment(
        new_members=new_members,
        rubric_version=draft.rubric_version,
        run_id=draft.run_id,
    )
    await tracker.comment_on_issue(match.id, comment)
    return DedupOutcome(
        draft=draft,
        action="commented",
        existing_issue=match,
        new_member_trace_ids=new_members,
    )


def _find_match_by_cluster_id(candidates: list[Issue], *, cluster_id: str) -> Issue | None:
    for issue in candidates:
        prov = IssueProvenance.parse_from_body(issue.body)
        if prov is not None and prov.cluster_id == cluster_id:
            return issue
    return None


def _find_match_by_member_overlap(candidates: list[Issue], *, draft: IssueDraft) -> Issue | None:
    """Match a grown cluster to its existing issue by member-trace overlap.

    Only issues carrying BOTH the `mode:<id>` and `rubric:<id>@<version>`
    labels are considered (the tracker query is label-filtered already, but
    we don't trust every backend to honor it). The labels are built via
    `make_labels` so the comparison uses the same tracker-safe normalization
    (length cap, space replacement) the issues were created with. Candidates
    whose provenance is missing or lists no `member_trace_ids` are skipped —
    without members the lineage is unknowable.
    """
    required_labels = set(make_labels(draft.mode_id, draft.rubric_version)) - {"docket"}
    draft_members = set(draft.member_trace_ids)
    for issue in candidates:
        if not required_labels.issubset(issue.labels):
            continue
        prov = IssueProvenance.parse_from_body(issue.body)
        if prov is None or not prov.member_trace_ids:
            continue
        if draft_members & set(prov.member_trace_ids):
            return issue
    return None


def _format_new_members_comment(
    *,
    new_members: list[str],
    rubric_version: str,
    run_id: str,
) -> str:
    shown = new_members[:_COMMENT_MEMBER_CAP]
    bullets = "\n".join(f"- `{tid}`" for tid in shown)
    if len(new_members) > len(shown):
        bullets += f"\n…and {len(new_members) - len(shown)} more"
    return (
        "**docket observed new cluster members in this failure mode.**\n\n"
        f"Run: `{run_id}`\n"
        f"Rubric: `{rubric_version}`\n\n"
        f"New trace IDs ({len(new_members)}):\n\n{bullets}"
    )

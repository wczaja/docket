"""Poster subagent: dedup-aware tracker writeback for drafted issues.

Per design §5.2 / §7 Phase 8, after the drafter produces an `IssueDraft` per
cluster, the poster decides one of five actions per draft:

  - `skipped`: the matching open issue already accounts for every current
    cluster member (full idempotency — re-running with no new traces posts
    nothing).
  - `commented`: an open issue for the same failure exists and new member
    trace IDs appeared; a comment listing only the diff is posted.
  - `created`: no existing issue and the cluster's severity meets the
    operator-configured `auto_post_threshold`; the draft is posted via
    `tracker.create_issue`.
  - `needs_create`: no existing issue and severity is below the threshold
    — or the tracker's listing was truncated, so "no existing issue" is
    unproven; the draft stays in the local queue for interactive
    `--review` or for the operator to inspect manually (`docket` does not
    auto-post below threshold).
  - `failed`: a tracker call (list/create/update/comment) raised
    `TrackerError` for this draft. Per design §4.4 the run is NOT aborted —
    the draft already sits in the local queue (the drafter writes it before
    posting), so it stays replayable; the redacted error is surfaced in the
    run report instead.

The dedup query uses the standard label set (`docket`, `mode:<id>`,
`rubric:<id>@<version>`); issue identity comes from the parsed HTML
provenance comment. A draft matches an open issue by, in order:

  1. `cluster_id` — the issue was filed for this exact cluster, or has
     already absorbed it (the id is a hash of the member trace IDs);
  2. member overlap — the cluster grew, so its hash-derived id changed,
     but it shares traces with the issue;
  3. lineage — neither: `docket serve` tiles windows exactly, so a failure
     that recurs in the next window arrives with entirely new trace IDs. A
     pluggable `LineageMatcher` decides whether an open issue in the
     draft's (mode, rubric_version) lineage is the same underlying failure;
     the default, `match_mode_lineage`, keeps one open issue per lineage.

Idempotency: before commenting, the poster records the reported cluster id
and trace IDs in the issue's provenance (`Tracker.update_provenance`), so a
re-run of the same window finds its cluster already reported and skips.
Recording first means comments go out at most once: if the comment then
fails, the outcome is `failed` (surfaced in the report) and re-runs skip
rather than retry it into a possible duplicate.

Truncation: a listing cut off at the tracker's page ceiling can't prove
that nothing matches, so a draft that matched nothing is queued as
`needs_create` even when its severity would auto-post. A match found in a
truncated listing is still trusted.
"""

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
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

log = logging.getLogger(__name__)

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

# Tracker keys number issues in creation order (`AGT-10`, GitHub's `10`).
# Zero-padding their digit runs makes `AGT-9` sort before `AGT-10`.
_KEY_DIGITS_RE = re.compile(r"[0-9]+")
_KEY_DIGITS_WIDTH = 20


def meets_auto_post_threshold(severity: Severity, threshold: AutoPostThreshold) -> bool:
    """Return True iff `severity` is at or above `threshold` (design §1.5).

    `threshold='never'` always returns False — the operator opts in to
    auto-post explicitly.
    """
    if threshold == "never":
        return False
    return _SEVERITY_RANK.get(severity, 0) >= _SEVERITY_RANK.get(threshold, 0)


DedupAction = Literal["skipped", "commented", "created", "needs_create", "failed"]
MatchKind = Literal["cluster_id", "member_overlap", "lineage"]


@dataclass(frozen=True)
class DedupOutcome:
    """Per-draft result of the dedup + threshold loop.

    `action` is one of: `skipped`, `commented`, `created`, `needs_create`,
    `failed`. `created_issue` is the post-create `Issue` from the tracker
    (only set when action=='created'); `existing_issue` is the open issue
    found by dedup (set when action=='skipped' or 'commented') and
    `matched_by` says how it was found; `failure_reason` is a redacted
    error summary (only set when action=='failed'). `auto_post_demoted`
    marks a `needs_create` that would have auto-posted had the tracker's
    listing not been truncated.
    """

    draft: IssueDraft
    action: DedupAction
    existing_issue: Issue | None = None
    created_issue: Issue | None = None
    new_member_trace_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    matched_by: MatchKind | None = None
    auto_post_demoted: bool = False


@dataclass(frozen=True)
class LineageCandidate:
    """An open issue in a draft's (mode, rubric_version) lineage.

    The issue carries the draft's `mode:<id>` and `rubric:<id>@<version>`
    labels, and `provenance` (its parsed provenance block) names the same
    mode and rubric version.
    """

    issue: Issue
    provenance: IssueProvenance


# The "same underlying failure?" decision for a draft that matched no open
# issue by cluster_id or member overlap. It receives the draft and the open
# issues in its lineage (never empty) and returns the one to comment on, or
# None to treat the draft as a new failure. Keep it deterministic so
# equivalent runs agree; re-runs stay idempotent regardless, because the
# poster matches an already-reported cluster by the issue's ledger first.
LineageMatcher = Callable[
    [IssueDraft, Sequence[LineageCandidate]],
    Awaitable[LineageCandidate | None],
]


async def match_mode_lineage(
    draft: IssueDraft,
    candidates: Sequence[LineageCandidate],
) -> LineageCandidate | None:
    """Default `LineageMatcher`: one open issue per (mode, rubric_version).

    Every candidate shares the draft's failure mode and rubric version, so
    by the rubric's own definition it is the same failure. When several are
    open (filed before lineage dedup existed, or posted from the queue by
    hand), the oldest by tracker key wins, so recurrences converge on one
    issue deterministically.
    """
    return min(candidates, key=_creation_order, default=None)


def _creation_order(candidate: LineageCandidate) -> str:
    key = candidate.issue.key or candidate.issue.id
    return _KEY_DIGITS_RE.sub(lambda m: m.group().zfill(_KEY_DIGITS_WIDTH), key)


async def dedup_drafts(
    drafts: list[IssueDraft],
    *,
    tracker: Tracker,
    auto_post_threshold: AutoPostThreshold = "never",
    lineage_matcher: LineageMatcher | None = match_mode_lineage,
    window: tuple[datetime, datetime] | None = None,
) -> list[DedupOutcome]:
    """Run the dedup + threshold loop for every draft.

    For each draft: list the open issues carrying its labels (once per
    label set per call), match one by `cluster_id`, member overlap, or
    lineage (`lineage_matcher`; None disables lineage matching, leaving
    cluster-only dedup), then:

      - match already accounts for every member → `skipped`;
      - match + new members → record them in the issue's provenance, then
        comment, action `commented`;
      - no match + severity meets threshold → create, action `created`
        (`needs_create` instead when the listing was truncated);
      - no match + below threshold → `needs_create` (queued for `--review`
        or manual operator action).

    `window` is the run's `(since, until)`, cited by recurrence comments.

    A `TrackerError` from any tracker call is contained per draft (design
    §4.4: tracker write failure must not fail the whole run): the draft's
    outcome records `action='failed'` with a redacted reason and the loop
    continues. An unavailable tracker (e.g. `list_open_issues` failing)
    therefore degrades to every draft being recorded as `failed` — the
    drafts remain in the local queue for replay.

    Returns one `DedupOutcome` per input draft, in input order.
    """
    listings: dict[tuple[str, ...], _Listing] = {}
    outcomes: list[DedupOutcome] = []
    for draft in drafts:
        try:
            outcome = await _dedup_one(
                draft,
                tracker=tracker,
                listings=listings,
                auto_post_threshold=auto_post_threshold,
                lineage_matcher=lineage_matcher,
                window=window,
            )
        except TrackerError as e:
            outcome = DedupOutcome(
                draft=draft,
                action="failed",
                failure_reason=redact(str(e)),
            )
        outcomes.append(outcome)
    return outcomes


@dataclass
class _Listing:
    """One label set's open issues, listed once per `dedup_drafts` call.

    The poster applies its own writes here (issues it created, ledgers it
    updated) so later drafts in the run see them without relying on the
    tracker's read-after-write consistency — Jira's search index lags
    writes.
    """

    issues: list[Issue]
    truncated: bool

    def replace(self, issue: Issue) -> None:
        self.issues = [issue if i.id == issue.id else i for i in self.issues]


@dataclass(frozen=True)
class _Match:
    issue: Issue
    provenance: IssueProvenance
    kind: MatchKind


async def _dedup_one(
    draft: IssueDraft,
    *,
    tracker: Tracker,
    listings: dict[tuple[str, ...], _Listing],
    auto_post_threshold: AutoPostThreshold,
    lineage_matcher: LineageMatcher | None,
    window: tuple[datetime, datetime] | None,
) -> DedupOutcome:
    labels = make_labels(draft.mode_id, draft.rubric_version)
    listing = listings.get(tuple(labels))
    if listing is None:
        fetched = await tracker.list_open_issues_v2(filter={"labels": labels})
        listing = _Listing(issues=list(fetched.issues), truncated=fetched.truncated)
        listings[tuple(labels)] = listing
    match = await _find_match(draft, listing.issues, lineage_matcher=lineage_matcher)
    if match is None:
        if not meets_auto_post_threshold(draft.severity, auto_post_threshold):
            return DedupOutcome(draft=draft, action="needs_create")
        if listing.truncated:
            log.warning(
                "tracker listing for %s was truncated at its page ceiling; queuing "
                "cluster %s for review instead of auto-posting (a duplicate can't be "
                "ruled out)",
                ", ".join(labels),
                draft.cluster_id,
            )
            return DedupOutcome(draft=draft, action="needs_create", auto_post_demoted=True)
        created = await tracker.create_issue(draft)
        # Later drafts in this lineage (sibling clusters) fold into it.
        listing.issues.append(created)
        return DedupOutcome(draft=draft, action="created", created_issue=created)
    prov = match.provenance
    # The id hashes the exact membership, so a ledgered id was reported in
    # full — even if some of its traces have since aged out of the ledger.
    already_reported = draft.cluster_id in prov.reported_cluster_ids
    new_members = (
        [] if already_reported else sorted(set(draft.member_trace_ids) - prov.known_trace_ids)
    )
    if not new_members:
        return DedupOutcome(
            draft=draft,
            action="skipped",
            existing_issue=match.issue,
            matched_by=match.kind,
        )
    updated = prov.with_reported(cluster_id=draft.cluster_id, trace_ids=new_members)
    await tracker.update_provenance(match.issue, updated)
    listing.replace(
        match.issue.model_copy(update={"body": updated.replace_in_body(match.issue.body)})
    )
    if match.kind == "lineage":
        comment = _format_lineage_comment(
            draft=draft,
            provenance=prov,
            new_members=new_members,
            window=window,
        )
    else:
        comment = _format_new_members_comment(
            new_members=new_members,
            rubric_version=draft.rubric_version,
            run_id=draft.run_id,
        )
    try:
        await tracker.comment_on_issue(match.issue.id, comment)
    except TrackerError as e:
        raise TrackerError(
            f"{e} (the new traces are already recorded in the issue's provenance, "
            "so re-runs will skip this comment rather than retry it)"
        ) from e
    return DedupOutcome(
        draft=draft,
        action="commented",
        existing_issue=match.issue,
        new_member_trace_ids=new_members,
        matched_by=match.kind,
    )


async def _find_match(
    draft: IssueDraft,
    candidates: list[Issue],
    *,
    lineage_matcher: LineageMatcher | None,
) -> _Match | None:
    parsed = [(issue, IssueProvenance.parse_from_body(issue.body)) for issue in candidates]
    for issue, prov in parsed:
        if prov is not None and draft.cluster_id in (prov.cluster_id, *prov.reported_cluster_ids):
            return _Match(issue, prov, "cluster_id")
    lineage = [
        LineageCandidate(issue=issue, provenance=prov)
        for issue, prov in parsed
        if prov is not None and _in_lineage(draft, issue, prov)
    ]
    # `cluster_id` is a hash of the member trace IDs, so a grown cluster gets
    # a NEW id. Shared traces identify the issue it grew from.
    members = set(draft.member_trace_ids)
    for candidate in lineage:
        if members & candidate.provenance.known_trace_ids:
            return _Match(candidate.issue, candidate.provenance, "member_overlap")
    if lineage_matcher is None or not lineage:
        return None
    chosen = await lineage_matcher(draft, lineage)
    if chosen is None:
        return None
    return _Match(chosen.issue, chosen.provenance, "lineage")


def _in_lineage(draft: IssueDraft, issue: Issue, prov: IssueProvenance) -> bool:
    """Whether `issue` belongs to the draft's (mode, rubric_version) lineage.

    Requires BOTH the `mode:<id>` and `rubric:<id>@<version>` labels (the
    tracker query is label-filtered already, but we don't trust every
    backend to honor it). The labels are built via `make_labels` so the
    comparison uses the same tracker-safe normalization (length cap, space
    replacement) the issues were created with. The provenance must name the
    same mode and rubric version too.
    """
    required_labels = set(make_labels(draft.mode_id, draft.rubric_version)) - {"docket"}
    return (
        required_labels.issubset(issue.labels)
        and prov.mode_id == draft.mode_id
        and prov.rubric_version == draft.rubric_version
    )


def _format_new_members_comment(
    *,
    new_members: list[str],
    rubric_version: str,
    run_id: str,
) -> str:
    return (
        "**docket observed new cluster members in this failure mode.**\n\n"
        f"Run: `{run_id}`\n"
        f"Rubric: `{rubric_version}`\n\n"
        f"New trace IDs ({len(new_members)}):\n\n{_format_trace_bullets(new_members)}"
    )


def _format_lineage_comment(
    *,
    draft: IssueDraft,
    provenance: IssueProvenance,
    new_members: list[str],
    window: tuple[datetime, datetime] | None,
) -> str:
    span = (
        f"window `{window[0].isoformat()}` → `{window[1].isoformat()}`"
        if window is not None
        else f"run `{draft.run_id}`"
    )
    if draft.run_id == provenance.run_id:
        # A sibling cluster from the run this issue was filed from.
        headline = (
            f"**docket observed another cluster of this failure** in {span}, "
            "the one this issue was filed from."
        )
    else:
        headline = f"**docket observed this failure recur.** Recurred in {span}."
    return (
        f"{headline}\n\n"
        f"Run: `{draft.run_id}`\n"
        f"Rubric: `{draft.rubric_version}`\n"
        f"Cluster: `{draft.cluster_id}` "
        f"(representative trace `{draft.representative_trace_id}`)\n\n"
        f"New trace IDs ({len(new_members)}):\n\n{_format_trace_bullets(new_members)}"
    )


def _format_trace_bullets(trace_ids: list[str]) -> str:
    shown = trace_ids[:_COMMENT_MEMBER_CAP]
    bullets = "\n".join(f"- `{tid}`" for tid in shown)
    if len(trace_ids) > len(shown):
        bullets += f"\n…and {len(trace_ids) - len(shown)} more"
    return bullets

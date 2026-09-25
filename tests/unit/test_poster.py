"""Unit tests for the Poster subagent's dedup loop (Phase 8 §5.2 / §7)."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from docket.adapters.base import Tracker
from docket.agent.subagents.poster import (
    DedupOutcome,
    LineageCandidate,
    dedup_drafts,
    match_mode_lineage,
    meets_auto_post_threshold,
)
from docket.errors import TrackerError
from docket.models.cluster import compute_cluster_id
from docket.models.issue import (
    Issue,
    IssueDraft,
    IssuePatch,
    IssueProvenance,
    make_labels,
)
from docket.models.listing import IssueListing


class _RecordingTracker(Tracker):
    """A scriptable tracker that returns whatever issues you hand it.

    Created issues and body patches persist, so a second `dedup_drafts`
    call against the same instance sees what the first one wrote.
    """

    def __init__(
        self,
        *,
        open_issues_by_filter: list[Issue] | None = None,
        comment_raises: bool = False,
        list_raises: bool = False,
        update_raises: bool = False,
        create_raises_for: set[str] | None = None,
        truncated: bool = False,
    ) -> None:
        self._open_issues = open_issues_by_filter or []
        self._comment_raises = comment_raises
        self._list_raises = list_raises
        self._update_raises = update_raises
        self._create_raises_for = create_raises_for or set()
        self._truncated = truncated
        self.list_calls: list[dict[str, Any] | None] = []
        self.comment_calls: list[tuple[str, str]] = []
        self.create_calls: list[IssueDraft] = []
        self.update_calls: list[tuple[str, IssuePatch]] = []
        # Every write, in order: ("create" | "update" | "comment", issue_id).
        self.writes: list[tuple[str, str]] = []

    async def list_open_issues(self, filter=None):  # type: ignore[no-untyped-def]
        self.list_calls.append(filter)
        if self._list_raises:
            raise TrackerError("simulated list failure: tracker unavailable")
        return list(self._open_issues)

    async def list_open_issues_v2(self, filter=None):  # type: ignore[no-untyped-def]
        issues = await self.list_open_issues(filter)
        return IssueListing(issues=issues, truncated=self._truncated)

    async def search_issues(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def create_issue(self, draft):  # type: ignore[no-untyped-def]
        if draft.cluster_id in self._create_raises_for:
            raise TrackerError(f"simulated create failure for {draft.cluster_id}")
        self.create_calls.append(draft)
        issue = Issue(
            id=f"NEW-{len(self.create_calls)}",
            key=f"NEW-{len(self.create_calls)}",
            title=draft.title,
            body=draft.body,
            labels=draft.labels,
        )
        self._open_issues.append(issue)
        self.writes.append(("create", issue.id))
        return issue

    async def update_issue(self, issue_id, patch):  # type: ignore[no-untyped-def]
        if self._update_raises:
            raise TrackerError("simulated update failure")
        self.update_calls.append((issue_id, patch))
        self.writes.append(("update", issue_id))
        for idx, issue in enumerate(self._open_issues):
            if issue.id == issue_id and patch.body is not None:
                self._open_issues[idx] = issue.model_copy(update={"body": patch.body})
        return Issue(id=issue_id, title="", body="", labels=[])

    async def comment_on_issue(self, issue_id, comment):  # type: ignore[no-untyped-def]
        self.writes.append(("comment", issue_id))
        if self._comment_raises:
            raise TrackerError("simulated comment failure")
        self.comment_calls.append((issue_id, comment))


def _draft(
    *,
    cluster_id: str = "cl-1",
    members: list[str] | None = None,
    mode_id: str = "hallucination",
    rubric_version: str = "agents@1.0.0",
    run_id: str = "r-1",
) -> IssueDraft:
    # Like the drafter's output, the body ends with the provenance block.
    prov = IssueProvenance(
        rubric_version=rubric_version,
        mode_id=mode_id,
        cluster_id=cluster_id,
        representative_trace_id=(members or ["t-1"])[0],
        run_id=run_id,
        member_trace_ids=members or ["t-1", "t-2"],
    )
    return IssueDraft(
        cluster_id=cluster_id,
        mode_id=mode_id,
        rubric_version=rubric_version,
        run_id=run_id,
        severity="high",
        representative_trace_id=(members or ["t-1"])[0],
        member_trace_ids=members or ["t-1", "t-2"],
        title="t",
        body=f"b\n\n{prov.to_html_comment()}",
        labels=make_labels(mode_id, rubric_version),
    )


def _existing_issue(
    *,
    cluster_id: str = "cl-1",
    members: list[str] | None = None,
    mode_id: str = "hallucination",
    rubric_version: str = "agents@1.0.0",
    key: str = "AGT-1",
    reported_cluster_ids: list[str] | None = None,
    reported_trace_ids: list[str] | None = None,
) -> Issue:
    prov = IssueProvenance(
        rubric_version=rubric_version,
        mode_id=mode_id,
        cluster_id=cluster_id,
        representative_trace_id=(members or ["t-1"])[0],
        run_id="r-prior",
        member_trace_ids=members or ["t-1", "t-2"],
        reported_cluster_ids=reported_cluster_ids or [],
        reported_trace_ids=reported_trace_ids or [],
    )
    body = f"original body\n\n{prov.to_html_comment()}"
    return Issue(
        id=key,
        key=key,
        title="Existing",
        body=body,
        labels=make_labels(mode_id, rubric_version),
    )


async def test_dedup_defaults_to_needs_create_when_threshold_never() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [_draft()]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert len(outcomes) == 1
    assert outcomes[0].action == "needs_create"
    assert outcomes[0].existing_issue is None
    # Default threshold='never' → no auto-post.
    assert tracker.create_calls == []


async def test_dedup_queries_with_correct_labels() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    await dedup_drafts([_draft()], tracker=tracker)
    assert tracker.list_calls[0] == {
        "labels": [
            "docket",
            "mode:hallucination",
            "rubric:agents@1.0.0",
        ]
    }


async def test_dedup_skips_when_existing_issue_has_all_members() -> None:
    """Same cluster, same members across runs → no comment, no create."""
    existing = _existing_issue(members=["t-1", "t-2"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(members=["t-1", "t-2"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert outcomes[0].action == "skipped"
    assert outcomes[0].existing_issue == existing
    assert tracker.comment_calls == []


async def test_dedup_comments_with_new_members_when_cluster_grew() -> None:
    """Same cluster ID but one new member → post a comment listing it."""
    existing = _existing_issue(members=["t-1", "t-2"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(members=["t-1", "t-2", "t-3"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert outcomes[0].action == "commented"
    assert outcomes[0].existing_issue == existing
    assert outcomes[0].new_member_trace_ids == ["t-3"]
    assert len(tracker.comment_calls) == 1
    issue_id, body = tracker.comment_calls[0]
    assert issue_id == "AGT-1"
    assert "t-3" in body
    # t-1/t-2 are NOT mentioned (idempotent — only the diff goes in).
    assert "t-1" not in body
    assert "t-2" not in body


async def test_dedup_comment_caps_listed_members_at_50() -> None:
    """A huge diff comment lists at most 50 trace IDs plus an overflow line."""
    existing = _existing_issue(members=["t-old"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    new_ids = [f"t-new-{i:03d}" for i in range(60)]
    drafts = [_draft(members=["t-old", *new_ids])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert outcomes[0].action == "commented"
    _issue_id, body = tracker.comment_calls[0]
    listed = [line for line in body.splitlines() if line.startswith("- `")]
    assert len(listed) == 50
    assert "…and 10 more" in body
    assert "New trace IDs (60)" in body


async def test_dedup_without_lineage_matcher_treats_disjoint_clusters_as_distinct() -> None:
    """`lineage_matcher=None` restores cluster-only dedup: a cluster sharing
    labels (mode + rubric) but not its cluster_id or traces is a new issue."""
    existing = _existing_issue(cluster_id="cl-1", members=["t-1"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(cluster_id="cl-2", members=["t-9"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker, lineage_matcher=None)
    assert outcomes[0].action == "needs_create"
    assert tracker.comment_calls == []


async def test_dedup_recurrence_in_next_window_comments_instead_of_duplicating() -> None:
    """`serve` tiles windows exactly, so a failure that recurs in the next
    window arrives with entirely new trace IDs: its hash-derived cluster_id
    differs and no member overlaps. The open issue for the same (mode,
    rubric_version) is still the same underlying failure — dedup must
    comment on it, not queue (or auto-post) a duplicate."""
    window_1 = ["w1-a", "w1-b", "w1-c"]
    window_2 = ["w2-a", "w2-b", "w2-c"]
    existing = _existing_issue(
        cluster_id=compute_cluster_id("hallucination", window_1),
        members=window_1,
    )
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    draft = _draft(cluster_id=compute_cluster_id("hallucination", window_2), members=window_2)

    outcomes = await dedup_drafts([draft], tracker=tracker)

    assert outcomes[0].action == "commented"
    assert outcomes[0].existing_issue == existing
    assert outcomes[0].new_member_trace_ids == window_2
    assert tracker.create_calls == []


def _windowed_drafts() -> tuple[Issue, IssueDraft]:
    """An issue filed for window 1's cluster and window 2's recurrence."""
    window_1 = ["w1-a", "w1-b", "w1-c"]
    window_2 = ["w2-a", "w2-b", "w2-c"]
    existing = _existing_issue(
        cluster_id=compute_cluster_id("hallucination", window_1),
        members=window_1,
    )
    draft = _draft(
        cluster_id=compute_cluster_id("hallucination", window_2),
        members=window_2,
        run_id="r-window-2",
    )
    return existing, draft


def _existing_cluster_id(issue: Issue) -> str:
    prov = IssueProvenance.parse_from_body(issue.body)
    assert prov is not None
    return prov.cluster_id


_WINDOW_2 = (
    datetime(2026, 9, 25, 11, tzinfo=UTC),
    datetime(2026, 9, 25, 12, tzinfo=UTC),
)


async def test_dedup_recurrence_with_auto_post_does_not_file_a_duplicate() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    outcomes = await dedup_drafts([draft], tracker=tracker, auto_post_threshold="low")
    assert outcomes[0].action == "commented"
    assert outcomes[0].matched_by == "lineage"
    assert tracker.create_calls == []


async def test_dedup_recurrence_comment_cites_window_run_and_cluster() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    await dedup_drafts([draft], tracker=tracker, window=_WINDOW_2)
    ((issue_id, body),) = tracker.comment_calls
    assert issue_id == "AGT-1"
    assert "Recurred in window `2026-09-25T11:00:00+00:00` → `2026-09-25T12:00:00+00:00`" in body
    assert "Run: `r-window-2`" in body
    assert f"Cluster: `{draft.cluster_id}` (representative trace `w2-a`)" in body
    assert "New trace IDs (3)" in body
    for tid in draft.member_trace_ids:
        assert f"- `{tid}`" in body
    # Window 1's traces are already on the issue; they are not repeated.
    assert "w1-a" not in body


async def test_dedup_recurrence_comment_without_window_cites_the_run() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    await dedup_drafts([draft], tracker=tracker)
    ((_issue_id, body),) = tracker.comment_calls
    assert "Recurred in run `r-window-2`" in body


async def test_dedup_rerunning_a_recurrence_window_posts_no_second_comment() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    first = await dedup_drafts([draft], tracker=tracker, auto_post_threshold="low")
    rerun = await dedup_drafts([draft], tracker=tracker, auto_post_threshold="low")
    assert first[0].action == "commented"
    assert rerun[0].action == "skipped"
    assert rerun[0].matched_by == "cluster_id"
    assert len(tracker.comment_calls) == 1
    assert tracker.create_calls == []


async def test_dedup_records_the_ledger_before_commenting() -> None:
    """The provenance write goes first, so a comment is posted at most once."""
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    await dedup_drafts([draft], tracker=tracker)
    assert tracker.writes == [("update", "AGT-1"), ("comment", "AGT-1")]
    ((_issue_id, patch),) = tracker.update_calls
    assert patch.body is not None
    assert patch.body.startswith("original body\n\n")
    prov = IssueProvenance.parse_from_body(patch.body)
    assert prov is not None
    assert prov.cluster_id == _existing_cluster_id(existing)
    assert prov.reported_cluster_ids == [draft.cluster_id]
    assert prov.reported_trace_ids == draft.member_trace_ids


async def test_dedup_ledger_write_failure_posts_no_comment() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing], update_raises=True)
    outcomes = await dedup_drafts([draft], tracker=tracker)
    assert outcomes[0].action == "failed"
    assert outcomes[0].failure_reason is not None
    assert "simulated update failure" in outcomes[0].failure_reason
    assert [kind for kind, _ in tracker.writes] == []


async def test_dedup_comment_failure_after_recording_is_not_retried() -> None:
    """At-most-once: once recorded, a re-run skips instead of re-commenting."""
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing], comment_raises=True)
    first = await dedup_drafts([draft], tracker=tracker)
    rerun = await dedup_drafts([draft], tracker=tracker)
    assert first[0].action == "failed"
    assert first[0].failure_reason is not None
    assert "simulated comment failure" in first[0].failure_reason
    assert "re-runs will skip this comment" in first[0].failure_reason
    assert rerun[0].action == "skipped"
    assert [kind for kind, _ in tracker.writes] == ["update", "comment"]


async def test_dedup_skips_a_ledgered_cluster_even_after_its_traces_age_out() -> None:
    """The cluster ledger outlives the capped trace ledger: an id recorded
    as reported means that exact membership was reported in full."""
    existing, draft = _windowed_drafts()
    aged_out = _existing_issue(
        cluster_id=_existing_cluster_id(existing),
        members=["w1-a", "w1-b", "w1-c"],
        reported_cluster_ids=[draft.cluster_id],
        reported_trace_ids=["w9-newer"],
    )
    tracker = _RecordingTracker(open_issues_by_filter=[aged_out])
    outcomes = await dedup_drafts([draft], tracker=tracker)
    assert outcomes[0].action == "skipped"
    assert tracker.writes == []


async def test_dedup_rerunning_a_grown_cluster_does_not_repeat_its_comment() -> None:
    """Growth is recorded too, so re-running the window that grew the
    cluster skips instead of re-listing the same new traces."""
    existing = _existing_issue(
        cluster_id=compute_cluster_id("hallucination", ["t-1", "t-2"]),
        members=["t-1", "t-2"],
    )
    grown = _draft(
        cluster_id=compute_cluster_id("hallucination", ["t-1", "t-2", "t-3"]),
        members=["t-1", "t-2", "t-3"],
    )
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    first = await dedup_drafts([grown], tracker=tracker)
    rerun = await dedup_drafts([grown], tracker=tracker)
    assert first[0].action == "commented"
    assert first[0].matched_by == "member_overlap"
    assert rerun[0].action == "skipped"
    assert len(tracker.comment_calls) == 1


async def test_dedup_further_growth_comments_only_unreported_traces() -> None:
    existing = _existing_issue(
        cluster_id=compute_cluster_id("hallucination", ["t-1", "t-2"]),
        members=["t-1", "t-2"],
    )
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    for members in (["t-1", "t-2", "t-3"], ["t-1", "t-2", "t-3", "t-4"]):
        cid = compute_cluster_id("hallucination", members)
        await dedup_drafts([_draft(cluster_id=cid, members=members)], tracker=tracker)
    assert len(tracker.comment_calls) == 2
    _issue_id, second = tracker.comment_calls[1]
    assert "- `t-4`" in second
    assert "t-3" not in second


async def test_dedup_sibling_clusters_in_one_run_fold_into_the_created_issue() -> None:
    """One open issue per lineage: once the first cluster of a mode is filed,
    a second (disjoint) cluster from the same run comments on it."""
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [
        _draft(cluster_id="cl-a", members=["a-1", "a-2"]),
        _draft(cluster_id="cl-b", members=["b-1", "b-2"]),
    ]
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="low")
    assert [o.action for o in outcomes] == ["created", "commented"]
    assert outcomes[1].matched_by == "lineage"
    assert outcomes[0].created_issue is not None
    assert outcomes[1].existing_issue is not None
    assert outcomes[1].existing_issue.id == outcomes[0].created_issue.id
    assert len(tracker.create_calls) == 1
    ((_issue_id, body),) = tracker.comment_calls
    assert "another cluster of this failure" in body


async def test_dedup_lists_each_label_set_once_per_call() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [
        _draft(cluster_id="cl-a", members=["a-1"]),
        _draft(cluster_id="cl-b", members=["b-1"]),
        _draft(cluster_id="cl-c", members=["c-1"], mode_id="other-mode"),
    ]
    await dedup_drafts(drafts, tracker=tracker)
    assert len(tracker.list_calls) == 2


async def test_dedup_lineage_prefers_the_oldest_open_issue() -> None:
    """Several open issues in one lineage (e.g. duplicates filed before
    lineage dedup): recurrences converge on the oldest by tracker key."""
    newer = _existing_issue(cluster_id="cl-x", members=["x-1"], key="AGT-10")
    older = _existing_issue(cluster_id="cl-y", members=["y-1"], key="AGT-9")
    tracker = _RecordingTracker(open_issues_by_filter=[newer, older])
    outcomes = await dedup_drafts([_draft(cluster_id="cl-z", members=["z-1"])], tracker=tracker)
    assert outcomes[0].existing_issue == older
    assert tracker.comment_calls[0][0] == "AGT-9"


async def test_dedup_lineage_requires_provenance_for_the_same_mode() -> None:
    """Labels alone don't place an issue in the lineage: its provenance must
    name the draft's mode and rubric version too."""
    mislabeled = _existing_issue(cluster_id="cl-x", members=["x-1"], mode_id="other-mode")
    mislabeled = mislabeled.model_copy(
        update={"labels": make_labels("hallucination", "agents@1.0.0")}
    )
    tracker = _RecordingTracker(open_issues_by_filter=[mislabeled])
    outcomes = await dedup_drafts([_draft(cluster_id="cl-z", members=["z-1"])], tracker=tracker)
    assert outcomes[0].action == "needs_create"
    assert tracker.writes == []


async def test_dedup_consults_a_custom_lineage_matcher() -> None:
    existing, draft = _windowed_drafts()
    seen: list[tuple[IssueDraft, list[LineageCandidate]]] = []

    async def never_same(
        d: IssueDraft, candidates: Sequence[LineageCandidate]
    ) -> LineageCandidate | None:
        seen.append((d, list(candidates)))
        return None

    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    outcomes = await dedup_drafts([draft], tracker=tracker, lineage_matcher=never_same)
    assert outcomes[0].action == "needs_create"
    ((seen_draft, candidates),) = seen
    assert seen_draft == draft
    assert [c.issue for c in candidates] == [existing]
    assert candidates[0].provenance.cluster_id == _existing_cluster_id(existing)

    async def always_first(
        d: IssueDraft, candidates: Sequence[LineageCandidate]
    ) -> LineageCandidate | None:
        return candidates[0]

    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    outcomes = await dedup_drafts([draft], tracker=tracker, lineage_matcher=always_first)
    assert outcomes[0].action == "commented"
    assert outcomes[0].existing_issue == existing


async def test_match_mode_lineage_returns_none_without_candidates() -> None:
    assert await match_mode_lineage(_draft(), []) is None


# -- truncated tracker listings ---------------------------------------------


async def test_dedup_truncated_listing_demotes_auto_post_to_needs_create() -> None:
    """Issues past the page ceiling were never seen, so "no match" is
    unproven: queue for review instead of risking a duplicate."""
    tracker = _RecordingTracker(open_issues_by_filter=[], truncated=True)
    outcomes = await dedup_drafts([_draft()], tracker=tracker, auto_post_threshold="low")
    assert outcomes[0].action == "needs_create"
    assert outcomes[0].auto_post_demoted is True
    assert tracker.create_calls == []


async def test_dedup_truncated_listing_still_trusts_a_found_match() -> None:
    existing, draft = _windowed_drafts()
    tracker = _RecordingTracker(open_issues_by_filter=[existing], truncated=True)
    outcomes = await dedup_drafts([draft], tracker=tracker, auto_post_threshold="low")
    assert outcomes[0].action == "commented"
    assert outcomes[0].auto_post_demoted is False


async def test_dedup_truncated_listing_below_threshold_is_plain_needs_create() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[], truncated=True)
    outcomes = await dedup_drafts([_draft()], tracker=tracker)
    assert outcomes[0].action == "needs_create"
    assert outcomes[0].auto_post_demoted is False


async def test_dedup_comments_when_grown_cluster_gets_new_hash_id() -> None:
    """cluster_id is a hash of members, so a grown cluster has a NEW id;
    dedup must still find the lineage via member overlap and comment."""
    existing = _existing_issue(cluster_id="hash-of-t1-t2", members=["t-1", "t-2"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(cluster_id="hash-of-t1-t2-t3", members=["t-1", "t-2", "t-3"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="low")
    assert outcomes[0].action == "commented"
    assert outcomes[0].existing_issue == existing
    assert outcomes[0].new_member_trace_ids == ["t-3"]
    # No duplicate issue was created despite the permissive threshold.
    assert tracker.create_calls == []
    issue_id, body = tracker.comment_calls[0]
    assert issue_id == "AGT-1"
    assert "t-3" in body
    assert "t-1" not in body


async def test_dedup_overlap_fallback_requires_mode_and_rubric_labels() -> None:
    """An overlapping issue missing the rubric label is not lineage-matched."""
    existing = _existing_issue(cluster_id="old-id", members=["t-1", "t-2"])
    unlabeled = existing.model_copy(update={"labels": ["docket", "mode:hallucination"]})
    tracker = _RecordingTracker(open_issues_by_filter=[unlabeled])
    drafts = [_draft(cluster_id="new-id", members=["t-1", "t-2", "t-3"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert outcomes[0].action == "needs_create"
    assert tracker.comment_calls == []


async def test_dedup_memberless_provenance_matches_by_lineage_not_overlap() -> None:
    """Provenance lacking member_trace_ids can't be matched by trace overlap,
    but the issue still anchors its (mode, rubric) lineage — so dedup
    reaches it that way, and cluster-only dedup has nothing to match on."""
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="old-id",
        representative_trace_id="t-1",
        run_id="r-prior",
        member_trace_ids=[],
    )
    memberless = Issue(
        id="AGT-1",
        title="Existing",
        body=f"original body\n\n{prov.to_html_comment()}",
        labels=make_labels("hallucination", "agents@1.0.0"),
    )
    tracker = _RecordingTracker(open_issues_by_filter=[memberless])
    drafts = [_draft(cluster_id="new-id", members=["t-1", "t-2"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert outcomes[0].action == "commented"
    assert outcomes[0].matched_by == "lineage"

    tracker = _RecordingTracker(open_issues_by_filter=[memberless])
    outcomes = await dedup_drafts(drafts, tracker=tracker, lineage_matcher=None)
    assert outcomes[0].action == "needs_create"
    assert tracker.comment_calls == []


async def test_dedup_returns_one_outcome_per_draft_in_order() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [
        _draft(cluster_id="a"),
        _draft(cluster_id="b"),
        _draft(cluster_id="c"),
    ]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    assert [o.draft.cluster_id for o in outcomes] == ["a", "b", "c"]


async def test_dedup_ignores_existing_issue_with_unparseable_provenance() -> None:
    """Existing issue without a valid provenance comment should not match."""
    broken = Issue(
        id="AGT-1",
        title="broken",
        body="no provenance here at all",
        labels=make_labels("hallucination", "agents@1.0.0"),
    )
    tracker = _RecordingTracker(open_issues_by_filter=[broken])
    outcomes = await dedup_drafts([_draft()], tracker=tracker)
    assert outcomes[0].action == "needs_create"


async def test_dedup_records_failed_outcome_on_comment_failure() -> None:
    """A TrackerError on comment must NOT abort the run (design §4.4)."""
    existing = _existing_issue(members=["t-1"])
    tracker = _RecordingTracker(
        open_issues_by_filter=[existing],
        comment_raises=True,
    )
    outcomes = await dedup_drafts([_draft(members=["t-1", "t-2"])], tracker=tracker)
    assert len(outcomes) == 1
    assert outcomes[0].action == "failed"
    assert outcomes[0].failure_reason is not None
    assert "simulated comment failure" in outcomes[0].failure_reason


async def test_dedup_create_failure_does_not_block_subsequent_drafts() -> None:
    """One draft's create failure is contained; the next draft still posts."""
    tracker = _RecordingTracker(
        open_issues_by_filter=[],
        create_raises_for={"cl-a"},
    )
    drafts = [_draft(cluster_id="cl-a"), _draft(cluster_id="cl-b", members=["t-9"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="low")
    assert [o.action for o in outcomes] == ["failed", "created"]
    assert outcomes[0].failure_reason is not None
    assert "cl-a" in outcomes[0].failure_reason
    # Draft 2 was still created despite draft 1's failure.
    assert len(tracker.create_calls) == 1
    assert tracker.create_calls[0].cluster_id == "cl-b"


async def test_dedup_records_all_drafts_failed_when_tracker_unavailable() -> None:
    """list_open_issues failing degrades gracefully: every draft is `failed`."""
    tracker = _RecordingTracker(list_raises=True)
    drafts = [_draft(cluster_id="cl-a"), _draft(cluster_id="cl-b")]
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="low")
    assert [o.action for o in outcomes] == ["failed", "failed"]
    for outcome in outcomes:
        assert outcome.failure_reason is not None
        assert "tracker unavailable" in outcome.failure_reason
    assert tracker.create_calls == []


def test_dedup_outcome_is_immutable() -> None:
    import dataclasses

    outcome = DedupOutcome(draft=_draft(), action="skipped")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.action = "commented"  # type: ignore[misc]


# -- auto-post threshold ----------------------------------------------------


def test_meets_threshold_critical_severity_against_high_passes() -> None:
    assert meets_auto_post_threshold("critical", "high")


def test_meets_threshold_exact_match_passes() -> None:
    assert meets_auto_post_threshold("medium", "medium")


def test_meets_threshold_below_fails() -> None:
    assert not meets_auto_post_threshold("low", "high")


def test_meets_threshold_never_always_fails() -> None:
    assert not meets_auto_post_threshold("critical", "never")


async def test_dedup_auto_posts_when_threshold_met() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [_draft()]  # severity="high"
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="medium")
    assert outcomes[0].action == "created"
    assert outcomes[0].created_issue is not None
    assert len(tracker.create_calls) == 1


async def test_dedup_does_not_auto_post_when_below_threshold() -> None:
    tracker = _RecordingTracker(open_issues_by_filter=[])
    drafts = [_draft()]  # severity="high"
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="critical")
    assert outcomes[0].action == "needs_create"
    assert tracker.create_calls == []


async def test_dedup_threshold_does_not_override_existing_issue_match() -> None:
    """Even with a permissive threshold, an existing match → comment, not create."""
    existing = _existing_issue(members=["t-1"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(members=["t-1", "t-2"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker, auto_post_threshold="low")
    assert outcomes[0].action == "commented"
    assert tracker.create_calls == []

"""Unit tests for the Poster subagent's dedup loop (Phase 8 §5.2 / §7)."""

from typing import Any

import pytest

from docket.adapters.base import Tracker
from docket.agent.subagents.poster import (
    DedupOutcome,
    dedup_drafts,
    meets_auto_post_threshold,
)
from docket.errors import TrackerError
from docket.models.issue import (
    Issue,
    IssueDraft,
    IssuePatch,
    IssueProvenance,
    make_labels,
)


class _RecordingTracker(Tracker):
    """A scriptable tracker that returns whatever issues you hand it."""

    def __init__(
        self,
        *,
        open_issues_by_filter: list[Issue] | None = None,
        comment_raises: bool = False,
        list_raises: bool = False,
        create_raises_for: set[str] | None = None,
    ) -> None:
        self._open_issues = open_issues_by_filter or []
        self._comment_raises = comment_raises
        self._list_raises = list_raises
        self._create_raises_for = create_raises_for or set()
        self.list_calls: list[dict[str, Any] | None] = []
        self.comment_calls: list[tuple[str, str]] = []
        self.create_calls: list[IssueDraft] = []
        self.update_calls: list[tuple[str, IssuePatch]] = []

    async def list_open_issues(self, filter=None):  # type: ignore[no-untyped-def]
        self.list_calls.append(filter)
        if self._list_raises:
            raise TrackerError("simulated list failure: tracker unavailable")
        return list(self._open_issues)

    async def search_issues(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def create_issue(self, draft):  # type: ignore[no-untyped-def]
        if draft.cluster_id in self._create_raises_for:
            raise TrackerError(f"simulated create failure for {draft.cluster_id}")
        self.create_calls.append(draft)
        return Issue(id="new", title=draft.title, body=draft.body, labels=draft.labels)

    async def update_issue(self, issue_id, patch):  # type: ignore[no-untyped-def]
        self.update_calls.append((issue_id, patch))
        return Issue(id=issue_id, title="", body="", labels=[])

    async def comment_on_issue(self, issue_id, comment):  # type: ignore[no-untyped-def]
        if self._comment_raises:
            raise TrackerError("simulated comment failure")
        self.comment_calls.append((issue_id, comment))


def _draft(
    *,
    cluster_id: str = "cl-1",
    members: list[str] | None = None,
    mode_id: str = "hallucination",
    rubric_version: str = "agents@1.0.0",
) -> IssueDraft:
    return IssueDraft(
        cluster_id=cluster_id,
        mode_id=mode_id,
        rubric_version=rubric_version,
        run_id="r-1",
        severity="high",
        representative_trace_id=(members or ["t-1"])[0],
        member_trace_ids=members or ["t-1", "t-2"],
        title="t",
        body="b",
        labels=make_labels(mode_id, rubric_version),
    )


def _existing_issue(
    *,
    cluster_id: str = "cl-1",
    members: list[str] | None = None,
    mode_id: str = "hallucination",
    rubric_version: str = "agents@1.0.0",
) -> Issue:
    prov = IssueProvenance(
        rubric_version=rubric_version,
        mode_id=mode_id,
        cluster_id=cluster_id,
        representative_trace_id=(members or ["t-1"])[0],
        run_id="r-prior",
        member_trace_ids=members or ["t-1", "t-2"],
    )
    body = f"original body\n\n{prov.to_html_comment()}"
    return Issue(
        id="AGT-1",
        key="AGT-1",
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


async def test_dedup_treats_different_cluster_ids_as_distinct_issues() -> None:
    """Two clusters can share labels (mode + rubric) but different cluster_ids."""
    # Existing issue is for cluster cl-1, but the draft is for cl-2.
    existing = _existing_issue(cluster_id="cl-1", members=["t-1"])
    tracker = _RecordingTracker(open_issues_by_filter=[existing])
    drafts = [_draft(cluster_id="cl-2", members=["t-9"])]
    outcomes = await dedup_drafts(drafts, tracker=tracker)
    # No cluster_id match -> treated as a new issue.
    assert outcomes[0].action == "needs_create"
    assert tracker.comment_calls == []


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


async def test_dedup_overlap_fallback_skips_provenance_without_members() -> None:
    """Provenance lacking member_trace_ids → lineage unknowable → no match."""
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

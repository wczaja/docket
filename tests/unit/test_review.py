"""Tests for the `--review` mode flow (design §7 Phase 8 PR 3)."""

from pathlib import Path

import pytest

from docket.adapters.base import Tracker
from docket.agent.review import (
    ReviewOutcome,
    _apply_markdown_edits,
    review_and_post,
)
from docket.agent.subagents.poster import DedupOutcome
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssueProvenance, make_labels


class _RecordingTracker(Tracker):
    def __init__(
        self,
        *,
        create_raises: bool = False,
    ) -> None:
        self.create_calls: list[IssueDraft] = []
        self._create_raises = create_raises

    async def list_open_issues(self, filter=None):  # type: ignore[no-untyped-def]
        return []

    async def search_issues(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def create_issue(self, draft):  # type: ignore[no-untyped-def]
        if self._create_raises:
            raise TrackerError("simulated create failure")
        self.create_calls.append(draft)
        return Issue(
            id="ISSUE-1",
            key="AGT-1",
            url="https://example.atlassian.net/browse/AGT-1",
            title=draft.title,
            body=draft.body,
            labels=draft.labels,
        )

    async def update_issue(self, issue_id, patch):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def comment_on_issue(self, issue_id, comment):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _make_draft() -> IssueDraft:
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="cl-1",
        representative_trace_id="t-1",
        run_id="r-1",
        member_trace_ids=["t-1"],
    )
    return IssueDraft(
        cluster_id="cl-1",
        mode_id="hallucination",
        rubric_version="agents@1.0.0",
        run_id="r-1",
        severity="high",
        representative_trace_id="t-1",
        member_trace_ids=["t-1"],
        title="Original title",
        body=f"Original body\n\n{prov.to_html_comment()}",
        labels=make_labels("hallucination", "agents@1.0.0"),
    )


def _needs_create(draft: IssueDraft) -> DedupOutcome:
    return DedupOutcome(draft=draft, action="needs_create")


def _skipped(draft: IssueDraft) -> DedupOutcome:
    return DedupOutcome(draft=draft, action="skipped")


# -- happy path -------------------------------------------------------------


async def test_review_posts_accepted_drafts() -> None:
    tracker = _RecordingTracker()
    draft = _make_draft()
    outputs: list[str] = []

    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor="",  # skip editor, just confirm
        confirm=lambda _prompt: True,
        print_fn=outputs.append,
    )
    assert len(results) == 1
    assert results[0].action == "posted"
    assert results[0].posted_issue is not None
    assert results[0].posted_issue.key == "AGT-1"
    assert len(tracker.create_calls) == 1


async def test_review_skips_rejected_drafts() -> None:
    tracker = _RecordingTracker()
    draft = _make_draft()
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor="",
        confirm=lambda _prompt: False,
        print_fn=lambda _msg: None,
    )
    assert results[0].action == "rejected"
    assert tracker.create_calls == []


async def test_review_ignores_non_needs_create_outcomes() -> None:
    tracker = _RecordingTracker()
    draft = _make_draft()
    results = await review_and_post(
        [_skipped(draft)],
        tracker=tracker,
        editor="",
        confirm=lambda _prompt: True,
        print_fn=lambda _msg: None,
    )
    assert results == []
    assert tracker.create_calls == []


async def test_review_handles_empty_input() -> None:
    tracker = _RecordingTracker()
    results = await review_and_post(
        [],
        tracker=tracker,
        editor="",
        confirm=lambda _prompt: True,
        print_fn=lambda _msg: None,
    )
    assert results == []


# -- $EDITOR integration ----------------------------------------------------


async def test_review_invokes_editor_when_set(tmp_path: Path) -> None:
    """Use a tiny editor stub that rewrites the title heading in-place."""
    editor_script = tmp_path / "fake_editor.sh"
    # sed -i is not portable (BSD sed requires an argument), so write to a
    # temp file and move it back.
    editor_script.write_text(
        "#!/usr/bin/env bash\n"
        'sed \'s/Original title/Edited title/\' "$1" > "$1.tmp" && mv "$1.tmp" "$1"\n'
    )
    editor_script.chmod(0o755)

    tracker = _RecordingTracker()
    draft = _make_draft()
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor=str(editor_script),
        confirm=lambda _prompt: True,
        print_fn=lambda _msg: None,
    )
    assert results[0].action == "posted"
    assert tracker.create_calls[0].title == "Edited title"


async def test_review_handles_editor_nonzero_exit(tmp_path: Path) -> None:
    """Editor exits non-zero → the draft is reported as edit_cancelled, no post."""
    editor_script = tmp_path / "failing_editor.sh"
    editor_script.write_text("#!/usr/bin/env bash\nexit 1\n")
    editor_script.chmod(0o755)

    tracker = _RecordingTracker()
    draft = _make_draft()
    outputs: list[str] = []
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor=str(editor_script),
        confirm=lambda _prompt: True,
        print_fn=outputs.append,
    )
    assert results[0].action == "edit_cancelled"
    assert tracker.create_calls == []
    assert any("Editor exited" in m for m in outputs)


async def test_review_falls_back_when_editor_binary_missing() -> None:
    """When $EDITOR points at a non-existent binary, we proceed with the unedited draft."""
    tracker = _RecordingTracker()
    draft = _make_draft()
    outputs: list[str] = []
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor="/nonexistent/path/to/editor",
        confirm=lambda _prompt: True,
        print_fn=outputs.append,
    )
    assert results[0].action == "posted"
    assert any("not found on PATH" in m for m in outputs)


async def test_review_uses_environ_editor_when_arg_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDITOR", "")
    tracker = _RecordingTracker()
    draft = _make_draft()
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        confirm=lambda _prompt: True,
        print_fn=lambda _msg: None,
    )
    assert results[0].action == "posted"


async def test_review_with_environ_editor_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    tracker = _RecordingTracker()
    draft = _make_draft()
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        confirm=lambda _prompt: True,
        print_fn=lambda _msg: None,
    )
    assert results[0].action == "posted"


# -- post failure handling --------------------------------------------------


async def test_review_records_failure_when_create_raises() -> None:
    tracker = _RecordingTracker(create_raises=True)
    draft = _make_draft()
    outputs: list[str] = []
    results = await review_and_post(
        [_needs_create(draft)],
        tracker=tracker,
        editor="",
        confirm=lambda _prompt: True,
        print_fn=outputs.append,
    )
    # A tracker failure is recorded distinctly from an operator rejection.
    assert results[0].action == "post_failed"
    assert any("simulated create failure" in m for m in outputs)


# -- markdown edit re-parse -------------------------------------------------


def test_apply_markdown_edits_recovers_title_and_body() -> None:
    draft = _make_draft()
    edited_md = draft.to_markdown().replace("Original title", "New title")
    edited_md = edited_md.replace("Original body", "Updated body")
    new = _apply_markdown_edits(draft, edited_md)
    assert new.title == "New title"
    assert "Updated body" in new.body
    # Provenance is preserved (it lives in body, but to_markdown drops it from
    # the Description section; re-attached below the body re-parse).
    assert "docket:provenance" in new.body


def test_apply_markdown_edits_preserves_provenance_when_operator_strips_it() -> None:
    draft = _make_draft()
    edited_md = "# Some new title\n\n## Description\n\nA shorter body without provenance.\n"
    new = _apply_markdown_edits(draft, edited_md)
    assert new.title == "Some new title"
    assert "shorter body" in new.body
    # Provenance is re-attached from the original draft.
    assert "docket:provenance" in new.body
    parsed = IssueProvenance.parse_from_body(new.body)
    assert parsed is not None
    assert parsed.cluster_id == draft.cluster_id


def test_apply_markdown_edits_keeps_original_when_markdown_unparseable() -> None:
    draft = _make_draft()
    new = _apply_markdown_edits(draft, "totally unrelated text")
    assert new.title == draft.title
    assert new.body == draft.body


def test_review_outcome_is_immutable() -> None:
    import dataclasses

    outcome = ReviewOutcome(draft=_make_draft(), action="posted")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.action = "rejected"  # type: ignore[misc]


def test_summarize_review_outcomes_renders_actions() -> None:
    from docket.agent.review import ReviewOutcome, summarize_review_outcomes
    from docket.models.issue import Issue

    draft = _make_draft()
    outcomes = [
        ReviewOutcome(
            draft=draft,
            action="posted",
            posted_issue=Issue(id="i-1", url="https://t.example/i/1", title="t", body="b"),
        ),
        ReviewOutcome(draft=draft, action="post_failed"),
        ReviewOutcome(draft=draft, action="rejected"),
    ]
    summary = summarize_review_outcomes(outcomes)
    assert "## Review outcomes" in summary
    assert "https://t.example/i/1" in summary
    assert "1 posted, 1 failed, 1 rejected/cancelled" in summary


def test_summarize_review_outcomes_empty_is_empty() -> None:
    from docket.agent.review import summarize_review_outcomes

    assert summarize_review_outcomes([]) == ""

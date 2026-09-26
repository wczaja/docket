"""Tests for the `docket queue` command group."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from docket.adapters.base import Tracker
from docket.agent.subagents.poster import DedupOutcome
from docket.cli import main
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssuePatch, IssueProvenance, make_labels


def _draft(cluster_id: str = "c1") -> IssueDraft:
    return IssueDraft(
        cluster_id=cluster_id,
        mode_id="hallucination",
        rubric_version="agents@1.0.0",
        run_id="run-1",
        severity="high",
        representative_trace_id="t-1",
        member_trace_ids=["t-1", "t-2"],
        title="Synthetic failure",
        body="A synthetic body.",
        labels=make_labels("hallucination", "agents@1.0.0"),
    )


def _write_queue_files(queue_dir: Path, draft: IssueDraft) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / f"{draft.cluster_id}.json").write_text(json.dumps(draft.to_json_record()))
    (queue_dir / f"{draft.cluster_id}.md").write_text(draft.to_markdown())


class _FakeTracker(Tracker):
    def __init__(
        self,
        *,
        fail_ids: set[str] | None = None,
        open_issues: list[Issue] | None = None,
    ) -> None:
        self.created: list[IssueDraft] = []
        self.comments: list[tuple[str, str]] = []
        self.closed = False
        self._fail_ids = fail_ids or set()
        self._open = {i.id: i for i in open_issues or []}

    async def list_open_issues(self, filter: dict[str, Any] | None = None) -> list[Issue]:
        wanted = set((filter or {}).get("labels") or [])
        return [i for i in self._open.values() if wanted.issubset(i.labels)]

    async def search_issues(self, query: str, k: int = 10) -> list[Issue]:
        raise NotImplementedError

    async def create_issue(self, draft: IssueDraft) -> Issue:
        if draft.cluster_id in self._fail_ids:
            raise TrackerError("synthetic create failure")
        self.created.append(draft)
        return Issue(
            id=f"i-{draft.cluster_id}",
            url=f"https://tracker.example/i/{draft.cluster_id}",
            title=draft.title,
            body=draft.body,
        )

    async def update_issue(self, issue_id: str, patch: IssuePatch) -> Issue:
        updated = self._open[issue_id].model_copy(update=patch.model_dump(exclude_none=True))
        self._open[issue_id] = updated
        return updated

    async def comment_on_issue(self, issue_id: str, comment: str) -> None:
        self.comments.append((issue_id, comment))

    async def close(self) -> None:
        self.closed = True


def _open_issue(*, members: list[str]) -> Issue:
    """An open issue already tracking cluster c1, filed by an earlier run."""
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="c1",
        representative_trace_id=members[0],
        run_id="run-0",
        member_trace_ids=members,
    )
    return Issue(
        id="AGT-7",
        key="AGT-7",
        url="https://tracker.example/i/AGT-7",
        title="Synthetic failure",
        body=f"Filed earlier.\n\n{prov.to_html_comment()}",
        labels=make_labels("hallucination", "agents@1.0.0"),
    )


def test_queue_list_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queue", "list", "--queue-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Queue is empty." in result.output


def test_queue_list_shows_drafts(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    _write_queue_files(tmp_path, _draft("c2"))
    runner = CliRunner()
    result = runner.invoke(main, ["queue", "list", "--queue-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "c1" in result.output
    assert "Synthetic failure" in result.output
    assert "2 draft(s) queued." in result.output


def _invoke_post(tmp_path: Path, tracker: _FakeTracker, *extra: str) -> Any:
    runner = CliRunner()
    with patch("docket.cli.build_tracker", return_value=tracker):
        return runner.invoke(
            main,
            ["queue", "post", "--queue-dir", str(tmp_path), "--tracker", "github", *extra],
        )


def test_queue_post_posts_and_retires(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    tracker = _FakeTracker()
    result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 0, result.output
    assert [d.cluster_id for d in tracker.created] == ["c1"]
    assert tracker.closed
    assert "posted c1 -> https://tracker.example/i/c1" in result.output
    assert "1 posted, 0 commented, 0 already tracked, 0 failed, 0 skipped." in result.output
    # Files retired into posted/ with the issue URL recorded.
    assert not (tmp_path / "c1.json").exists()
    record = json.loads((tmp_path / "posted" / "c1.json").read_text())
    assert record["posted_issue_url"] == "https://tracker.example/i/c1"


def test_queue_post_comments_on_open_issue_instead_of_filing_duplicate(tmp_path: Path) -> None:
    # c1's issue already exists; say its comment failed mid-run, leaving the
    # draft queued. Replay adds the unreported trace instead of re-filing.
    _write_queue_files(tmp_path, _draft("c1"))  # members t-1, t-2
    tracker = _FakeTracker(open_issues=[_open_issue(members=["t-1"])])
    result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 0, result.output
    assert tracker.created == []
    ((issue_id, comment),) = tracker.comments
    assert issue_id == "AGT-7"
    assert "`t-2`" in comment
    assert "commented c1 -> https://tracker.example/i/AGT-7 (1 new trace(s))" in result.output
    assert "0 posted, 1 commented, 0 already tracked, 0 failed, 0 skipped." in result.output
    assert not (tmp_path / "c1.json").exists()
    record = json.loads((tmp_path / "posted" / "c1.json").read_text())
    assert record["posted_issue_url"] == "https://tracker.example/i/AGT-7"


def test_queue_post_retires_already_tracked_draft_without_posting(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    tracker = _FakeTracker(open_issues=[_open_issue(members=["t-1", "t-2"])])
    result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 0, result.output
    assert tracker.created == []
    assert tracker.comments == []
    assert "already tracked c1 -> https://tracker.example/i/AGT-7" in result.output
    assert "0 posted, 0 commented, 1 already tracked, 0 failed, 0 skipped." in result.output
    assert not (tmp_path / "c1.json").exists()
    assert (tmp_path / "posted" / "c1.json").exists()


def test_queue_post_keeps_draft_queued_when_dedup_withholds_the_create(tmp_path: Path) -> None:
    # Dedup may decline to create even at the lowest threshold (e.g. when it
    # can't rule out a duplicate); such a draft must stay queued, not retire.
    _write_queue_files(tmp_path, _draft("c1"))

    async def _withheld(drafts: list[IssueDraft], **_kwargs: Any) -> list[DedupOutcome]:
        return [DedupOutcome(draft=d, action="needs_create") for d in drafts]

    tracker = _FakeTracker()
    with patch("docket.agent.subagents.poster.dedup_drafts", new=_withheld):
        result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 0, result.output
    assert "skipped c1: needs_create, left queued" in result.output
    assert "0 posted, 0 commented, 0 already tracked, 0 failed, 1 skipped." in result.output
    assert (tmp_path / "c1.json").exists()
    assert not (tmp_path / "posted").exists()


def test_queue_post_failure_leaves_draft_and_continues(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    _write_queue_files(tmp_path, _draft("c2"))
    tracker = _FakeTracker(fail_ids={"c1"})
    result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 1  # at least one failure
    assert "FAILED c1: synthetic create failure" in result.output
    assert "1 posted, 0 commented, 0 already tracked, 1 failed, 0 skipped." in result.output
    # c2 still posted; c1 remains queued for the next replay.
    assert [d.cluster_id for d in tracker.created] == ["c2"]
    assert (tmp_path / "c1.json").exists()
    assert not (tmp_path / "c2.json").exists()


def test_queue_post_cluster_filter(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    _write_queue_files(tmp_path, _draft("c2"))
    tracker = _FakeTracker()
    result = _invoke_post(tmp_path, tracker, "--yes", "--cluster", "c2")
    assert result.exit_code == 0, result.output
    assert [d.cluster_id for d in tracker.created] == ["c2"]
    assert (tmp_path / "c1.json").exists()


def test_queue_post_requires_tracker(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    runner = CliRunner()
    with patch("docket.cli.build_tracker", return_value=None):
        result = runner.invoke(main, ["queue", "post", "--queue-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No tracker configured" in result.output


def test_queue_post_prompts_without_yes(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    tracker = _FakeTracker()
    runner = CliRunner()
    with patch("docket.cli.build_tracker", return_value=tracker):
        result = runner.invoke(
            main,
            ["queue", "post", "--queue-dir", str(tmp_path), "--tracker", "github"],
            input="n\n",
        )
    assert result.exit_code == 0, result.output
    assert tracker.created == []
    assert "0 posted, 0 commented, 0 already tracked, 0 failed, 1 skipped." in result.output
    assert (tmp_path / "c1.json").exists()


def test_queue_clear(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    runner = CliRunner()
    result = runner.invoke(main, ["queue", "clear", "--queue-dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert "Removed 1 draft(s)." in result.output
    assert list(tmp_path.glob("*.json")) == []


def test_queue_clear_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["queue", "clear", "--queue-dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert "Queue is empty." in result.output

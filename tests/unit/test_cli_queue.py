"""Tests for the `agent-triage queue` command group."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from agent_triage.cli import main
from agent_triage.errors import TrackerError
from agent_triage.models.issue import Issue, IssueDraft, make_labels


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


class _FakeTracker:
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        self.created: list[IssueDraft] = []
        self.closed = False
        self._fail_ids = fail_ids or set()

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

    async def close(self) -> None:
        self.closed = True


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
    with patch("agent_triage.cli.build_tracker", return_value=tracker):
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
    # Files retired into posted/ with the issue URL recorded.
    assert not (tmp_path / "c1.json").exists()
    record = json.loads((tmp_path / "posted" / "c1.json").read_text())
    assert record["posted_issue_url"] == "https://tracker.example/i/c1"


def test_queue_post_failure_leaves_draft_and_continues(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    _write_queue_files(tmp_path, _draft("c2"))
    tracker = _FakeTracker(fail_ids={"c1"})
    result = _invoke_post(tmp_path, tracker, "--yes")
    assert result.exit_code == 1  # at least one failure
    assert "FAILED c1" in result.output
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
    with patch("agent_triage.cli.build_tracker", return_value=None):
        result = runner.invoke(main, ["queue", "post", "--queue-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No tracker configured" in result.output


def test_queue_post_prompts_without_yes(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    tracker = _FakeTracker()
    runner = CliRunner()
    with patch("agent_triage.cli.build_tracker", return_value=tracker):
        result = runner.invoke(
            main,
            ["queue", "post", "--queue-dir", str(tmp_path), "--tracker", "github"],
            input="n\n",
        )
    assert result.exit_code == 0, result.output
    assert tracker.created == []
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

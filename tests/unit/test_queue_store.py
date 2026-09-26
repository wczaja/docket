import json
import logging
from pathlib import Path

import pytest

from docket.models.issue import Issue, IssueDraft, make_labels
from docket.queue_store import (
    POSTED_SUBDIR,
    clear_queue,
    list_queued_drafts,
    mark_posted,
    retire_drafts,
)


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


def test_list_empty_or_missing_dir(tmp_path: Path) -> None:
    assert list_queued_drafts(tmp_path / "nope") == []
    (tmp_path / "empty").mkdir()
    assert list_queued_drafts(tmp_path / "empty") == []


def test_list_parses_drafts_in_name_order(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("b-cluster"))
    _write_queue_files(tmp_path, _draft("a-cluster"))
    queued = list_queued_drafts(tmp_path)
    assert [q.draft.cluster_id for q in queued] == ["a-cluster", "b-cluster"]
    assert all(q.md_path is not None for q in queued)


def test_list_skips_unparseable_files(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("good"))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "wrong-shape.json").write_text('{"title": "missing fields"}')
    queued = list_queued_drafts(tmp_path)
    assert [q.draft.cluster_id for q in queued] == ["good"]


def test_mark_posted_moves_files_and_records_url(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c9"))
    (queued,) = list_queued_drafts(tmp_path)
    target = mark_posted(queued, issue_url="https://tracker.example/i/9")
    assert not queued.json_path.exists()
    assert target == tmp_path / POSTED_SUBDIR / "c9.json"
    record = json.loads(target.read_text())
    assert record["posted_issue_url"] == "https://tracker.example/i/9"
    assert (tmp_path / POSTED_SUBDIR / "c9.md").exists()
    # The queue no longer lists the posted draft (posted/ is not scanned).
    assert list_queued_drafts(tmp_path) == []


def _issue(key: str) -> Issue:
    return Issue(id=key, key=key, url=f"https://tracker.example/{key}", title="t", body="b")


def test_retire_drafts_moves_only_the_named_drafts(tmp_path: Path) -> None:
    for cluster_id in ("c1", "c2", "c3"):
        _write_queue_files(tmp_path, _draft(cluster_id))
    retired = retire_drafts(
        # c2's issue is unknown; "gone" has no queued draft and is ignored.
        {"c1": _issue("AGT-1"), "c2": None, "gone": _issue("AGT-9")},
        queue_dir=tmp_path,
    )
    assert retired == 2
    assert [q.draft.cluster_id for q in list_queued_drafts(tmp_path)] == ["c3"]
    posted = tmp_path / POSTED_SUBDIR
    c1 = json.loads((posted / "c1.json").read_text())
    assert c1["posted_issue_url"] == "https://tracker.example/AGT-1"
    assert "posted_issue_url" not in json.loads((posted / "c2.json").read_text())
    assert (posted / "c1.md").exists()
    assert (posted / "c2.md").exists()


def test_retire_drafts_replaces_an_earlier_retirement(tmp_path: Path) -> None:
    # Re-running a window re-drafts the same cluster into the queue; retiring
    # it again must overwrite the posted/ copy, not trip over it.
    _write_queue_files(tmp_path, _draft("c1"))
    assert retire_drafts({"c1": _issue("AGT-1")}, queue_dir=tmp_path) == 1
    _write_queue_files(tmp_path, _draft("c1").model_copy(update={"title": "Re-drafted"}))
    assert retire_drafts({"c1": _issue("AGT-1")}, queue_dir=tmp_path) == 1
    assert list_queued_drafts(tmp_path) == []
    posted = tmp_path / POSTED_SUBDIR
    assert json.loads((posted / "c1.json").read_text())["title"] == "Re-drafted"
    assert "Re-drafted" in (posted / "c1.md").read_text()


def test_retire_drafts_failure_is_logged_and_leaves_draft_queued(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    (tmp_path / POSTED_SUBDIR).write_text("a file where the directory should be")
    with caplog.at_level(logging.WARNING, logger="docket.queue_store"):
        assert retire_drafts({"c1": _issue("AGT-1")}, queue_dir=tmp_path) == 0
    assert "could not retire queued draft c1" in caplog.text
    # Still queued, so the next replay dedups it again.
    assert [q.draft.cluster_id for q in list_queued_drafts(tmp_path)] == ["c1"]


def test_clear_queue_removes_only_queued(tmp_path: Path) -> None:
    _write_queue_files(tmp_path, _draft("c1"))
    _write_queue_files(tmp_path, _draft("c2"))
    (queued, _) = list_queued_drafts(tmp_path)
    mark_posted(queued)
    removed = clear_queue(tmp_path)
    assert removed == 1
    assert list_queued_drafts(tmp_path) == []
    # Posted records survive a clear.
    assert (tmp_path / POSTED_SUBDIR / "c1.json").exists()


def test_clear_missing_dir_is_zero(tmp_path: Path) -> None:
    assert clear_queue(tmp_path / "nope") == 0

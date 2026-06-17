import json
from pathlib import Path

from agent_triage.models.issue import IssueDraft, make_labels
from agent_triage.queue_store import (
    POSTED_SUBDIR,
    clear_queue,
    list_queued_drafts,
    mark_posted,
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

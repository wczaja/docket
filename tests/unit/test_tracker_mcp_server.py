"""Unit tests for the Jira MCP server dispatch layer.

Exercises `dispatch_tracker_tool` against an in-process fake tracker; the
stdio MCP transport is third-party and not unit-tested here.
"""

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

import agent_triage.mcp_servers._tracker_common as _tracker_common
from agent_triage.adapters.base import Tracker
from agent_triage.errors import TrackerError
from agent_triage.mcp_servers.adapter_github import SERVER_NAME as GITHUB_SERVER_NAME
from agent_triage.mcp_servers.adapter_jira import (
    SERVER_NAME,
    TRACKER_TOOLS,
    dispatch_tracker_tool,
)
from agent_triage.mcp_servers.adapter_linear import SERVER_NAME as LINEAR_SERVER_NAME
from agent_triage.models.issue import Issue, IssueDraft, IssuePatch


class _FakeTracker(Tracker):
    def __init__(self) -> None:
        self.list_filters: list[dict[str, Any] | None] = []
        self.search_calls: list[tuple[str, int]] = []
        self.created: list[IssueDraft] = []
        self.updated: list[tuple[str, IssuePatch]] = []
        self.comments: list[tuple[str, str]] = []
        self.closed = False

    async def list_open_issues(self, filter=None):  # type: ignore[no-untyped-def]
        self.list_filters.append(filter)
        return [
            Issue(id="1", key="AGT-1", title="existing", body="body", labels=["x"]),
        ]

    async def search_issues(self, query, k=10):  # type: ignore[no-untyped-def]
        self.search_calls.append((query, k))
        return []

    async def create_issue(self, draft):  # type: ignore[no-untyped-def]
        self.created.append(draft)
        return Issue(id="2", key="AGT-2", title=draft.title, body=draft.body, labels=draft.labels)

    async def update_issue(self, issue_id, patch):  # type: ignore[no-untyped-def]
        self.updated.append((issue_id, patch))
        return Issue(id=issue_id, title=patch.title or "old", body=patch.body or "", labels=[])

    async def comment_on_issue(self, issue_id, comment):  # type: ignore[no-untyped-def]
        self.comments.append((issue_id, comment))

    async def close(self) -> None:
        self.closed = True


def _draft_payload() -> dict[str, Any]:
    return {
        "cluster_id": "cl-1",
        "mode_id": "hallucination",
        "rubric_version": "agents@1.0.0",
        "run_id": "r-1",
        "severity": "high",
        "representative_trace_id": "t-1",
        "member_trace_ids": ["t-1"],
        "title": "title",
        "body": "body",
        "labels": ["agent-triage"],
    }


def test_tool_manifest_lists_all_five_methods() -> None:
    names = {t.name for t in TRACKER_TOOLS}
    assert names == {
        "list_open_issues",
        "search_issues",
        "create_issue",
        "update_issue",
        "comment_on_issue",
    }


def test_server_name() -> None:
    assert SERVER_NAME == "agent-triage-adapter-jira"


def test_linear_server_name() -> None:
    assert LINEAR_SERVER_NAME == "agent-triage-adapter-linear"


def test_github_server_name() -> None:
    assert GITHUB_SERVER_NAME == "agent-triage-adapter-github"


async def test_dispatch_list_open_issues_passes_filter() -> None:
    tracker = _FakeTracker()
    result = await dispatch_tracker_tool(
        tracker,
        "list_open_issues",
        {"filter": {"labels": ["agent-triage", "mode:hallucination"]}},
    )
    payload = json.loads(result)
    assert isinstance(payload, list)
    assert payload[0]["id"] == "1"
    assert tracker.list_filters[0] == {"labels": ["agent-triage", "mode:hallucination"]}


async def test_dispatch_list_open_issues_without_filter() -> None:
    tracker = _FakeTracker()
    await dispatch_tracker_tool(tracker, "list_open_issues", {})
    assert tracker.list_filters[0] is None


async def test_dispatch_search_issues() -> None:
    tracker = _FakeTracker()
    result = await dispatch_tracker_tool(
        tracker,
        "search_issues",
        {"query": "hello", "k": 3},
    )
    assert json.loads(result) == []
    assert tracker.search_calls == [("hello", 3)]


async def test_dispatch_create_issue() -> None:
    tracker = _FakeTracker()
    result = await dispatch_tracker_tool(
        tracker,
        "create_issue",
        {"draft": _draft_payload()},
    )
    payload = json.loads(result)
    assert payload["id"] == "2"
    assert tracker.created[0].title == "title"


async def test_dispatch_update_issue() -> None:
    tracker = _FakeTracker()
    result = await dispatch_tracker_tool(
        tracker,
        "update_issue",
        {"issue_id": "AGT-1", "patch": {"title": "new title"}},
    )
    payload = json.loads(result)
    assert payload["id"] == "AGT-1"
    assert tracker.updated[0][0] == "AGT-1"
    assert tracker.updated[0][1].title == "new title"


async def test_dispatch_comment_on_issue() -> None:
    tracker = _FakeTracker()
    result = await dispatch_tracker_tool(
        tracker,
        "comment_on_issue",
        {"issue_id": "AGT-1", "comment": "look here"},
    )
    assert json.loads(result) == {"ok": True}
    assert tracker.comments == [("AGT-1", "look here")]


async def test_dispatch_unknown_tool_raises() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="Unknown MCP tool"):
        await dispatch_tracker_tool(tracker, "frob", {})


async def test_dispatch_malformed_draft_raises_tracker_error() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="invalid tool argument.*draft"):
        await dispatch_tracker_tool(tracker, "create_issue", {"draft": {"title": 42}})
    assert tracker.created == []


async def test_dispatch_missing_draft_raises_tracker_error() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="invalid tool argument.*draft"):
        await dispatch_tracker_tool(tracker, "create_issue", {})


async def test_dispatch_malformed_patch_raises_tracker_error() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="invalid tool argument.*patch"):
        await dispatch_tracker_tool(
            tracker,
            "update_issue",
            {"issue_id": "AGT-1", "patch": {"title": ["not", "a", "string"]}},
        )
    assert tracker.updated == []


async def test_dispatch_missing_issue_id_raises_tracker_error() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="invalid tool argument.*issue_id"):
        await dispatch_tracker_tool(tracker, "comment_on_issue", {"comment": "look here"})
    with pytest.raises(TrackerError, match="invalid tool argument.*issue_id"):
        await dispatch_tracker_tool(tracker, "update_issue", {"patch": {"title": "x"}})
    assert tracker.comments == []
    assert tracker.updated == []


async def test_dispatch_missing_comment_raises_tracker_error() -> None:
    tracker = _FakeTracker()
    with pytest.raises(TrackerError, match="invalid tool argument.*comment"):
        await dispatch_tracker_tool(tracker, "comment_on_issue", {"issue_id": "AGT-1"})
    assert tracker.comments == []


class _FakeServer:
    def __init__(self, run_error: Exception | None = None) -> None:
        self._run_error = run_error

    def create_initialization_options(self) -> None:
        return None

    async def run(self, read_stream: Any, write_stream: Any, options: Any) -> None:
        if self._run_error is not None:
            raise self._run_error


@asynccontextmanager
async def _fake_stdio():  # type: ignore[no-untyped-def]
    yield (None, None)


async def test_serve_tracker_closes_backend_in_loop_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _FakeTracker()
    monkeypatch.setattr(_tracker_common, "build_tracker_server", lambda t, name: _FakeServer())
    monkeypatch.setattr(_tracker_common, "stdio_server", _fake_stdio)
    await _tracker_common.serve_tracker(tracker, SERVER_NAME)
    assert tracker.closed


async def test_serve_tracker_closes_backend_when_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _FakeTracker()
    monkeypatch.setattr(
        _tracker_common,
        "build_tracker_server",
        lambda t, name: _FakeServer(run_error=RuntimeError("boom")),
    )
    monkeypatch.setattr(_tracker_common, "stdio_server", _fake_stdio)
    with pytest.raises(RuntimeError, match="boom"):
        await _tracker_common.serve_tracker(tracker, SERVER_NAME)
    assert tracker.closed

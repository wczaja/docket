"""Unit tests for the Phoenix MCP server dispatch layer.

We exercise `dispatch_tool` directly against an in-process fake backend; the
stdio MCP transport itself is third-party and not unit-tested here.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

import docket.mcp_servers._common as _common
from docket.adapters.base import TraceBackend
from docket.errors import BackendError
from docket.mcp_servers.adapter_phoenix import (
    SERVER_NAME,
    TOOLS,
    dispatch_tool,
)
from docket.models.classification import Annotation
from docket.models.trace import OpenInferenceTrace, Span


class _FakeBackend(TraceBackend):
    def __init__(self) -> None:
        self.list_calls: list[tuple[datetime, datetime | None, dict | None]] = []
        self.annotations: list[Annotation] = []
        self.search_called = False
        self.marked: list[tuple[str, str, str]] = []
        self.processed_calls: list[tuple[str, datetime, datetime | None]] = []
        self.closed = False
        self._trace = OpenInferenceTrace(
            trace_id="t-fake",
            spans=[
                Span(
                    span_id="s",
                    trace_id="t-fake",
                    name="x",
                    start_time_unix_nano=1,
                    end_time_unix_nano=2,
                )
            ],
        )

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        self.list_calls.append((since, until, filter))
        return ["t-fake"]

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        return self._trace

    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        self.annotations.append(annotation)

    async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
        self.search_called = True
        raise NotImplementedError("Phoenix semantic search not available")

    async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
        self.marked.append((trace_id, run_id, rubric_version))

    async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
        self.processed_calls.append((run_id, since, until))
        return {"t-b", "t-a"}

    async def close(self) -> None:
        self.closed = True


def test_tool_manifest_lists_all_six_methods() -> None:
    names = {t.name for t in TOOLS}
    assert names == {
        "list_traces",
        "get_trace",
        "annotate_trace",
        "search_traces",
        "mark_trace_processed",
        "list_processed_trace_ids",
    }


def test_server_name() -> None:
    assert SERVER_NAME == "docket-adapter-phoenix"


async def test_dispatch_list_traces_passes_args() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(
        backend,
        "list_traces",
        {"since": "2026-05-22T00:00:00Z", "until": "2026-05-22T01:00:00Z"},
    )
    assert json.loads(result) == ["t-fake"]
    assert len(backend.list_calls) == 1
    since, until, _filter = backend.list_calls[0]
    assert since == datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    assert until == datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)


async def test_dispatch_list_traces_without_until() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(backend, "list_traces", {"since": "2026-05-22T00:00:00Z"})
    assert json.loads(result) == ["t-fake"]
    assert backend.list_calls[0][1] is None


async def test_dispatch_get_trace_returns_otlp() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(backend, "get_trace", {"trace_id": "t-fake"})
    payload = json.loads(result)
    assert "resourceSpans" in payload
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans[0]["traceId"] == "t-fake"


async def test_dispatch_annotate_trace_writes_annotation() -> None:
    backend = _FakeBackend()
    annotation = Annotation(
        trace_id="t-fake",
        run_id="r-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    result = await dispatch_tool(
        backend,
        "annotate_trace",
        {"trace_id": "t-fake", "annotation": annotation.model_dump(mode="json")},
    )
    assert json.loads(result) == {"ok": True}
    assert len(backend.annotations) == 1
    assert backend.annotations[0].mode_id == "hallucination"


async def test_dispatch_search_traces_propagates_not_implemented() -> None:
    backend = _FakeBackend()
    with pytest.raises(NotImplementedError):
        await dispatch_tool(backend, "search_traces", {"query": "q", "k": 5})


async def test_dispatch_unknown_tool_raises() -> None:
    backend = _FakeBackend()
    with pytest.raises(BackendError, match="Unknown MCP tool"):
        await dispatch_tool(backend, "frob", {})


async def test_dispatch_mark_trace_processed() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(
        backend,
        "mark_trace_processed",
        {"trace_id": "t-fake", "run_id": "r-1", "rubric_version": "agents-builtin@1.0.0"},
    )
    assert json.loads(result) == {"ok": True}
    assert backend.marked == [("t-fake", "r-1", "agents-builtin@1.0.0")]


async def test_dispatch_list_processed_trace_ids_returns_sorted_list() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(
        backend,
        "list_processed_trace_ids",
        {"run_id": "r-1", "since": "2026-05-22T00:00:00Z", "until": "2026-05-22T01:00:00Z"},
    )
    assert json.loads(result) == ["t-a", "t-b"]
    run_id, since, until = backend.processed_calls[0]
    assert run_id == "r-1"
    assert since == datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    assert until == datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)


async def test_dispatch_list_processed_trace_ids_without_until() -> None:
    backend = _FakeBackend()
    result = await dispatch_tool(
        backend,
        "list_processed_trace_ids",
        {"run_id": "r-1", "since": "2026-05-22T00:00:00Z"},
    )
    assert json.loads(result) == ["t-a", "t-b"]
    assert backend.processed_calls[0][2] is None


async def test_dispatch_malformed_annotation_raises_backend_error() -> None:
    backend = _FakeBackend()
    with pytest.raises(BackendError, match="invalid tool argument"):
        await dispatch_tool(
            backend,
            "annotate_trace",
            {"trace_id": "t-fake", "annotation": {"not": "an annotation"}},
        )
    assert backend.annotations == []


async def test_dispatch_missing_required_argument_raises_backend_error() -> None:
    backend = _FakeBackend()
    with pytest.raises(BackendError, match="invalid tool argument.*trace_id"):
        await dispatch_tool(backend, "get_trace", {})
    with pytest.raises(BackendError, match="invalid tool argument.*since"):
        await dispatch_tool(backend, "list_traces", {})
    with pytest.raises(BackendError, match="invalid tool argument.*run_id"):
        await dispatch_tool(backend, "mark_trace_processed", {"trace_id": "t-fake"})


async def test_dispatch_bad_datetime_raises_backend_error() -> None:
    backend = _FakeBackend()
    with pytest.raises(BackendError, match="invalid tool argument.*since"):
        await dispatch_tool(backend, "list_traces", {"since": "not-a-date"})


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


async def test_serve_closes_backend_in_loop_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(_common, "build_server", lambda b, name: _FakeServer())
    monkeypatch.setattr(_common, "stdio_server", _fake_stdio)
    await _common.serve(backend, SERVER_NAME)
    assert backend.closed


async def test_serve_closes_backend_when_run_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(
        _common, "build_server", lambda b, name: _FakeServer(run_error=RuntimeError("boom"))
    )
    monkeypatch.setattr(_common, "stdio_server", _fake_stdio)
    with pytest.raises(RuntimeError, match="boom"):
        await _common.serve(backend, SERVER_NAME)
    assert backend.closed

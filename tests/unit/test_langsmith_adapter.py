"""Unit tests for `LangsmithAdapter` against a mocked HTTP transport."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_triage.adapters.trace.langsmith import (
    DEFAULT_LANGSMITH_ENDPOINT,
    LangsmithAdapter,
)
from agent_triage.errors import BackendError
from agent_triage.models.classification import Annotation


def _make_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    api_key: str | None = "ls-test",  # noqa: S107
    project: str | None = None,
) -> LangsmithAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://langsmith.test")
    return LangsmithAdapter(
        endpoint="http://langsmith.test",
        api_key=api_key,
        project=project,
        client=client,
    )


async def test_list_traces_returns_run_ids() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.read().decode()))
        offset = captured[-1].get("offset", 0)
        if offset == 0:
            return httpx.Response(
                200,
                json={"runs": [{"id": f"r-{i}"} for i in range(100)]},
            )
        return httpx.Response(200, json={"runs": [{"id": "r-final"}]})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, tzinfo=UTC)
    ids = await adapter.list_traces(since)
    assert len(ids) == 101
    assert ids[0] == "r-0"
    assert ids[-1] == "r-final"
    assert captured[0]["is_root"] is True


async def test_list_traces_resolves_project_name_to_session_uuid() -> None:
    captured: dict[str, Any] = {}
    session_uuid = "11111111-1111-1111-1111-111111111111"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/sessions":
            assert request.url.params.get("name") == "my-project"
            return httpx.Response(200, json=[{"id": session_uuid, "name": "my-project"}])
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"runs": []})

    adapter = _make_adapter(handler, project="my-project")
    await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert captured["body"]["session"] == [session_uuid]


async def test_list_traces_accepts_project_uuid_directly() -> None:
    captured: dict[str, Any] = {}
    session_uuid = "22222222-2222-2222-2222-222222222222"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/api/v1/sessions", (
            "should not look up sessions when project is already a UUID"
        )
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"runs": []})

    adapter = _make_adapter(handler, project=session_uuid)
    await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert captured["body"]["session"] == [session_uuid]


async def test_list_traces_raises_when_project_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/sessions":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"runs": []})

    adapter = _make_adapter(handler, project="ghost-project")
    with pytest.raises(BackendError, match="ghost-project"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_get_trace_retries_on_429() -> None:
    """A transient 429 on the read path is retried; the second attempt
    succeeds. Retry-After: 0 keeps the shared helper from sleeping."""
    get_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        get_calls.append(1)
        if len(get_calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(
            200,
            json={
                "id": "r-1",
                "name": "completion",
                "run_type": "llm",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("r-1")
    assert len(get_calls) == 2
    assert trace.trace_id == "r-1"


async def test_annotate_trace_retries_on_429() -> None:
    """A transient 429 on the feedback write is retried; the deterministic
    feedback id makes the retry a safe upsert."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="r-1",
        run_id="run-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    await adapter.annotate_trace("r-1", annotation)
    assert len(calls) == 2


async def test_get_trace_surfaces_429_after_max_attempts() -> None:
    """If 429 persists across all retry attempts, the final response is surfaced."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "0"}, text="still throttled")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="429"):
        await adapter.get_trace("r-1")
    assert len(calls) == 8  # the shared helper's default max_attempts


async def test_list_traces_passes_until_when_provided() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"runs": []})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    await adapter.list_traces(since, until)
    assert "end_time" in captured["body"]
    assert "2026-05-22T01:00:00" in captured["body"]["end_time"]


async def test_list_traces_raises_on_http_error() -> None:
    # 500 is terminal (the shared helper only retries 502/503/504 + 429).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="langsmith unavailable")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="failed with 500"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_list_traces_handles_data_field_fallback() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "r-1"}]})

    adapter = _make_adapter(handler)
    ids = await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert ids == ["r-1"]


async def test_get_trace_translates_llm_run() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "run-1",
                "name": "completion",
                "run_type": "llm",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "extra": {"invocation_params": {"model": "gpt-4o-mini"}},
                "inputs": {
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
                "outputs": {
                    "generations": [
                        [
                            {
                                "message": {"role": "assistant", "content": "Four."},
                            }
                        ]
                    ],
                    "llm_output": {"token_usage": {"total_tokens": 15}},
                },
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("run-1")
    assert trace.trace_id == "run-1"
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.kind == "LLM"
    assert span.llm_model_name == "gpt-4o-mini"
    assert span.llm_token_count_total == 15
    inp = span.llm_input_messages
    assert inp[0].role == "user"
    assert inp[0].content == "What is 2+2?"
    out = span.llm_output_messages
    assert out[0].role == "assistant"
    assert out[0].content == "Four."


async def test_get_trace_translates_tool_run() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "run-tool",
                "name": "get_weather",
                "run_type": "tool",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "inputs": {"city": "Paris"},
                "outputs": "Sunny, 21C",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("run-tool")
    span = trace.spans[0]
    assert span.kind == "TOOL"
    assert span.tool_name == "get_weather"
    assert span.tool_parameters == {"city": "Paris"}
    assert span.tool_output == "Sunny, 21C"


async def test_get_trace_walks_child_runs() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "root",
                "name": "agent_run",
                "run_type": "chain",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:05Z",
                "child_runs": [
                    {
                        "id": "child-1",
                        "name": "completion",
                        "run_type": "llm",
                        "start_time": "2026-05-22T00:00:01Z",
                        "end_time": "2026-05-22T00:00:02Z",
                    },
                    {
                        "id": "child-2",
                        "name": "search",
                        "run_type": "tool",
                        "start_time": "2026-05-22T00:00:03Z",
                        "end_time": "2026-05-22T00:00:04Z",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("root")
    span_ids = [s.span_id for s in trace.spans]
    assert span_ids == ["root", "child-1", "child-2"]
    child_one = next(s for s in trace.spans if s.span_id == "child-1")
    assert child_one.parent_span_id == "root"
    assert child_one.kind == "LLM"


async def test_get_trace_maps_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "run-err",
                "name": "fail",
                "run_type": "chain",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "error": "destructive op rejected",
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("run-err")
    assert trace.spans[0].status.code == "ERROR"
    assert trace.spans[0].status.message == "destructive op rejected"


async def test_get_trace_raises_on_404() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="failed with 404"):
        await adapter.get_trace("missing")


async def test_get_trace_non_json_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-JSON"):
        await adapter.get_trace("run-1")


async def test_annotate_trace_posts_feedback() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "feedback-1"})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="run-1",
        run_id="r-7",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
        confidence=0.9,
        excerpt="The capital of France is Tokyo.",
    )
    await adapter.annotate_trace("run-1", annotation)
    assert captured["path"] == "/api/v1/feedback"
    body = captured["body"]
    assert body["run_id"] == "run-1"
    assert body["key"] == "agent-triage:hallucination"
    assert body["value"] == "positive"
    assert body["score"] == 0.9
    assert body["extra"]["idempotency_key"].startswith("run-1|r-7|")
    # Client-supplied feedback id derived from the idempotency key (upsert key).
    assert body["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, annotation.idempotency_key()))


async def test_annotate_trace_feedback_id_is_idempotent_and_run_scoped() -> None:
    ids_sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids_sent.append(json.loads(request.read().decode())["id"])
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="run-1",
        run_id="r-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    await adapter.annotate_trace("run-1", annotation)
    await adapter.annotate_trace("run-1", annotation)
    other_run = annotation.model_copy(update={"run_id": "r-2"})
    await adapter.annotate_trace("run-1", other_run)
    assert ids_sent[0] == ids_sent[1]  # re-run upserts the same feedback
    assert ids_sent[2] != ids_sent[0]  # a different run_id is new feedback


async def test_default_client_sets_api_key_header() -> None:
    """The lazily-built default client includes x-api-key. The MockTransport
    path doesn't exercise this — it injects an already-built client — so we
    spot-check it directly here."""
    adapter = LangsmithAdapter(api_key="ls-test")  # noqa: S106
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert client.headers.get("x-api-key") == "ls-test"
    await adapter.close()


async def test_annotate_trace_writes_negative_value() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="r-1",
        run_id="r",
        rubric_version="v",
        mode_id="m",
        positive=False,
        severity="low",
    )
    await adapter.annotate_trace("r-1", annotation)
    assert captured["body"]["value"] == "negative"


async def test_annotate_trace_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t",
        run_id="r",
        rubric_version="v",
        mode_id="m",
        positive=False,
        severity="low",
    )
    with pytest.raises(BackendError, match="feedback POST failed"):
        await adapter.annotate_trace("t", annotation)


async def test_search_traces_raises_not_implemented() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    adapter = _make_adapter(handler)
    with pytest.raises(NotImplementedError, match="semantic search"):
        await adapter.search_traces("query")


async def test_default_client_constructed_lazily() -> None:
    adapter = LangsmithAdapter(api_key="ls-test")  # noqa: S106
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert isinstance(client, httpx.AsyncClient)
    await adapter.close()
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001


async def test_close_when_never_used_is_safe() -> None:
    adapter = LangsmithAdapter()
    await adapter.close()


def test_default_endpoint_points_at_cloud() -> None:
    assert DEFAULT_LANGSMITH_ENDPOINT == "https://api.smith.langchain.com"


async def test_mark_trace_processed_posts_sentinel_feedback() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True})

    adapter = _make_adapter(handler)
    await adapter.mark_trace_processed("t-42", run_id="run-abc", rubric_version="agents/v1@1")
    assert captured["path"] == "/api/v1/feedback"
    assert captured["body"]["key"] == "agent-triage:processed"
    assert captured["body"]["run_id"] == "t-42"
    assert captured["body"]["extra"]["run_id"] == "run-abc"
    # Deterministic client-supplied id so re-marking the same trace upserts.
    expected_key = "t-42|run-abc|agents/v1@1|agent-triage:processed"
    assert captured["body"]["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, expected_key))


async def test_list_processed_trace_ids_filters_by_run_id() -> None:
    page_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page_calls.append(dict(request.url.params))
        return httpx.Response(
            200,
            json=[
                {"run_id": "t-1", "extra": {"run_id": "run-abc"}},
                {"run_id": "t-2", "extra": {"run_id": "run-OTHER"}},
                {"run_id": "t-3", "extra": {"run_id": "run-abc"}},
            ],
        )

    adapter = _make_adapter(handler)
    processed = await adapter.list_processed_trace_ids(
        run_id="run-abc",
        since=datetime(2026, 5, 22, tzinfo=UTC),
    )
    assert processed == {"t-1", "t-3"}
    assert page_calls[0]["key"] == "agent-triage:processed"


async def test_list_processed_trace_ids_returns_empty_for_unknown_run() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    adapter = _make_adapter(handler)
    out = await adapter.list_processed_trace_ids(
        run_id="never-seen",
        since=datetime(2026, 5, 22, tzinfo=UTC),
    )
    assert out == set()


async def test_mark_trace_processed_raises_on_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="langsmith outage")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="sentinel"):
        await adapter.mark_trace_processed("t-1", run_id="r", rubric_version="agents/v1@1")


async def test_get_trace_fetches_children_via_runs_query_when_absent() -> None:
    """GET /runs/{id} doesn't include child runs by default; the adapter must
    follow up with POST /runs/query filtered to the trace (M-8)."""
    query_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            query_bodies.append(json.loads(request.read().decode()))
            return httpx.Response(
                200,
                json={
                    "runs": [
                        {
                            "id": "root",
                            "name": "agent_run",
                            "run_type": "chain",
                            "start_time": "2026-05-22T00:00:00Z",
                            "end_time": "2026-05-22T00:00:05Z",
                        },
                        {
                            "id": "child-1",
                            "parent_run_id": "root",
                            "name": "completion",
                            "run_type": "llm",
                            "start_time": "2026-05-22T00:00:01Z",
                            "end_time": "2026-05-22T00:00:02Z",
                        },
                        {
                            "id": "child-2",
                            "parent_run_id": "child-1",
                            "name": "search",
                            "run_type": "tool",
                            "start_time": "2026-05-22T00:00:03Z",
                            "end_time": "2026-05-22T00:00:04Z",
                        },
                    ]
                },
            )
        # Run detail without any child_runs field.
        return httpx.Response(
            200,
            json={
                "id": "root",
                "name": "agent_run",
                "run_type": "chain",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:05Z",
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("root")
    assert [s.span_id for s in trace.spans] == ["root", "child-1", "child-2"]
    child_one = next(s for s in trace.spans if s.span_id == "child-1")
    assert child_one.parent_span_id == "root"
    child_two = next(s for s in trace.spans if s.span_id == "child-2")
    assert child_two.parent_span_id == "child-1"
    assert query_bodies[0]["trace"] == "root"


async def test_get_trace_single_span_when_query_returns_only_root() -> None:
    """A genuine single-span trace: the follow-up query returns only the
    root run and the adapter proceeds without error."""

    def handler(request: httpx.Request) -> httpx.Response:
        run = {
            "id": "solo",
            "name": "completion",
            "run_type": "llm",
            "start_time": "2026-05-22T00:00:00Z",
            "end_time": "2026-05-22T00:00:01Z",
        }
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": [run]})
        return httpx.Response(200, json=run)

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("solo")
    assert [s.span_id for s in trace.spans] == ["solo"]


async def test_naive_timestamp_is_treated_as_utc() -> None:
    """Offset-less timestamps normalize identically to their Z-suffixed twin
    regardless of host timezone (M-5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": "r-naive",
                "name": "completion",
                "run_type": "llm",
                "start_time": "2026-05-22T00:00:00",  # no offset -> UTC
                "end_time": "2026-05-22T00:00:01Z",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("r-naive")
    span = trace.spans[0]
    expected_ns = int(datetime(2026, 5, 22, tzinfo=UTC).timestamp() * 1_000_000_000)
    assert span.start_time_unix_nano == expected_ns
    assert span.end_time_unix_nano == expected_ns + 1_000_000_000


async def test_unparseable_start_time_yields_zero_duration() -> None:
    """An unparseable start timestamp falls back to the end timestamp
    (zero-duration span), never epoch / a ~56-year latency (M-5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": "r-bad-start",
                "name": "completion",
                "run_type": "llm",
                "start_time": "not-a-timestamp",
                "end_time": "2026-05-22T00:00:01Z",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("r-bad-start")
    span = trace.spans[0]
    assert span.start_time_unix_nano == span.end_time_unix_nano
    assert trace.to_trace_like().metrics["latency_ms"] == 0.0


async def test_run_with_no_parseable_timestamps_is_skipped() -> None:
    """Runs where both timestamps are unparseable are dropped rather than
    fabricating epoch values (M-5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": "root",
                "name": "agent_run",
                "run_type": "chain",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:05Z",
                "child_runs": [
                    {
                        "id": "child-bad",
                        "name": "completion",
                        "run_type": "llm",
                        "start_time": "garbage",
                        "end_time": "also-garbage",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("root")
    assert [s.span_id for s in trace.spans] == ["root"]


async def test_get_trace_absent_status_normalizes_to_unset() -> None:
    """No error and no explicit success -> UNSET, matching Phoenix (M-9)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": "r-1",
                "name": "completion",
                "run_type": "llm",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("r-1")
    assert trace.spans[0].status.code == "UNSET"


async def test_get_trace_explicit_success_status_maps_to_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/runs/query":
            return httpx.Response(200, json={"runs": []})
        return httpx.Response(
            200,
            json={
                "id": "r-1",
                "name": "completion",
                "run_type": "llm",
                "status": "success",
                "start_time": "2026-05-22T00:00:00Z",
                "end_time": "2026-05-22T00:00:01Z",
                "child_runs": [],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("r-1")
    assert trace.spans[0].status.code == "OK"


async def test_annotation_note_cannot_clobber_provenance_run_id() -> None:
    """A note keyed `run_id` must not override the provenance run_id (m-7)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="r-1",
        run_id="run-real",
        rubric_version="v",
        mode_id="m",
        positive=True,
        severity="low",
        notes={"run_id": "run-EVIL", "color": "blue"},
    )
    await adapter.annotate_trace("r-1", annotation)
    assert captured["body"]["extra"]["run_id"] == "run-real"
    assert captured["body"]["extra"]["color"] == "blue"


async def test_error_response_body_is_redacted() -> None:
    """Trace content echoed back in error bodies must be redacted before it
    lands in a BackendError message (M-17)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request: user jane.doe@example.com")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match=r"\[REDACTED_EMAIL\]") as excinfo:
        await adapter.get_trace("r-1")
    assert "jane.doe@example.com" not in str(excinfo.value)

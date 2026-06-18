"""Unit tests for `LangfuseAdapter` against a mocked HTTP transport.

Mirrors the Phoenix adapter's test structure: `httpx.MockTransport` intercepts
requests so we can assert on outgoing shape and synthesize typed responses
without spinning up a Langfuse server.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from docket.adapters.trace.langfuse import LangfuseAdapter
from docket.errors import BackendError
from docket.models.classification import Annotation


def _make_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    public_key: str | None = "pk-test",  # noqa: S107
    secret_key: str | None = "sk-test",  # noqa: S107
) -> LangfuseAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://langfuse.test")
    return LangfuseAdapter(
        host="http://langfuse.test",
        public_key=public_key,
        secret_key=secret_key,
        client=client,
    )


async def test_list_traces_returns_ids_from_paginated_response() -> None:
    pages_seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        pages_seen.append(page)
        if page == 1:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": f"t-{i}"} for i in range(100)],
                },
            )
        return httpx.Response(200, json={"data": [{"id": "t-final"}]})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    ids = await adapter.list_traces(since)
    assert len(ids) == 101  # 100 from page 1 + 1 from page 2
    assert ids[0] == "t-0"
    assert ids[-1] == "t-final"
    assert pages_seen == [1, 2]


async def test_list_traces_dedupes_ids_across_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(
                200,
                json={"data": [{"id": "t-1"}, {"id": "t-2"}]},
            )
        return httpx.Response(200, json={"data": []})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, tzinfo=UTC)
    ids = await adapter.list_traces(since)
    assert ids == ["t-1", "t-2"]


async def test_list_traces_passes_until_when_provided() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"data": []})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    await adapter.list_traces(since, until)
    assert "fromTimestamp" in captured["params"]
    assert "toTimestamp" in captured["params"]
    assert "2026-05-22T01:00:00" in captured["params"]["toTimestamp"]


async def test_list_traces_raises_on_http_error() -> None:
    # 500 is terminal (the shared helper only retries 502/503/504 + 429).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="langfuse unavailable")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="failed with 500"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_list_traces_raises_on_non_list_data() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": "not a list"})

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-list `data`"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_get_trace_translates_generation_to_llm_span() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t-1",
                "observations": [
                    {
                        "id": "obs-1",
                        "type": "GENERATION",
                        "name": "completion",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:01Z",
                        "model": "claude-haiku-4-5-20251001",
                        "input": [
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                        "output": {"role": "assistant", "content": "Four."},
                        "usage": {"total": 15},
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-1")
    assert trace.trace_id == "t-1"
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.kind == "LLM"
    assert span.llm_model_name == "claude-haiku-4-5-20251001"
    assert span.llm_token_count_total == 15
    msgs_in = span.llm_input_messages
    assert msgs_in[0].role == "user"
    assert msgs_in[0].content == "What is 2+2?"
    msgs_out = span.llm_output_messages
    assert msgs_out[0].role == "assistant"
    assert msgs_out[0].content == "Four."


async def test_get_trace_translates_tool_observation() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t-tool",
                "observations": [
                    {
                        "id": "obs-tool",
                        "type": "TOOL",
                        "name": "get_weather",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:01Z",
                        "input": {"city": "Paris"},
                        "output": "Sunny, 21C",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-tool")
    span = trace.spans[0]
    assert span.kind == "TOOL"
    assert span.tool_name == "get_weather"
    assert span.tool_parameters == {"city": "Paris"}
    assert span.tool_output == "Sunny, 21C"


async def test_get_trace_preserves_parent_observation_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t-parent",
                "observations": [
                    {
                        "id": "root",
                        "type": "SPAN",
                        "name": "agent_run",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:05Z",
                    },
                    {
                        "id": "child",
                        "parentObservationId": "root",
                        "type": "GENERATION",
                        "name": "completion",
                        "startTime": "2026-05-22T00:00:01Z",
                        "endTime": "2026-05-22T00:00:02Z",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-parent")
    child = next(s for s in trace.spans if s.span_id == "child")
    assert child.parent_span_id == "root"


async def test_get_trace_maps_error_level_to_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t-err",
                "observations": [
                    {
                        "id": "obs",
                        "type": "SPAN",
                        "name": "fail",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:01Z",
                        "level": "ERROR",
                        "statusMessage": "destructive op rejected",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-err")
    assert trace.spans[0].status.code == "ERROR"
    assert trace.spans[0].status.message == "destructive op rejected"


async def test_get_trace_raises_when_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "t-1", "observations": []})

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="has no observations"):
        await adapter.get_trace("t-1")


async def test_get_trace_raises_on_404() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="failed with 404"):
        await adapter.get_trace("t-missing")


async def test_annotate_trace_posts_score() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "score-1"})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="run-7",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
        confidence=0.9,
        excerpt="The capital of France is Tokyo.",
    )
    await adapter.annotate_trace("t-1", annotation)
    assert captured["path"] == "/api/public/scores"
    body = captured["body"]
    assert body["traceId"] == "t-1"
    assert body["name"] == "docket:hallucination"
    assert body["value"] == 1.0
    assert body["metadata"]["run_id"] == "run-7"
    assert body["metadata"]["idempotency_key"].startswith("t-1|run-7|")
    # Client-supplied score id derived from the idempotency key (upsert key).
    assert body["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, annotation.idempotency_key()))


async def test_annotate_trace_score_id_is_idempotent_and_run_scoped() -> None:
    ids_sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids_sent.append(json.loads(request.read().decode())["id"])
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="run-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    await adapter.annotate_trace("t-1", annotation)
    await adapter.annotate_trace("t-1", annotation)
    other_run = annotation.model_copy(update={"run_id": "run-2"})
    await adapter.annotate_trace("t-1", other_run)
    assert ids_sent[0] == ids_sent[1]  # re-run upserts the same score
    assert ids_sent[2] != ids_sent[0]  # a different run_id is a new score


async def test_annotate_trace_writes_zero_for_negative() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="r",
        rubric_version="v",
        mode_id="m",
        positive=False,
        severity="low",
    )
    await adapter.annotate_trace("t-1", annotation)
    assert captured["body"]["value"] == 0.0


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
    with pytest.raises(BackendError, match="score POST failed"):
        await adapter.annotate_trace("t", annotation)


async def test_search_traces_raises_not_implemented() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    adapter = _make_adapter(handler)
    with pytest.raises(NotImplementedError, match="semantic search"):
        await adapter.search_traces("query")


async def test_default_client_constructed_lazily() -> None:
    adapter = LangfuseAdapter(host="http://test", public_key="pk", secret_key="sk")  # noqa: S106
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert isinstance(client, httpx.AsyncClient)
    await adapter.close()
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001


async def test_close_when_never_used_is_safe() -> None:
    adapter = LangfuseAdapter(host="http://test")
    await adapter.close()


async def test_mark_trace_processed_writes_sentinel_score() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True})

    adapter = _make_adapter(handler)
    await adapter.mark_trace_processed("t-42", run_id="run-abc", rubric_version="agents/v1@1")
    assert captured["path"] == "/api/public/scores"
    assert captured["body"]["traceId"] == "t-42"
    assert captured["body"]["name"] == "docket:processed"
    assert captured["body"]["metadata"]["run_id"] == "run-abc"
    # Deterministic client-supplied id so re-marking the same trace upserts.
    expected_key = "t-42|run-abc|agents/v1@1|docket:processed"
    assert captured["body"]["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, expected_key))


async def test_list_processed_trace_ids_pages_and_filters_by_run_id() -> None:
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"traceId": "t-1", "metadata": {"run_id": "run-abc"}},
                        {"traceId": "t-2", "metadata": {"run_id": "run-OTHER"}},
                        {"traceId": "t-3", "metadata": {"run_id": "run-abc"}},
                    ]
                    * 34  # 102 items > _DEFAULT_PAGE_LIMIT(100), forces a 2nd page
                },
            )
        return httpx.Response(200, json={"data": []})

    adapter = _make_adapter(handler)
    processed = await adapter.list_processed_trace_ids(
        run_id="run-abc",
        since=datetime(2026, 5, 22, tzinfo=UTC),
    )
    assert processed == {"t-1", "t-3"}
    assert call_count == 2


async def test_mark_trace_processed_raises_on_error() -> None:
    # 500 is terminal (the shared helper only retries 502/503/504 + 429).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="langfuse hiccup")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="sentinel"):
        await adapter.mark_trace_processed("t-1", run_id="r", rubric_version="agents/v1@1")


async def test_get_trace_non_json_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-JSON"):
        await adapter.get_trace("t")


async def test_list_traces_retries_on_429() -> None:
    """A transient 429 on the read path is retried; Retry-After: 0 keeps the
    shared helper from sleeping (M-1)."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(200, json={"data": [{"id": "t-1"}]})

    adapter = _make_adapter(handler)
    ids = await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert ids == ["t-1"]
    assert len(calls) == 2


async def test_annotate_trace_retries_on_429() -> None:
    """A transient 429 on the score write is retried; the deterministic
    client-supplied id makes the retry a safe upsert (M-1)."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="run-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    await adapter.annotate_trace("t-1", annotation)
    assert len(calls) == 2


def _single_observation_response(start: str, end: str) -> dict[str, Any]:
    return {
        "id": "t",
        "observations": [
            {
                "id": "obs",
                "type": "SPAN",
                "name": "x",
                "startTime": start,
                "endTime": end,
            }
        ],
    }


async def test_naive_timestamp_is_treated_as_utc() -> None:
    """Offset-less timestamps normalize identically to their Z-suffixed twin
    regardless of host timezone (M-5)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_single_observation_response("2026-05-22T00:00:00", "2026-05-22T00:00:01Z"),
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    span = trace.spans[0]
    expected_ns = int(datetime(2026, 5, 22, tzinfo=UTC).timestamp() * 1_000_000_000)
    assert span.start_time_unix_nano == expected_ns
    assert span.end_time_unix_nano == expected_ns + 1_000_000_000


async def test_unparseable_start_time_yields_zero_duration() -> None:
    """An unparseable start timestamp falls back to the end timestamp
    (zero-duration span), never epoch / a ~56-year latency (M-5)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_single_observation_response("not-a-timestamp", "2026-05-22T00:00:01Z"),
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    span = trace.spans[0]
    assert span.start_time_unix_nano == span.end_time_unix_nano
    assert trace.to_trace_like().metrics["latency_ms"] == 0.0


async def test_observation_with_no_parseable_timestamps_is_skipped() -> None:
    """Observations where both timestamps are unparseable are dropped rather
    than fabricating epoch values (M-5)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "t",
                "observations": [
                    {
                        "id": "obs-bad",
                        "type": "SPAN",
                        "name": "bad",
                        "startTime": "garbage",
                        "endTime": "also-garbage",
                    },
                    {
                        "id": "obs-good",
                        "type": "SPAN",
                        "name": "good",
                        "startTime": "2026-05-22T00:00:00Z",
                        "endTime": "2026-05-22T00:00:01Z",
                    },
                ],
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    assert [s.span_id for s in trace.spans] == ["obs-good"]


async def test_get_trace_absent_level_normalizes_to_unset_status() -> None:
    """No error signal and no explicit success -> UNSET, matching Phoenix (M-9)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_single_observation_response("2026-05-22T00:00:00Z", "2026-05-22T00:00:01Z"),
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    assert trace.spans[0].status.code == "UNSET"


async def test_annotation_note_cannot_clobber_provenance_run_id() -> None:
    """A note keyed `run_id` must not override the provenance run_id (m-7)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="run-real",
        rubric_version="v",
        mode_id="m",
        positive=True,
        severity="low",
        notes={"run_id": "run-EVIL", "color": "blue"},
    )
    await adapter.annotate_trace("t-1", annotation)
    assert captured["body"]["metadata"]["run_id"] == "run-real"
    assert captured["body"]["metadata"]["color"] == "blue"


async def test_list_processed_trace_ids_raises_on_non_list_data() -> None:
    """A malformed scores listing raises instead of silently truncating (m-8)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": "not a list"})

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-list `data`"):
        await adapter.list_processed_trace_ids(
            run_id="r",
            since=datetime(2026, 5, 22, tzinfo=UTC),
        )


async def test_list_processed_trace_ids_raises_on_non_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-JSON"):
        await adapter.list_processed_trace_ids(
            run_id="r",
            since=datetime(2026, 5, 22, tzinfo=UTC),
        )


async def test_error_response_body_is_redacted() -> None:
    """Trace content echoed back in error bodies must be redacted before it
    lands in a BackendError message (M-17)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom: user jane.doe@example.com")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match=r"\[REDACTED_EMAIL\]") as excinfo:
        await adapter.get_trace("t")
    assert "jane.doe@example.com" not in str(excinfo.value)

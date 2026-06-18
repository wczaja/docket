"""Unit tests for `PhoenixAdapter` against a mocked HTTP transport.

`httpx.MockTransport` is purpose-built for this — it intercepts requests at
the transport layer so we can assert on the outgoing payload and synthesize
typed responses without subclassing.
"""

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from docket.adapters.trace.phoenix import PhoenixAdapter
from docket.errors import BackendError
from docket.models.classification import Annotation


def _make_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
) -> PhoenixAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://phoenix.test")
    return PhoenixAdapter(base_url="http://phoenix.test", client=client)


def _root_span_response(span_id: str = "s-root") -> dict[str, Any]:
    """GraphQL response for the root-span resolution query: one parentless
    span plus a child, so the adapter has to pick the parentless one."""
    return {
        "data": {
            "spans": {
                "edges": [
                    {"node": {"context": {"spanId": "s-child"}, "parentId": span_id}},
                    {"node": {"context": {"spanId": span_id}, "parentId": None}},
                ],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }


async def test_list_traces_returns_deduped_trace_ids() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {"node": {"context": {"traceId": "t-1"}}},
                            {"node": {"context": {"traceId": "t-2"}}},
                            {"node": {"context": {"traceId": "t-1"}}},
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    ids = await adapter.list_traces(since)
    assert ids == ["t-1", "t-2"]
    assert captured["path"] == "/graphql"
    assert "ListSpans" in captured["body"]


async def test_list_traces_passes_until_when_provided() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"data": {"spans": {"edges": []}}})

    adapter = _make_adapter(handler)
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    await adapter.list_traces(since, until)
    assert "2026-05-22T01:00:00" in captured["body"]


async def test_list_traces_empty_window() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"spans": {"edges": []}}})

    adapter = _make_adapter(handler)
    ids = await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert ids == []


async def test_get_trace_decodes_phoenix_span_shape() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {
                                "node": {
                                    "context": {"traceId": "t-1", "spanId": "s-1"},
                                    "parentId": None,
                                    "name": "completion",
                                    "startTime": "2026-05-22T00:00:00Z",
                                    "endTime": "2026-05-22T00:00:01Z",
                                    "attributes": {
                                        "openinference.span.kind": "LLM",
                                        "llm.model_name": "claude-haiku-4-5-20251001",
                                    },
                                    "statusCode": "OK",
                                    "statusMessage": None,
                                    "events": [],
                                }
                            }
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-1")
    assert trace.trace_id == "t-1"
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.kind == "LLM"
    assert span.llm_model_name == "claude-haiku-4-5-20251001"
    assert span.start_time_unix_nano > 0
    assert span.end_time_unix_nano > span.start_time_unix_nano


async def test_get_trace_decodes_attributes_when_serialized_as_string() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {
                                "node": {
                                    "context": {"traceId": "t", "spanId": "s"},
                                    "name": "x",
                                    "startTime": "2026-05-22T00:00:00Z",
                                    "endTime": "2026-05-22T00:00:01Z",
                                    "attributes": '{"foo": "bar"}',
                                    "statusCode": "OK",
                                }
                            }
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    assert trace.spans[0].attributes == {"foo": "bar"}


async def test_get_trace_raises_when_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"spans": {"edges": []}}})

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="returned no spans"):
        await adapter.get_trace("missing")


async def test_get_trace_decodes_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {
                                "node": {
                                    "context": {"traceId": "t", "spanId": "s"},
                                    "name": "x",
                                    "startTime": "2026-05-22T00:00:00Z",
                                    "endTime": "2026-05-22T00:00:01Z",
                                    "attributes": {},
                                    "statusCode": "STATUS_CODE_ERROR",
                                    "statusMessage": "boom",
                                }
                            }
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    assert trace.spans[0].status.code == "ERROR"
    assert trace.spans[0].status.message == "boom"


async def test_annotate_trace_sends_payload_targeting_root_span() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response("s-root"))
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"ok": True})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="run-1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
        confidence=0.9,
        excerpt="suspicious",
    )
    await adapter.annotate_trace("t-1", annotation)
    assert captured["path"] == "/v1/span_annotations"
    record = captured["body"]["data"][0]
    # The annotation targets the trace's root span, not the trace ID.
    assert record["span_id"] == "s-root"
    assert record["trace_id"] == "t-1"
    assert record["name"] == "docket:hallucination"
    assert record["identifier"] == annotation.idempotency_key()
    assert record["metadata"]["run_id"] == "run-1"
    assert record["metadata"]["idempotency_key"] == annotation.idempotency_key()


async def test_annotate_trace_identifier_is_idempotent_and_run_scoped() -> None:
    identifiers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response())
        body = json.loads(request.read().decode())
        identifiers.append(body["data"][0]["identifier"])
        return httpx.Response(201, json={"ok": True})

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
    assert identifiers[0] == identifiers[1]  # re-run upserts the same record
    assert identifiers[2] != identifiers[0]  # a different run_id is a new record


async def test_annotate_trace_raises_when_no_root_span() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json={"data": {"spans": {"edges": []}}})
        return httpx.Response(201, json={"ok": True})

    adapter = _make_adapter(handler)
    annotation = Annotation(
        trace_id="t-1",
        run_id="r",
        rubric_version="v",
        mode_id="m",
        positive=False,
        severity="low",
    )
    with pytest.raises(BackendError, match="root span"):
        await adapter.annotate_trace("t-1", annotation)


async def test_annotate_trace_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response())
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
    with pytest.raises(BackendError, match="annotation POST failed"):
        await adapter.annotate_trace("t", annotation)


async def test_search_traces_raises_not_implemented() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    adapter = _make_adapter(handler)
    with pytest.raises(NotImplementedError, match="semantic search"):
        await adapter.search_traces("query")


async def test_graphql_http_error_raises_backend_error() -> None:
    # 500 is terminal (the shared helper only retries 502/503/504 + 429).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="phoenix unavailable")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="failed with 500"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_graphql_errors_field_raises_backend_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"errors": [{"message": "field 'spans' not found"}]},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="Phoenix GraphQL errors"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_graphql_non_json_raises_backend_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="non-JSON"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))


async def test_default_client_constructed_lazily() -> None:
    adapter = PhoenixAdapter(base_url="http://test", api_key="key123")
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert isinstance(client, httpx.AsyncClient)
    await adapter.close()
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001


async def test_close_when_never_used_is_safe() -> None:
    adapter = PhoenixAdapter(base_url="http://test")
    await adapter.close()


async def test_mark_trace_processed_writes_sentinel_annotation() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response("s-root"))
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"ok": True})

    adapter = _make_adapter(handler)
    await adapter.mark_trace_processed("t-42", run_id="run-abc", rubric_version="agents/v1@1")
    assert captured["path"] == "/v1/span_annotations"
    record = captured["body"]["data"][0]
    assert record["name"] == "docket:processed"
    # The sentinel targets the trace's root span, not the trace ID.
    assert record["span_id"] == "s-root"
    assert record["trace_id"] == "t-42"
    assert record["identifier"] == "t-42|run-abc|agents/v1@1|docket:processed"
    assert record["metadata"]["run_id"] == "run-abc"
    assert record["metadata"]["rubric_version"] == "agents/v1@1"


async def test_list_processed_trace_ids_filters_by_run_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spanAnnotations": [
                        {"traceId": "t-1", "metadata": {"run_id": "run-abc"}},
                        {"traceId": "t-2", "metadata": {"run_id": "run-OTHER"}},
                        {"traceId": "t-3", "metadata": {"run_id": "run-abc"}},
                        {"spanId": "t-4", "metadata": {"run_id": "run-abc"}},
                    ]
                }
            },
        )

    adapter = _make_adapter(handler)
    processed = await adapter.list_processed_trace_ids(
        run_id="run-abc",
        since=datetime(2026, 5, 22, tzinfo=UTC),
    )
    assert processed == {"t-1", "t-3", "t-4"}


async def test_list_processed_trace_ids_empty_when_no_match() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"spanAnnotations": []}})

    adapter = _make_adapter(handler)
    out = await adapter.list_processed_trace_ids(
        run_id="never-seen",
        since=datetime(2026, 5, 22, tzinfo=UTC),
    )
    assert out == set()


async def test_mark_trace_processed_raises_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response())
        return httpx.Response(500, text="phoenix down")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="sentinel"):
        await adapter.mark_trace_processed("t-1", run_id="r", rubric_version="agents/v1@1")


def _span_node(trace_id: str, span_id: str) -> dict[str, Any]:
    return {
        "node": {
            "context": {"traceId": trace_id, "spanId": span_id},
            "parentId": None,
            "name": "step",
            "startTime": "2026-05-22T00:00:00Z",
            "endTime": "2026-05-22T00:00:01Z",
            "attributes": {},
            "statusCode": "OK",
            "statusMessage": None,
            "events": [],
        }
    }


async def test_list_traces_follows_cursor_across_pages() -> None:
    cursors_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        after = body["variables"]["after"]
        cursors_seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "spans": {
                            "edges": [
                                {"node": {"context": {"traceId": "t-1"}}},
                                {"node": {"context": {"traceId": "t-2"}}},
                            ],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {"node": {"context": {"traceId": "t-2"}}},
                            {"node": {"context": {"traceId": "t-3"}}},
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    ids = await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert ids == ["t-1", "t-2", "t-3"]
    assert cursors_seen == [None, "cursor-1"]


async def test_list_traces_raises_backend_error_on_page_cap() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [{"node": {"context": {"traceId": f"t-{calls}"}}}],
                        "pageInfo": {"hasNextPage": True, "endCursor": f"cursor-{calls}"},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="narrow the time window"):
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert calls == 50  # _MAX_PAGES


async def test_get_trace_follows_cursor_across_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        after = body["variables"]["after"]
        if after is None:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "spans": {
                            "edges": [_span_node("t-long", "s-1")],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [_span_node("t-long", "s-2")],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t-long")
    assert [s.span_id for s in trace.spans] == ["s-1", "s-2"]


async def test_list_traces_retries_on_429() -> None:
    """A transient 429 on the GraphQL read is retried; Retry-After: 0 keeps
    the shared helper from sleeping (M-1)."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(
            200,
            json={"data": {"spans": {"edges": [{"node": {"context": {"traceId": "t-1"}}}]}}},
        )

    adapter = _make_adapter(handler)
    ids = await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert ids == ["t-1"]
    assert len(calls) == 2


async def test_annotate_trace_retries_on_429() -> None:
    """A transient 429 on the annotation write is retried; the deterministic
    identifier makes the retry a safe upsert (M-1)."""
    write_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response())
        write_calls.append(1)
        if len(write_calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(201, json={"ok": True})

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
    assert len(write_calls) == 2


def _timestamp_span_response(start: str, end: str) -> dict[str, Any]:
    return {
        "data": {
            "spans": {
                "edges": [
                    {
                        "node": {
                            "context": {"traceId": "t", "spanId": "s"},
                            "name": "x",
                            "startTime": start,
                            "endTime": end,
                            "attributes": {},
                        }
                    }
                ]
            }
        }
    }


async def test_naive_timestamp_is_treated_as_utc() -> None:
    """Offset-less timestamps normalize identically to their Z-suffixed twin
    regardless of host timezone (M-5)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_timestamp_span_response("2026-05-22T00:00:00", "2026-05-22T00:00:01Z"),
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
            json=_timestamp_span_response("not-a-timestamp", "2026-05-22T00:00:01Z"),
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    span = trace.spans[0]
    assert span.start_time_unix_nano == span.end_time_unix_nano
    assert trace.to_trace_like().metrics["latency_ms"] == 0.0


async def test_span_with_no_parseable_timestamps_is_skipped() -> None:
    """Spans where both timestamps are unparseable are dropped rather than
    fabricating epoch values; a trace with no usable spans raises (M-5)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "spans": {
                        "edges": [
                            {
                                "node": {
                                    "context": {"traceId": "t", "spanId": "s-bad"},
                                    "name": "bad",
                                    "startTime": "garbage",
                                    "endTime": "also-garbage",
                                    "attributes": {},
                                }
                            },
                            {
                                "node": {
                                    "context": {"traceId": "t", "spanId": "s-good"},
                                    "name": "good",
                                    "startTime": "2026-05-22T00:00:00Z",
                                    "endTime": "2026-05-22T00:00:01Z",
                                    "attributes": {},
                                }
                            },
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    trace = await adapter.get_trace("t")
    assert [s.span_id for s in trace.spans] == ["s-good"]


async def test_trace_with_only_unparseable_spans_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_timestamp_span_response("garbage", "also-garbage"))

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match="returned no spans"):
        await adapter.get_trace("t")


async def test_annotation_note_cannot_clobber_provenance_run_id() -> None:
    """A note keyed `run_id` must not override the provenance run_id (m-7)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_root_span_response())
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"ok": True})

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
    metadata = captured["body"]["data"][0]["metadata"]
    assert metadata["run_id"] == "run-real"
    assert metadata["color"] == "blue"


async def test_list_processed_trace_ids_uses_z_suffixed_iso() -> None:
    """The sentinel listing must use the same `_to_iso` Z-suffixed encoding
    as every other Phoenix query (m-9)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"data": {"spanAnnotations": []}})

    adapter = _make_adapter(handler)
    await adapter.list_processed_trace_ids(
        run_id="r",
        since=datetime(2026, 5, 22, tzinfo=UTC),
        until=datetime(2026, 5, 23, tzinfo=UTC),
    )
    variables = captured["body"]["variables"]
    assert variables["start"] == "2026-05-22T00:00:00Z"
    assert variables["end"] == "2026-05-23T00:00:00Z"


async def test_error_response_body_is_redacted() -> None:
    """Trace content echoed back in error bodies must be redacted before it
    lands in a BackendError message (M-17)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom: user jane.doe@example.com")

    adapter = _make_adapter(handler)
    with pytest.raises(BackendError, match=r"\[REDACTED_EMAIL\]") as excinfo:
        await adapter.list_traces(datetime(2026, 5, 22, tzinfo=UTC))
    assert "jane.doe@example.com" not in str(excinfo.value)

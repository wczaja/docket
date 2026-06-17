"""Phoenix trace-backend adapter (design §5.1).

Phoenix exposes traces as collections of spans through a GraphQL endpoint at
`/graphql` and accepts span annotations through `/v1/span_annotations` (the
Phoenix 5.x annotation API). This adapter wraps both surfaces in the
`TraceBackend` contract.

The GraphQL queries here are intentionally minimal — Phoenix's schema is
broad and varies by version. Production deployments should pin a Phoenix
version and the adapter should be re-validated against that version's schema
on upgrade. The unit tests in this repo run against a mocked HTTP transport;
end-to-end validation against a real Phoenix happens in the integration test
(C3) under `tests/integration/test_phoenix_e2e.py`.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

import httpx

from agent_triage.adapters._retry import request_with_retry
from agent_triage.adapters.base import TraceBackend
from agent_triage.errors import BackendError
from agent_triage.models.classification import Annotation
from agent_triage.models.trace import Event, OpenInferenceTrace, Span, Status
from agent_triage.observability import redact

log = logging.getLogger(__name__)

_LIST_SPANS_QUERY = """\
query ListSpans($start: DateTime!, $end: DateTime, $first: Int!, $after: String) {
  spans(timeRange: {start: $start, end: $end}, first: $first, after: $after) {
    edges {
      node {
        context { traceId }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

_GET_TRACE_SPANS_QUERY = """\
query GetTraceSpans($traceId: ID!, $first: Int!, $after: String) {
  spans(traceIds: [$traceId], first: $first, after: $after) {
    edges {
      node {
        context { traceId spanId }
        parentId
        name
        startTime
        endTime
        attributes
        statusCode
        statusMessage
        events {
          name
          timeUnixNano
          attributes
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

_GET_ROOT_SPAN_QUERY = """\
query GetRootSpan($traceId: ID!, $first: Int!) {
  spans(traceIds: [$traceId], first: $first) {
    edges {
      node {
        context { spanId }
        parentId
      }
    }
  }
}
"""

_DEFAULT_PAGE_SIZE = 500
_MAX_PAGES = 50  # safety cap to avoid runaway pagination; mirrors the other adapters
_PROCESSED_SENTINEL_NAME = "agent-triage:processed"


class PhoenixAdapter(TraceBackend):
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_traces(
        self,
        since: datetime,
        until: datetime | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[str]:
        del filter  # Phoenix filter passthrough lands when a real use-case shows up
        edges = await self._collect_span_edges(
            _LIST_SPANS_QUERY,
            {
                "start": _to_iso(since),
                "end": _to_iso(until) if until is not None else None,
                "first": _DEFAULT_PAGE_SIZE,
            },
            context="list_traces",
        )
        seen: list[str] = []
        seen_set: set[str] = set()
        for edge in edges:
            trace_id = _safe_get(edge, "node", "context", "traceId")
            if isinstance(trace_id, str) and trace_id not in seen_set:
                seen.append(trace_id)
                seen_set.add(trace_id)
        return seen

    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        edges = await self._collect_span_edges(
            _GET_TRACE_SPANS_QUERY,
            {"traceId": trace_id, "first": _DEFAULT_PAGE_SIZE},
            context=f"get_trace({trace_id!r})",
        )
        spans = [
            span
            for edge in edges
            if "node" in edge and (span := _decode_phoenix_span(edge["node"])) is not None
        ]
        if not spans:
            raise BackendError(f"Phoenix: trace {trace_id!r} returned no spans")
        return OpenInferenceTrace(trace_id=trace_id, spans=spans)

    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        # A trace ID is not a span ID: Phoenix span annotations must target a
        # real span, so we anchor triage annotations on the trace's root span.
        root_span_id = await self._resolve_root_span_id(trace_id)
        body = {
            "data": [
                {
                    "span_id": root_span_id,
                    "trace_id": trace_id,
                    "name": f"agent-triage:{annotation.mode_id}",
                    # Phoenix upserts span annotations on (span, name, identifier),
                    # which is what makes re-runs idempotent (design §5.1).
                    "identifier": annotation.idempotency_key(),
                    "annotator_kind": "LLM",
                    "result": {
                        "label": "positive" if annotation.positive else "negative",
                        "score": annotation.confidence,
                        "explanation": annotation.excerpt,
                    },
                    "metadata": {
                        # Notes first; provenance keys last so a stray note
                        # (e.g. keyed "run_id") can't clobber them.
                        **annotation.notes,
                        "run_id": annotation.run_id,
                        "rubric_version": annotation.rubric_version,
                        "mode_id": annotation.mode_id,
                        "severity": annotation.severity,
                        "idempotency_key": annotation.idempotency_key(),
                    },
                }
            ]
        }
        client = self._get_client()
        # The deterministic `identifier` makes this POST a true upsert, so a
        # retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await request_with_retry(
            client,
            "POST",
            "/v1/span_annotations",
            error_cls=BackendError,
            idempotent=True,
            json_body=body,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Phoenix: annotation POST failed with "
                f"{response.status_code}: {redact(response.text)}"
            )

    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        del query, k
        raise NotImplementedError(
            "Phoenix semantic search is not exposed via GraphQL in v1.0; "
            "use `list_traces` with a time window and post-filter in-process."
        )

    async def mark_trace_processed(
        self,
        trace_id: str,
        *,
        run_id: str,
        rubric_version: str,
    ) -> None:
        root_span_id = await self._resolve_root_span_id(trace_id)
        # Same upsert mechanics as annotate_trace: re-marking the same
        # (trace, run, rubric) overwrites rather than duplicates.
        sentinel_key = f"{trace_id}|{run_id}|{rubric_version}|{_PROCESSED_SENTINEL_NAME}"
        body = {
            "data": [
                {
                    "span_id": root_span_id,
                    "trace_id": trace_id,
                    "name": _PROCESSED_SENTINEL_NAME,
                    "identifier": sentinel_key,
                    "annotator_kind": "LLM",
                    "result": {
                        "label": "processed",
                        "score": 1.0,
                        "explanation": "agent-triage Phase 11 resumability sentinel",
                    },
                    "metadata": {
                        "run_id": run_id,
                        "rubric_version": rubric_version,
                    },
                }
            ]
        }
        client = self._get_client()
        # The deterministic `identifier` makes this POST a true upsert, so a
        # retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await request_with_retry(
            client,
            "POST",
            "/v1/span_annotations",
            error_cls=BackendError,
            idempotent=True,
            json_body=body,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Phoenix: sentinel annotation POST failed with "
                f"{response.status_code}: {redact(response.text)}"
            )

    async def list_processed_trace_ids(
        self,
        *,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> set[str]:
        # Phoenix's GraphQL schema exposes span annotations as a sub-field of
        # spans; we query annotations matching our sentinel name + run_id and
        # collect the parent trace IDs. The window scope keeps the result set
        # bounded even for high-volume projects.
        query = """
        query ListProcessed($name: String!, $start: DateTime!, $end: DateTime) {
          spanAnnotations(filter: {name: $name, timeRange: {start: $start, end: $end}}) {
            spanId
            traceId
            metadata
          }
        }
        """
        payload = {
            "query": query,
            "variables": {
                "name": _PROCESSED_SENTINEL_NAME,
                "start": _to_iso(since),
                "end": _to_iso(until) if until is not None else None,
            },
        }
        data = await self._graphql(payload)
        annotations = data.get("spanAnnotations") or []
        processed: set[str] = set()
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            metadata = ann.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("run_id") == run_id:
                tid = ann.get("traceId") or ann.get("spanId")
                if isinstance(tid, str):
                    processed.add(tid)
        return processed

    async def _collect_span_edges(
        self,
        query: str,
        variables: dict[str, Any],
        *,
        context: str,
    ) -> list[dict[str, Any]]:
        """Page through a `spans` connection with an `after` cursor.

        Raises `BackendError` when `_MAX_PAGES` is exhausted while more pages
        remain — the design forbids silently truncating a window or a trace.
        """
        collected: list[dict[str, Any]] = []
        after: str | None = None
        for _ in range(_MAX_PAGES):
            payload = {"query": query, "variables": {**variables, "after": after}}
            data = await self._graphql(payload)
            edges = _safe_get(data, "spans", "edges", default=[])
            if not isinstance(edges, list):
                raise BackendError(
                    f"Phoenix: unexpected `spans.edges` shape: {type(edges).__name__}"
                )
            collected.extend(edge for edge in edges if isinstance(edge, dict))
            page_info = _safe_get(data, "spans", "pageInfo", default={})
            if not isinstance(page_info, dict):
                page_info = {}
            end_cursor = page_info.get("endCursor")
            if not page_info.get("hasNextPage") or not isinstance(end_cursor, str):
                return collected
            after = end_cursor
        raise BackendError(
            f"Phoenix: {context} exceeded the {_MAX_PAGES}-page safety cap "
            f"({_MAX_PAGES * _DEFAULT_PAGE_SIZE} spans) with more pages remaining; "
            "narrow the time window and re-run rather than truncating silently."
        )

    async def _resolve_root_span_id(self, trace_id: str) -> str:
        """Resolve the trace's root span (the span with no parent).

        Annotations must target a real span; the first page of spans is
        enough to find the root. No cross-call caching — the runtime is
        stateless by design.
        """
        payload = {
            "query": _GET_ROOT_SPAN_QUERY,
            "variables": {"traceId": trace_id, "first": _DEFAULT_PAGE_SIZE},
        }
        data = await self._graphql(payload)
        edges = _safe_get(data, "spans", "edges", default=[])
        if not isinstance(edges, list):
            raise BackendError(f"Phoenix: unexpected `spans.edges` shape: {type(edges).__name__}")
        for edge in edges:
            node = _safe_get(edge, "node")
            if not isinstance(node, dict) or node.get("parentId"):
                continue
            span_id = _safe_get(node, "context", "spanId")
            if isinstance(span_id, str) and span_id:
                return span_id
        raise BackendError(f"Phoenix: could not resolve a root span for trace {trace_id!r}")

    async def _graphql(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        # GraphQL queries are reads — idempotent, safe to retry on 5xx/timeouts.
        response = await request_with_retry(
            client,
            "POST",
            "/graphql",
            error_cls=BackendError,
            idempotent=True,
            json_body=payload,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Phoenix GraphQL request failed with "
                f"{response.status_code}: {redact(response.text)}"
            )
        try:
            parsed = response.json()
        except json.JSONDecodeError as e:
            raise BackendError(f"Phoenix GraphQL returned non-JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise BackendError(f"Phoenix GraphQL returned non-object: {type(parsed).__name__}")
        if "errors" in parsed and parsed["errors"]:
            raise BackendError(f"Phoenix GraphQL errors: {redact(str(parsed['errors']))}")
        data = parsed.get("data")
        if not isinstance(data, dict):
            raise BackendError("Phoenix GraphQL response missing `data` object")
        return cast(dict[str, Any], data)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _decode_phoenix_span(node: dict[str, Any]) -> Span | None:
    """Decode one Phoenix span node; returns None (with a redacted warning)
    when neither timestamp is parseable rather than fabricating epoch values."""
    ctx = node.get("context") or {}
    trace_id = ctx.get("traceId", "")
    span_id = ctx.get("spanId", "")
    parent = node.get("parentId") or None
    attributes = node.get("attributes") or {}
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except json.JSONDecodeError:
            attributes = {}
    if not isinstance(attributes, dict):
        attributes = {}
    status_code_raw = node.get("statusCode") or "UNSET"
    status_code = _normalize_status_code(status_code_raw)
    status = Status(code=status_code, message=node.get("statusMessage"))
    events = [
        Event(
            name=e.get("name", ""),
            time_unix_nano=int(e.get("timeUnixNano", 0)),
            attributes=e.get("attributes") or {},
        )
        for e in node.get("events") or []
    ]
    start_ns = _iso_to_unix_nano(node.get("startTime", ""))
    end_ns = _iso_to_unix_nano(node.get("endTime", ""))
    if start_ns is None:
        start_ns = end_ns  # zero-duration span beats a ~56-year epoch latency
    if end_ns is None:
        end_ns = start_ns
    if start_ns is None or end_ns is None:
        log.warning(
            "Phoenix: skipping span %s in trace %s: no parseable timestamps",
            redact(str(span_id)),
            redact(str(trace_id)),
        )
        return None
    return Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent,
        name=node.get("name", ""),
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        attributes=attributes,
        events=events,
        status=status,
    )


def _normalize_status_code(raw: Any) -> Any:
    if isinstance(raw, str):
        upper = raw.upper()
        if upper in ("OK", "ERROR", "UNSET"):
            return upper
        if upper == "STATUS_CODE_OK":
            return "OK"
        if upper == "STATUS_CODE_ERROR":
            return "ERROR"
    return "UNSET"


def _iso_to_unix_nano(iso: str) -> int | None:
    """ISO-8601 -> unix nanos. Offset-less timestamps are UTC (never host
    local time); unparseable input returns None for the caller to handle."""
    if not iso:
        return None
    cleaned = iso.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)

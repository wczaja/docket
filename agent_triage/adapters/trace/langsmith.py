"""LangSmith trace-backend adapter (design §5.1).

HTTP-only via httpx; no SDK dependency. LangSmith is SaaS-only (no
self-hosted in v1.0), so there's no docker-compose entry; testing uses
mocked transport + the cross-adapter parity check.

Endpoints used (public API at https://api.smith.langchain.com):

  - POST /api/v1/runs/query    (list runs in a time window, optionally
                                filtered to root runs to get trace IDs; also
                                used with a trace filter to fetch child runs)
  - GET  /api/v1/runs/{id}     (fetch a run; LangSmith uses run_id == trace_id
                                for root runs; child runs are NOT included by
                                default, so get_trace falls back to runs/query)
  - POST /api/v1/feedback      (write annotations as feedback objects)

Auth is the `x-api-key` header. The classifier's annotation lands as a
feedback object whose `key` carries the mode id and whose `feedback_source`
+ `metadata` carry the full provenance (run_id, rubric_version,
idempotency_key) for queryable dedup.

LangSmith's `run_type` enumeration maps cleanly onto OpenInference span
kinds: `llm` -> LLM, `tool` -> TOOL, `retriever` -> RETRIEVER, `embedding`
-> EMBEDDING, `chain` / anything else -> CHAIN.
"""

import json
import logging
import uuid
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

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
_DEFAULT_PAGE_LIMIT = 100
_MAX_PAGES = 50
_PROCESSED_SENTINEL_KEY = "agent-triage:processed"


class LangsmithAdapter(TraceBackend):
    def __init__(
        self,
        endpoint: str = DEFAULT_LANGSMITH_ENDPOINT,
        *,
        api_key: str | None = None,
        project: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._project = project
        self._timeout = timeout
        self._client = client
        self._session_id: str | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["x-api-key"] = self._api_key
            self._client = httpx.AsyncClient(
                base_url=self._endpoint,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        idempotent: bool = False,
    ) -> httpx.Response:
        """Send an HTTP request through the shared retry helper.

        429 is always retried (capped Retry-After honored, GMT HTTP-dates);
        5xx/transport errors are retried only when `idempotent=True`.
        """
        return await request_with_retry(
            self._get_client(),
            method,
            url,
            error_cls=BackendError,
            idempotent=idempotent,
            json_body=json_body,
            params=params,
        )

    async def _resolve_session_id(self, client: httpx.AsyncClient) -> str | None:
        """Resolve `self._project` (a session name) to a LangSmith session UUID.

        LangSmith's /runs/query endpoint requires `session` to be a list of
        UUIDs, not names. Names are resolved via /api/v1/sessions?name=...
        and cached for the lifetime of the adapter. If the caller already
        passed a UUID, it's accepted as-is.
        """
        if self._project is None:
            return None
        if self._session_id is not None:
            return self._session_id
        try:
            uuid.UUID(self._project)
            self._session_id = self._project
            return self._session_id
        except ValueError:
            pass
        response = await self._request(
            "GET", "/api/v1/sessions", params={"name": self._project}, idempotent=True
        )
        if response.status_code >= 400:
            raise BackendError(
                f"LangSmith sessions lookup for {self._project!r} failed with "
                f"{response.status_code}: {redact(response.text)}"
            )
        try:
            parsed = response.json()
        except json.JSONDecodeError as e:
            raise BackendError(f"LangSmith sessions lookup returned non-JSON: {e}") from e
        if isinstance(parsed, list):
            sessions = parsed
        elif isinstance(parsed, dict):
            sessions = parsed.get("sessions") or parsed.get("data") or []
        else:
            sessions = []
        for session in sessions:
            if (
                isinstance(session, dict)
                and session.get("name") == self._project
                and isinstance(session.get("id"), str)
            ):
                self._session_id = session["id"]
                return self._session_id
        raise BackendError(
            f"LangSmith project {self._project!r} not found (no session with that name)."
        )

    async def list_traces(
        self,
        since: datetime,
        until: datetime | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[str]:
        """LangSmith calls them 'runs' -- we list root runs whose IDs serve
        as trace IDs in the rest of the agent-triage runtime."""
        del filter
        client = self._get_client()
        session_id = await self._resolve_session_id(client)
        ids: list[str] = []
        seen: set[str] = set()
        for offset in range(_MAX_PAGES):
            body: dict[str, Any] = {
                "start_time": _to_iso(since),
                "is_root": True,
                "limit": _DEFAULT_PAGE_LIMIT,
                "offset": offset * _DEFAULT_PAGE_LIMIT,
            }
            if until is not None:
                body["end_time"] = _to_iso(until)
            if session_id:
                body["session"] = [session_id]
            # runs/query is a read despite the POST verb — idempotent.
            response = await self._request(
                "POST", "/api/v1/runs/query", json_body=body, idempotent=True
            )
            if response.status_code >= 400:
                raise BackendError(
                    f"LangSmith runs/query failed with "
                    f"{response.status_code}: {redact(response.text)}"
                )
            data = _parse_json(response, context="list_traces")
            page_items = data.get("runs") or data.get("data") or []
            if not isinstance(page_items, list):
                raise BackendError(
                    f"LangSmith runs/query returned non-list runs: {type(page_items).__name__}"
                )
            for item in page_items:
                run_id = item.get("id") if isinstance(item, dict) else None
                if isinstance(run_id, str) and run_id not in seen:
                    seen.add(run_id)
                    ids.append(run_id)
            if len(page_items) < _DEFAULT_PAGE_LIMIT:
                break
        return ids

    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        response = await self._request("GET", f"/api/v1/runs/{trace_id}", idempotent=True)
        if response.status_code >= 400:
            raise BackendError(
                f"LangSmith get_trace({trace_id!r}) failed with "
                f"{response.status_code}: {redact(response.text)}"
            )
        run = _parse_json(response, context="get_trace")
        root_span = _run_to_span(run, trace_id, parent_run_id=None)
        spans: list[Span] = [root_span] if root_span is not None else []
        children = run.get("child_runs") or []
        if isinstance(children, list) and children:
            for child in children:
                if isinstance(child, dict):
                    spans.extend(_walk_run_tree(child, trace_id, parent_run_id=run.get("id")))
        else:
            # GET /runs/{id} doesn't include child runs by default — without
            # this follow-up query, deep traces would silently collapse to
            # the root span. If the query returns only the root, it's a
            # genuine single-span trace and we proceed.
            for child_run in await self._query_trace_runs(trace_id):
                if child_run.get("id") in (trace_id, run.get("id")):
                    continue  # root already decoded (with full detail) above
                child_span = _run_to_span(
                    child_run, trace_id, parent_run_id=child_run.get("parent_run_id")
                )
                if child_span is not None:
                    spans.append(child_span)
        if not spans:
            raise BackendError(
                f"LangSmith trace {trace_id!r} has no spans with parseable timestamps"
            )
        if len(spans) == 1 and not run.get("name"):
            raise BackendError(f"LangSmith trace {trace_id!r} returned an empty run")
        return OpenInferenceTrace(trace_id=trace_id, spans=spans)

    async def _query_trace_runs(self, trace_id: str) -> list[dict[str, Any]]:
        """Fetch every run in a trace via POST /runs/query with the trace
        filter, paginating like `list_traces`."""
        runs: list[dict[str, Any]] = []
        for offset in range(_MAX_PAGES):
            body: dict[str, Any] = {
                "trace": trace_id,
                "limit": _DEFAULT_PAGE_LIMIT,
                "offset": offset * _DEFAULT_PAGE_LIMIT,
            }
            # runs/query is a read despite the POST verb — idempotent.
            response = await self._request(
                "POST", "/api/v1/runs/query", json_body=body, idempotent=True
            )
            if response.status_code >= 400:
                raise BackendError(
                    f"LangSmith runs/query for trace {trace_id!r} failed with "
                    f"{response.status_code}: {redact(response.text)}"
                )
            data = _parse_json(response, context="get_trace")
            page_items = data.get("runs") or data.get("data") or []
            if not isinstance(page_items, list):
                raise BackendError(
                    f"LangSmith runs/query returned non-list runs: {type(page_items).__name__}"
                )
            runs.extend(item for item in page_items if isinstance(item, dict))
            if len(page_items) < _DEFAULT_PAGE_LIMIT:
                break
        return runs

    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        body = {
            # LangSmith upserts feedback by client-supplied id; deriving it
            # from the idempotency key makes re-runs overwrite, not duplicate.
            "id": _deterministic_id(annotation.idempotency_key()),
            "run_id": trace_id,
            "key": f"agent-triage:{annotation.mode_id}",
            "value": "positive" if annotation.positive else "negative",
            "score": annotation.confidence,
            "comment": annotation.excerpt,
            "feedback_source": {"type": "model", "metadata": {"source": "agent-triage"}},
            "extra": {
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
        # The deterministic client-supplied id makes this POST a true upsert,
        # so a retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await self._request("POST", "/api/v1/feedback", json_body=body, idempotent=True)
        if response.status_code >= 400:
            raise BackendError(
                f"LangSmith feedback POST failed with "
                f"{response.status_code}: {redact(response.text)}"
            )

    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        del query, k
        raise NotImplementedError(
            "LangSmith semantic search is not exposed in v1.0; use `list_traces` with a "
            "time window and post-filter in-process."
        )

    async def mark_trace_processed(
        self,
        trace_id: str,
        *,
        run_id: str,
        rubric_version: str,
    ) -> None:
        sentinel_key = f"{trace_id}|{run_id}|{rubric_version}|{_PROCESSED_SENTINEL_KEY}"
        body = {
            # Same upsert mechanics as annotate_trace: re-marking the same
            # (trace, run, rubric) is a no-op, not a duplicate.
            "id": _deterministic_id(sentinel_key),
            "run_id": trace_id,
            "key": _PROCESSED_SENTINEL_KEY,
            "value": "processed",
            "score": 1.0,
            "comment": "agent-triage Phase 11 resumability sentinel",
            "feedback_source": {"type": "model", "metadata": {"source": "agent-triage"}},
            "extra": {
                "run_id": run_id,
                "rubric_version": rubric_version,
            },
        }
        # The deterministic client-supplied id makes this POST a true upsert,
        # so a retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await self._request("POST", "/api/v1/feedback", json_body=body, idempotent=True)
        if response.status_code >= 400:
            raise BackendError(
                f"LangSmith: sentinel feedback POST failed with "
                f"{response.status_code}: {redact(response.text)}"
            )

    async def list_processed_trace_ids(
        self,
        *,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> set[str]:
        """Query /api/v1/feedback for sentinel records, paginate, and return
        the LangSmith run IDs (= trace IDs in agent-triage's vocabulary) where
        `extra.run_id` matches the agent-triage run_id we care about.
        """
        processed: set[str] = set()
        for offset in range(_MAX_PAGES):
            params: dict[str, Any] = {
                "key": _PROCESSED_SENTINEL_KEY,
                "limit": _DEFAULT_PAGE_LIMIT,
                "offset": offset * _DEFAULT_PAGE_LIMIT,
                "start_time": _to_iso(since),
            }
            if until is not None:
                params["end_time"] = _to_iso(until)
            response = await self._request(
                "GET", "/api/v1/feedback", params=params, idempotent=True
            )
            if response.status_code >= 400:
                raise BackendError(
                    f"LangSmith feedback listing failed with "
                    f"{response.status_code}: {redact(response.text)}"
                )
            try:
                data = response.json()
            except ValueError as e:
                raise BackendError(f"LangSmith feedback listing returned non-JSON: {e}") from e
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("feedbacks") or data.get("data") or []
            else:
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                extra = item.get("extra") or {}
                if not (isinstance(extra, dict) and extra.get("run_id") == run_id):
                    continue
                tid = item.get("run_id")
                if isinstance(tid, str):
                    processed.add(tid)
            if not isinstance(items, list) or len(items) < _DEFAULT_PAGE_LIMIT:
                break
        return processed


def _to_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _deterministic_id(idempotency_key: str) -> str:
    """Stable UUIDv5 derived from the idempotency key, used as the
    client-supplied feedback id that LangSmith upserts on."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key))


def _parse_json(response: httpx.Response, *, context: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as e:
        raise BackendError(f"LangSmith {context} returned non-JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise BackendError(f"LangSmith {context} returned non-object: {type(parsed).__name__}")
    return cast(dict[str, Any], parsed)


_RUN_TYPE_KIND_MAP: dict[str, str] = {
    "llm": "LLM",
    "chat_model": "LLM",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "embedding": "EMBEDDING",
    "chain": "CHAIN",
    "parser": "CHAIN",
    "agent": "AGENT",
}


def _walk_run_tree(
    run: dict[str, Any],
    trace_id: str,
    *,
    parent_run_id: Any,
) -> list[Span]:
    span = _run_to_span(run, trace_id, parent_run_id=parent_run_id)
    spans = [span] if span is not None else []
    children = run.get("child_runs") or []
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                spans.extend(_walk_run_tree(child, trace_id, parent_run_id=run.get("id")))
    return spans


def _run_to_span(run: dict[str, Any], trace_id: str, *, parent_run_id: Any) -> Span | None:
    """Heuristic LangSmith run -> OpenInference Span translation.

    Returns None (with a redacted warning) when neither timestamp is
    parseable rather than fabricating epoch values."""
    span_id = str(run.get("id", ""))
    parent = str(parent_run_id) if isinstance(parent_run_id, str) else None
    run_type = str(run.get("run_type", "chain")).lower()
    kind = _RUN_TYPE_KIND_MAP.get(run_type, "CHAIN")
    name = str(run.get("name", run_type))
    start_ns = _iso_to_unix_nano(run.get("start_time", ""))
    end_ns = _iso_to_unix_nano(run.get("end_time", ""))
    if start_ns is None:
        start_ns = end_ns  # zero-duration span beats a ~56-year epoch latency
    if end_ns is None:
        end_ns = start_ns
    if start_ns is None or end_ns is None:
        log.warning(
            "LangSmith: skipping run %s in trace %s: no parseable timestamps",
            redact(span_id),
            redact(str(trace_id)),
        )
        return None

    attributes: dict[str, Any] = {"openinference.span.kind": kind}
    extra = run.get("extra") or {}
    model_name = None
    if isinstance(extra, dict):
        invocation = extra.get("invocation_params")
        if isinstance(invocation, dict):
            model_candidate = invocation.get("model") or invocation.get("model_name")
            if isinstance(model_candidate, str):
                model_name = model_candidate
    serialized = run.get("serialized") or {}
    if model_name is None and isinstance(serialized, dict):
        candidate = serialized.get("name") or serialized.get("id")
        if isinstance(candidate, str) and run_type in ("llm", "chat_model"):
            model_name = candidate
    if isinstance(model_name, str):
        attributes["llm.model_name"] = model_name

    usage = (
        (run.get("outputs") or {}).get("llm_output", {})
        if isinstance(run.get("outputs"), dict)
        else {}
    )
    if isinstance(usage, dict):
        token_usage = (
            usage.get("token_usage") if isinstance(usage.get("token_usage"), dict) else None
        )
        if token_usage and isinstance(token_usage.get("total_tokens"), (int, float)):
            attributes["llm.token_count.total"] = int(token_usage["total_tokens"])

    inputs = run.get("inputs")
    outputs = run.get("outputs")
    if kind == "LLM":
        _attach_llm_messages(attributes, inputs, outputs)
    elif kind == "TOOL":
        attributes["tool.name"] = name
        if inputs is not None:
            attributes["tool.parameters"] = _stringify(inputs)
        if outputs is not None:
            attributes["output.value"] = _stringify(outputs)

    # Absent status normalizes to UNSET (parity with the Phoenix adapter);
    # OK only when LangSmith explicitly reports success.
    status_code = "UNSET"
    status_message = None
    error = run.get("error")
    if isinstance(error, str) and error:
        status_code = "ERROR"
        status_message = error
    elif run.get("status") == "error":
        status_code = "ERROR"
    elif run.get("status") == "success":
        status_code = "OK"

    return Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        attributes=attributes,
        events=_extract_events(run),
        status=Status(code=cast(Any, status_code), message=status_message),
    )


def _attach_llm_messages(
    attributes: dict[str, Any],
    inputs: Any,
    outputs: Any,
) -> None:
    """Best-effort message extraction. LangSmith stores inputs/outputs as
    nested dicts; the shape varies by integration."""
    messages_in = None
    if isinstance(inputs, dict):
        messages_in = inputs.get("messages") or inputs.get("input")
    if isinstance(messages_in, list):
        for i, msg in enumerate(messages_in):
            role, content = _extract_role_content(msg)
            if role is not None:
                attributes[f"llm.input_messages.{i}.message.role"] = role
            if content is not None:
                attributes[f"llm.input_messages.{i}.message.content"] = content
    elif isinstance(inputs, dict) and isinstance(inputs.get("input"), str):
        attributes["llm.input_messages.0.message.role"] = "user"
        attributes["llm.input_messages.0.message.content"] = inputs["input"]

    output_text = None
    output_role = "assistant"
    if isinstance(outputs, dict):
        generations = outputs.get("generations")
        if isinstance(generations, list) and generations:
            first_choice = generations[0]
            if isinstance(first_choice, list) and first_choice:
                first_choice = first_choice[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    role, content = _extract_role_content(message)
                    output_role = role or output_role
                    output_text = content
                else:
                    text = first_choice.get("text")
                    if isinstance(text, str):
                        output_text = text
        if output_text is None:
            output_value = outputs.get("output") or outputs.get("content")
            if isinstance(output_value, str):
                output_text = output_value
    elif isinstance(outputs, str):
        output_text = outputs

    if output_text is not None:
        attributes["llm.output_messages.0.message.role"] = output_role
        attributes["llm.output_messages.0.message.content"] = output_text


def _extract_role_content(msg: Any) -> tuple[str | None, str | None]:
    if not isinstance(msg, dict):
        return None, None
    role = msg.get("role") or msg.get("type")
    content = msg.get("content")
    return (
        str(role) if isinstance(role, str) else None,
        content if isinstance(content, str) else None,
    )


def _extract_events(run: dict[str, Any]) -> list[Event]:
    events_raw = run.get("events") or []
    if not isinstance(events_raw, list):
        return []
    out: list[Event] = []
    for e in events_raw:
        if not isinstance(e, dict):
            continue
        metadata = e.get("metadata")
        attrs: dict[str, Any] = metadata if isinstance(metadata, dict) else {}
        out.append(
            Event(
                name=str(e.get("name", "event")),
                time_unix_nano=_iso_to_unix_nano(e.get("time", "")) or 0,
                attributes=attrs,
            )
        )
    return out


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return str(v)


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

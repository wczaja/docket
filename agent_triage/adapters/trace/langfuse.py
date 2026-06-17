"""Langfuse trace-backend adapter (design §5.1).

Talks to Langfuse's public REST API directly via httpx; no SDK dependency.
Endpoints used:

  - GET  /api/public/traces            (list with timestamp + page filters)
  - GET  /api/public/traces/{traceId}  (single trace with nested observations)
  - POST /api/public/scores            (write annotations as Langfuse scores)

Auth is HTTP Basic with `(public_key, secret_key)`. The classifier's
annotation lands as a Langfuse score whose `name` carries the mode id and
whose `metadata` carries the full provenance (`run_id`, `rubric_version`,
idempotency key) for queryable dedup.

Langfuse observations don't map 1:1 to OpenInference spans -- we apply a
small heuristic translation that mirrors what the openinference-langfuse
exporters do: GENERATION -> LLM, the rest become SPAN/CHAIN with our
attribute namespaces filled in best-effort from the Langfuse input/output
shape.
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

_DEFAULT_PAGE_SIZE = 100
_DEFAULT_PAGE_LIMIT = 100
_MAX_PAGES = 50  # safety cap to avoid runaway pagination
_PROCESSED_SENTINEL_NAME = "agent-triage:processed"


class LangfuseAdapter(TraceBackend):
    def __init__(
        self,
        host: str,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._public_key = public_key
        self._secret_key = secret_key
        self._timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            auth: httpx.BasicAuth | None = None
            if self._public_key and self._secret_key:
                auth = httpx.BasicAuth(self._public_key, self._secret_key)
            self._client = httpx.AsyncClient(
                base_url=self._host,
                auth=auth,
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
        del filter  # Langfuse filter passthrough lands when a real use-case shows up
        client = self._get_client()
        ids: list[str] = []
        seen: set[str] = set()
        for page in range(1, _MAX_PAGES + 1):
            params: dict[str, Any] = {
                "fromTimestamp": _to_iso(since),
                "limit": _DEFAULT_PAGE_SIZE,
                "page": page,
            }
            if until is not None:
                params["toTimestamp"] = _to_iso(until)
            # Reads are idempotent — safe to retry on 5xx/timeouts too.
            response = await request_with_retry(
                client,
                "GET",
                "/api/public/traces",
                error_cls=BackendError,
                idempotent=True,
                params=params,
            )
            if response.status_code >= 400:
                raise BackendError(
                    f"Langfuse list_traces failed with "
                    f"{response.status_code}: {redact(response.text)}"
                )
            data = _parse_json(response, context="list_traces")
            page_items = data.get("data") or []
            if not isinstance(page_items, list):
                raise BackendError(
                    f"Langfuse list_traces returned non-list `data`: {type(page_items).__name__}"
                )
            for item in page_items:
                trace_id = item.get("id") if isinstance(item, dict) else None
                if isinstance(trace_id, str) and trace_id not in seen:
                    seen.add(trace_id)
                    ids.append(trace_id)
            if len(page_items) < _DEFAULT_PAGE_SIZE:
                break
        return ids

    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        client = self._get_client()
        response = await request_with_retry(
            client,
            "GET",
            f"/api/public/traces/{trace_id}",
            error_cls=BackendError,
            idempotent=True,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Langfuse get_trace({trace_id!r}) failed with "
                f"{response.status_code}: {redact(response.text)}"
            )
        data = _parse_json(response, context="get_trace")
        observations = data.get("observations") or []
        if not isinstance(observations, list):
            raise BackendError(
                f"Langfuse trace {trace_id!r}: observations is not a list "
                f"({type(observations).__name__})"
            )
        spans = [
            span
            for obs in observations
            if (span := _observation_to_span(obs, trace_id)) is not None
        ]
        if not spans:
            raise BackendError(f"Langfuse trace {trace_id!r} has no observations")
        return OpenInferenceTrace(trace_id=trace_id, spans=spans)

    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        client = self._get_client()
        body = {
            # Langfuse upserts scores by client-supplied id; deriving it from
            # the idempotency key makes re-runs overwrite instead of duplicate.
            "id": _deterministic_id(annotation.idempotency_key()),
            "traceId": trace_id,
            "name": f"agent-triage:{annotation.mode_id}",
            "value": 1.0 if annotation.positive else 0.0,
            "dataType": "NUMERIC",
            "comment": annotation.excerpt,
            "metadata": {
                # Notes first; provenance keys last so a stray note
                # (e.g. keyed "run_id") can't clobber them.
                **annotation.notes,
                "run_id": annotation.run_id,
                "rubric_version": annotation.rubric_version,
                "mode_id": annotation.mode_id,
                "severity": annotation.severity,
                "confidence": annotation.confidence,
                "idempotency_key": annotation.idempotency_key(),
            },
        }
        # The deterministic client-supplied id makes this POST a true upsert,
        # so a retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await request_with_retry(
            client,
            "POST",
            "/api/public/scores",
            error_cls=BackendError,
            idempotent=True,
            json_body=body,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Langfuse score POST failed with {response.status_code}: {redact(response.text)}"
            )

    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        del query, k
        raise NotImplementedError(
            "Langfuse semantic search is not exposed via the public REST API in v1.0; "
            "use `list_traces` with a time window and post-filter in-process."
        )

    async def mark_trace_processed(
        self,
        trace_id: str,
        *,
        run_id: str,
        rubric_version: str,
    ) -> None:
        client = self._get_client()
        sentinel_key = f"{trace_id}|{run_id}|{rubric_version}|{_PROCESSED_SENTINEL_NAME}"
        body = {
            # Same upsert mechanics as annotate_trace: re-marking the same
            # (trace, run, rubric) is a no-op, not a duplicate.
            "id": _deterministic_id(sentinel_key),
            "traceId": trace_id,
            "name": _PROCESSED_SENTINEL_NAME,
            "value": 1.0,
            "dataType": "NUMERIC",
            "comment": "agent-triage Phase 11 resumability sentinel",
            "metadata": {
                "run_id": run_id,
                "rubric_version": rubric_version,
            },
        }
        # The deterministic client-supplied id makes this POST a true upsert,
        # so a retry after a 5xx/timeout cannot duplicate — idempotent.
        response = await request_with_retry(
            client,
            "POST",
            "/api/public/scores",
            error_cls=BackendError,
            idempotent=True,
            json_body=body,
        )
        if response.status_code >= 400:
            raise BackendError(
                f"Langfuse: sentinel score POST failed with "
                f"{response.status_code}: {redact(response.text)}"
            )

    async def list_processed_trace_ids(
        self,
        *,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> set[str]:
        client = self._get_client()
        processed: set[str] = set()
        page = 1
        while page <= _MAX_PAGES:
            params: dict[str, Any] = {
                "name": _PROCESSED_SENTINEL_NAME,
                "fromTimestamp": _to_iso(since),
                "page": page,
                "limit": _DEFAULT_PAGE_LIMIT,
            }
            if until is not None:
                params["toTimestamp"] = _to_iso(until)
            # Reads are idempotent — safe to retry on 5xx/timeouts too.
            response = await request_with_retry(
                client,
                "GET",
                "/api/public/scores",
                error_cls=BackendError,
                idempotent=True,
                params=params,
            )
            if response.status_code >= 400:
                raise BackendError(
                    f"Langfuse scores listing failed with "
                    f"{response.status_code}: {redact(response.text)}"
                )
            data = _parse_json(response, context="list_processed_trace_ids")
            items = data.get("data") or []
            if not isinstance(items, list):
                raise BackendError(
                    f"Langfuse scores listing returned non-list `data`: {type(items).__name__}"
                )
            for item in items:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("run_id") == run_id:
                    tid = item.get("traceId")
                    if isinstance(tid, str):
                        processed.add(tid)
            if len(items) < _DEFAULT_PAGE_LIMIT:
                break
            page += 1
        return processed


def _to_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _deterministic_id(idempotency_key: str) -> str:
    """Stable UUIDv5 derived from the idempotency key, used as the
    client-supplied score id that Langfuse upserts on."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key))


def _parse_json(response: httpx.Response, *, context: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as e:
        raise BackendError(f"Langfuse {context} returned non-JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise BackendError(f"Langfuse {context} returned non-object: {type(parsed).__name__}")
    return cast(dict[str, Any], parsed)


_OBSERVATION_KIND_MAP: dict[str, str] = {
    "GENERATION": "LLM",
    "SPAN": "CHAIN",
    "EVENT": "CHAIN",
    "AGENT": "AGENT",
    "TOOL": "TOOL",
    "EMBEDDING": "EMBEDDING",
    "RETRIEVER": "RETRIEVER",
}


def _observation_to_span(obs: dict[str, Any], trace_id: str) -> Span | None:
    """Heuristic Langfuse observation -> OpenInference Span translation.

    Returns None (with a redacted warning) when neither timestamp is
    parseable rather than fabricating epoch values."""
    span_id = str(obs.get("id", ""))
    parent = obs.get("parentObservationId") or None
    obs_type = str(obs.get("type", "SPAN")).upper()
    kind = _OBSERVATION_KIND_MAP.get(obs_type, "CHAIN")
    name = str(obs.get("name", obs_type.lower()))
    start_ns = _iso_to_unix_nano(obs.get("startTime", ""))
    end_ns = _iso_to_unix_nano(obs.get("endTime", ""))
    if start_ns is None:
        start_ns = end_ns  # zero-duration span beats a ~56-year epoch latency
    if end_ns is None:
        end_ns = start_ns
    if start_ns is None or end_ns is None:
        log.warning(
            "Langfuse: skipping observation %s in trace %s: no parseable timestamps",
            redact(span_id),
            redact(str(trace_id)),
        )
        return None

    attributes: dict[str, Any] = {"openinference.span.kind": kind}
    model = obs.get("model")
    if isinstance(model, str):
        attributes["llm.model_name"] = model
    usage = obs.get("usage") or {}
    if isinstance(usage, dict):
        total = usage.get("total") or usage.get("totalTokens")
        if isinstance(total, (int, float)):
            attributes["llm.token_count.total"] = int(total)

    input_value = obs.get("input")
    output_value = obs.get("output")
    if kind == "LLM":
        _attach_llm_messages(attributes, input_value, output_value)
    elif kind == "TOOL":
        attributes["tool.name"] = name
        if input_value is not None:
            attributes["tool.parameters"] = _stringify(input_value)
        if output_value is not None:
            attributes["output.value"] = _stringify(output_value)

    # Langfuse has no explicit success signal, so absent/non-error levels
    # normalize to UNSET (parity with the Phoenix adapter), never OK.
    status_code = "UNSET"
    status_message = None
    level = obs.get("level")
    if isinstance(level, str) and level.upper() in ("ERROR", "FATAL"):
        status_code = "ERROR"
        message = obs.get("statusMessage")
        if isinstance(message, str):
            status_message = message
    status = Status(code=cast(Any, status_code), message=status_message)

    return Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        attributes=attributes,
        events=_extract_events(obs),
        status=status,
    )


def _attach_llm_messages(
    attributes: dict[str, Any],
    input_value: Any,
    output_value: Any,
) -> None:
    """Best-effort LLM message extraction from Langfuse input/output payloads."""
    if isinstance(input_value, list):
        for i, msg in enumerate(input_value):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(role, str):
                attributes[f"llm.input_messages.{i}.message.role"] = role
            if isinstance(content, str):
                attributes[f"llm.input_messages.{i}.message.content"] = content
    elif isinstance(input_value, str):
        attributes["llm.input_messages.0.message.role"] = "user"
        attributes["llm.input_messages.0.message.content"] = input_value

    if isinstance(output_value, dict):
        content = output_value.get("content")
        role = output_value.get("role", "assistant")
        if isinstance(content, str):
            attributes["llm.output_messages.0.message.role"] = str(role)
            attributes["llm.output_messages.0.message.content"] = content
    elif isinstance(output_value, str):
        attributes["llm.output_messages.0.message.role"] = "assistant"
        attributes["llm.output_messages.0.message.content"] = output_value


def _extract_events(obs: dict[str, Any]) -> list[Event]:
    events_raw = obs.get("events") or []
    if not isinstance(events_raw, list):
        return []
    out: list[Event] = []
    for e in events_raw:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "event"))
        ts = _iso_to_unix_nano(e.get("timestamp", "")) or 0
        attrs = e.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        out.append(Event(name=name, time_unix_nano=ts, attributes=attrs))
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

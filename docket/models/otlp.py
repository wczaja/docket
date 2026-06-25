"""OTLP <-> OpenInferenceTrace conversion.

OTLP JSON uses a nested resourceSpans / scopeSpans / spans structure with
typed attribute values (`stringValue`, `intValue`, `doubleValue`, `boolValue`,
`arrayValue`, `kvlistValue`). This module flattens the typed-wrapper encoding
into Python primitives on `from_otlp` and reverses it on `to_otlp` so the
two are round-trippable.

OTLP encodes int64 attribute values as JSON strings (the wire format avoids
loss of precision); we decode to `int` and re-encode as string on output.
"""

import base64
from typing import Any, cast

from docket.errors import BackendError
from docket.models.trace import Event, OpenInferenceTrace, Span, Status

_STATUS_CODE_FROM_OTLP = {
    "STATUS_CODE_OK": "OK",
    "STATUS_CODE_ERROR": "ERROR",
    "STATUS_CODE_UNSET": "UNSET",
    "OK": "OK",
    "ERROR": "ERROR",
    "UNSET": "UNSET",
    0: "UNSET",
    1: "OK",
    2: "ERROR",
}
_STATUS_CODE_TO_OTLP = {
    "OK": "STATUS_CODE_OK",
    "ERROR": "STATUS_CODE_ERROR",
    "UNSET": "STATUS_CODE_UNSET",
}


def from_otlp(otlp: dict[str, Any]) -> OpenInferenceTrace:
    """Parse an OTLP JSON payload into an OpenInferenceTrace.

    The payload must contain spans for exactly one trace; a multi-trace
    payload raises rather than silently mislabeling foreign spans under
    the first span's trace id.
    """
    all_spans: list[Span] = []
    trace_ids: set[str] = set()
    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                span = _decode_span(s)
                trace_ids.add(span.trace_id)
                all_spans.append(span)
    if not trace_ids:
        raise BackendError("OTLP payload contained no spans")
    if len(trace_ids) > 1:
        raise BackendError(
            f"OTLP payload contained spans from {len(trace_ids)} distinct traces; "
            "from_otlp expects exactly one trace per payload"
        )
    return OpenInferenceTrace(trace_id=all_spans[0].trace_id, spans=all_spans)


def to_otlp(trace: OpenInferenceTrace) -> dict[str, Any]:
    """Serialize an OpenInferenceTrace back to OTLP JSON shape."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "openinference"},
                        "spans": [_encode_span(s) for s in trace.spans],
                    }
                ],
            }
        ]
    }


def to_otlp_protobuf(trace: OpenInferenceTrace) -> bytes:
    """Serialize a trace to an OTLP/protobuf ``ExportTraceServiceRequest``.

    OTLP/HTTP collectors — Phoenix's ``/v1/traces`` among them — accept
    protobuf (``application/x-protobuf``), not JSON. Trace and span ids must be
    valid hex (16-byte trace, 8-byte span); they are decoded to the raw bytes
    the wire format requires. Needs ``opentelemetry-proto`` + ``protobuf``,
    which ship transitively with ``arize-phoenix-otel`` (a core dependency).
    """
    # Imported lazily: the protobuf stack is only needed for wire export, not
    # for the JSON round-trip the rest of this module does.
    from google.protobuf import json_format
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    payload = to_otlp(trace)
    for resource_spans in payload["resourceSpans"]:
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                span["traceId"] = _hex_to_base64(span["traceId"])
                span["spanId"] = _hex_to_base64(span["spanId"])
                if span.get("parentSpanId"):
                    span["parentSpanId"] = _hex_to_base64(span["parentSpanId"])
    request = json_format.ParseDict(payload, ExportTraceServiceRequest())
    return cast(bytes, request.SerializeToString())


def _hex_to_base64(hex_id: str) -> str:
    """Re-encode a hex id as base64 for OTLP/JSON ``bytes``-field parsing."""
    return base64.b64encode(bytes.fromhex(hex_id)).decode("ascii")


def _decode_span(s: dict[str, Any]) -> Span:
    parent = s.get("parentSpanId") or None
    try:
        return Span(
            span_id=s["spanId"],
            trace_id=s["traceId"],
            parent_span_id=parent,
            name=s["name"],
            start_time_unix_nano=int(s["startTimeUnixNano"]),
            end_time_unix_nano=int(s["endTimeUnixNano"]),
            attributes=_decode_attributes(s.get("attributes", [])),
            events=[_decode_event(e) for e in s.get("events", [])],
            status=_decode_status(s.get("status", {})),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise BackendError(f"OTLP span is missing or has a malformed required field: {e!r}") from e


def _decode_event(e: dict[str, Any]) -> Event:
    return Event(
        name=e["name"],
        time_unix_nano=int(e.get("timeUnixNano", 0)),
        attributes=_decode_attributes(e.get("attributes", [])),
    )


def _decode_status(s: dict[str, Any]) -> Status:
    raw = s.get("code", "UNSET")
    code = _STATUS_CODE_FROM_OTLP.get(raw, "UNSET")
    return Status(code=cast(Any, code), message=s.get("message"))


def _decode_attributes(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in attrs:
        out[a["key"]] = _decode_value(a["value"])
    return out


def _decode_value(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_decode_value(item) for item in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {
            item["key"]: _decode_value(item["value"]) for item in v["kvlistValue"].get("values", [])
        }
    # Empty AnyValue ({}) is OTLP's encoding of an absent/null value.
    return None


def _encode_span(span: Span) -> dict[str, Any]:
    out: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "startTimeUnixNano": str(span.start_time_unix_nano),
        "endTimeUnixNano": str(span.end_time_unix_nano),
        "attributes": _encode_attributes(span.attributes),
        "events": [_encode_event(e) for e in span.events],
        "status": _encode_status(span.status),
    }
    if span.parent_span_id:
        out["parentSpanId"] = span.parent_span_id
    return out


def _encode_event(e: Event) -> dict[str, Any]:
    return {
        "name": e.name,
        "timeUnixNano": str(e.time_unix_nano),
        "attributes": _encode_attributes(e.attributes),
    }


def _encode_status(s: Status) -> dict[str, Any]:
    out: dict[str, Any] = {"code": _STATUS_CODE_TO_OTLP[s.code]}
    if s.message is not None:
        out["message"] = s.message
    return out


def _encode_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _encode_value(v)} for k, v in attrs.items()]


def _encode_value(v: Any) -> dict[str, Any]:
    if v is None:
        # Mirror of _decode_value's empty-AnyValue handling: None round-trips
        # as {} instead of becoming the string "None".
        return {}
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_encode_value(item) for item in v]}}
    if isinstance(v, dict):
        return {
            "kvlistValue": {
                "values": [{"key": k, "value": _encode_value(val)} for k, val in v.items()]
            }
        }
    return {"stringValue": str(v)}

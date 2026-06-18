"""OTLP <-> OpenInferenceTrace round-trip tests against the shared fixtures.

The §7 Phase 3 acceptance criterion is "models round-trip valid OpenInference
OTLP spans without data loss". Round-trip here means: decode the fixture into
the Pydantic model, re-encode it back into OTLP, decode that, compare the two
decoded traces. The second decode should equal the first.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from docket.errors import BackendError
from docket.models.otlp import from_otlp, to_otlp


def _minimal_otlp_span(span_id: str, trace_id: str) -> dict[str, Any]:
    return {
        "spanId": span_id,
        "traceId": trace_id,
        "name": "synthetic",
        "startTimeUnixNano": "1000",
        "endTimeUnixNano": "2000",
        "attributes": [],
    }


FIXTURES = (
    "simple_llm_call.json",
    "tool_calling_agent.json",
    "retrieval_augmented.json",
    "multi_agent.json",
    "error_trace.json",
    "embedding_trace.json",
)


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_round_trip_preserves_trace(traces_dir: Path, fixture_name: str) -> None:
    otlp = json.loads((traces_dir / fixture_name).read_text())
    trace1 = from_otlp(otlp)
    re_encoded = to_otlp(trace1)
    trace2 = from_otlp(re_encoded)
    assert trace1 == trace2


def test_from_otlp_extracts_trace_id(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "simple_llm_call.json").read_text())
    trace = from_otlp(otlp)
    assert trace.trace_id == "trace-simple-001"
    assert len(trace.spans) == 1


def test_from_otlp_decodes_int_value_as_python_int(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "simple_llm_call.json").read_text())
    trace = from_otlp(otlp)
    total = trace.spans[0].attributes["llm.token_count.total"]
    assert total == 21
    assert isinstance(total, int)


def test_from_otlp_decodes_double_value_as_python_float(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "retrieval_augmented.json").read_text())
    trace = from_otlp(otlp)
    retriever = next(s for s in trace.spans if s.kind == "RETRIEVER")
    score = retriever.attributes["retrieval.documents.0.document.score"]
    assert isinstance(score, float)
    assert score == pytest.approx(0.91)


def test_from_otlp_decodes_array_value(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "embedding_trace.json").read_text())
    trace = from_otlp(otlp)
    emb_span = next(s for s in trace.spans if s.kind == "EMBEDDING")
    vector = emb_span.attributes["embedding.embeddings.0.embedding.vector"]
    assert vector == [0.01, -0.02, 0.03]


def test_from_otlp_decodes_status_codes(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "error_trace.json").read_text())
    trace = from_otlp(otlp)
    root = next(s for s in trace.spans if s.span_id == "span-root")
    assert root.status.code == "ERROR"
    assert root.status.message == "tool call rejected by safety filter"
    llm = next(s for s in trace.spans if s.span_id == "span-llm")
    assert llm.status.code == "OK"


def test_from_otlp_preserves_parent_span_id(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "tool_calling_agent.json").read_text())
    trace = from_otlp(otlp)
    tool = next(s for s in trace.spans if s.kind == "TOOL")
    assert tool.parent_span_id == "span-agent"


def test_from_otlp_decodes_events(traces_dir: Path) -> None:
    otlp = json.loads((traces_dir / "error_trace.json").read_text())
    trace = from_otlp(otlp)
    tool = next(s for s in trace.spans if s.kind == "TOOL")
    assert len(tool.events) == 1
    assert tool.events[0].name == "exception"
    assert tool.events[0].attributes["exception.type"] == "SafetyRejection"


def test_from_otlp_raises_on_empty_payload() -> None:
    with pytest.raises(BackendError, match="no spans"):
        from_otlp({"resourceSpans": []})


def test_to_otlp_encodes_int_as_string() -> None:
    from docket.models.trace import Span

    trace_dict = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "openinference"},
                        "spans": [
                            {
                                "traceId": "t",
                                "spanId": "s",
                                "name": "x",
                                "startTimeUnixNano": "1",
                                "endTimeUnixNano": "2",
                                "attributes": [],
                                "status": {"code": "STATUS_CODE_OK"},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    trace = from_otlp(trace_dict)
    trace.spans.append(
        Span(
            span_id="s2",
            trace_id="t",
            name="y",
            start_time_unix_nano=10,
            end_time_unix_nano=20,
            attributes={"some.int": 12345},
        )
    )
    encoded = to_otlp(trace)
    second_span = encoded["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
    int_attr = next(a for a in second_span["attributes"] if a["key"] == "some.int")
    assert int_attr["value"] == {"intValue": "12345"}


def test_to_otlp_encodes_bool_distinct_from_int() -> None:
    from docket.models.otlp import _encode_value

    assert _encode_value(True) == {"boolValue": True}
    assert _encode_value(1) == {"intValue": "1"}


def test_to_otlp_encodes_nested_kvlist() -> None:
    from docket.models.otlp import _encode_value

    encoded = _encode_value({"a": 1, "b": "x"})
    assert encoded == {
        "kvlistValue": {
            "values": [
                {"key": "a", "value": {"intValue": "1"}},
                {"key": "b", "value": {"stringValue": "x"}},
            ]
        }
    }


def test_decode_value_unknown_returns_none() -> None:
    from docket.models.otlp import _decode_value

    assert _decode_value({"weirdValue": "x"}) is None


def test_from_otlp_rejects_multi_trace_payload() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            _minimal_otlp_span("s1", "trace-a"),
                            _minimal_otlp_span("s2", "trace-b"),
                        ]
                    }
                ]
            }
        ]
    }
    with pytest.raises(BackendError, match="2 distinct traces"):
        from_otlp(payload)


def test_from_otlp_missing_required_field_raises_typed_error() -> None:
    bad = _minimal_otlp_span("s1", "trace-a")
    del bad["startTimeUnixNano"]
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [bad]}]}]}
    with pytest.raises(BackendError, match="malformed required field"):
        from_otlp(payload)


def test_none_attribute_round_trips() -> None:
    span = _minimal_otlp_span("s1", "trace-a")
    span["attributes"] = [{"key": "maybe", "value": {}}]
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    trace = from_otlp(payload)
    assert trace.spans[0].attributes["maybe"] is None
    re_encoded = to_otlp(trace)
    attr = re_encoded["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"][0]
    assert attr == {"key": "maybe", "value": {}}
    # And decoding again still yields None, not the string "None".
    assert from_otlp(re_encoded).spans[0].attributes["maybe"] is None

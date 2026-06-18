"""Navigation helpers and `to_trace_like` bridge — §7 Phase 3 acceptance #2."""

import json
from pathlib import Path

import pytest

from docket.models.otlp import from_otlp


@pytest.fixture
def simple_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "simple_llm_call.json").read_text()))


@pytest.fixture
def tool_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "tool_calling_agent.json").read_text()))


@pytest.fixture
def rag_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "retrieval_augmented.json").read_text()))


@pytest.fixture
def multi_agent_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "multi_agent.json").read_text()))


@pytest.fixture
def embedding_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "embedding_trace.json").read_text()))


@pytest.fixture
def error_trace(traces_dir: Path):  # type: ignore[no-untyped-def]
    return from_otlp(json.loads((traces_dir / "error_trace.json").read_text()))


def test_get_llm_spans_simple(simple_trace) -> None:  # type: ignore[no-untyped-def]
    llm = simple_trace.get_llm_spans()
    assert len(llm) == 1
    assert llm[0].kind == "LLM"


def test_get_llm_spans_multi_agent(multi_agent_trace) -> None:  # type: ignore[no-untyped-def]
    llm = multi_agent_trace.get_llm_spans()
    assert len(llm) == 3
    assert all(s.kind == "LLM" for s in llm)


def test_get_tool_call_spans(tool_trace) -> None:  # type: ignore[no-untyped-def]
    tools = tool_trace.get_tool_call_spans()
    assert len(tools) == 1
    assert tools[0].tool_name == "get_weather"


def test_get_retriever_spans(rag_trace) -> None:  # type: ignore[no-untyped-def]
    retr = rag_trace.get_retriever_spans()
    assert len(retr) == 1
    docs = retr[0].retrieval_documents
    assert [d.id for d in docs] == ["doc-policy-01", "doc-policy-02"]


def test_get_embedding_spans(embedding_trace) -> None:  # type: ignore[no-untyped-def]
    emb = embedding_trace.get_embedding_spans()
    assert len(emb) == 1
    assert emb[0].embeddings[0].text == "What is the return policy?"


def test_get_final_response_returns_last_llm_output(tool_trace) -> None:  # type: ignore[no-untyped-def]
    final = tool_trace.get_final_response()
    assert final == "It's sunny in Paris with a temperature of 21C."


def test_get_final_response_orders_by_end_time(multi_agent_trace) -> None:  # type: ignore[no-untyped-def]
    final = multi_agent_trace.get_final_response()
    assert final is not None
    assert "Day 1" in final


def test_get_final_response_none_when_no_llm_spans(simple_trace) -> None:  # type: ignore[no-untyped-def]
    from docket.models.trace import OpenInferenceTrace

    empty = OpenInferenceTrace(trace_id="t1", spans=[])
    assert empty.get_final_response() is None


def test_to_trace_like_basic(simple_trace) -> None:  # type: ignore[no-untyped-def]
    trace_like = simple_trace.to_trace_like()
    assert trace_like.trace_id == "trace-simple-001"
    assert "What is 2+2?" in trace_like.full_text
    assert "2 plus 2 equals 4." in trace_like.full_text
    assert trace_like.final_response == "2 plus 2 equals 4."
    assert trace_like.metrics["span_count"] == 1.0
    assert trace_like.metrics["total_tokens"] == 21.0
    assert trace_like.metrics["error_count"] == 0.0
    assert trace_like.metrics["latency_ms"] == 1000.0


def test_to_trace_like_includes_tool_calls(tool_trace) -> None:  # type: ignore[no-untyped-def]
    trace_like = tool_trace.to_trace_like()
    assert len(trace_like.tool_calls) == 1
    tc = trace_like.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}
    assert "[tool:get_weather] input: {'city': 'Paris'}" in trace_like.full_text
    assert "[tool:get_weather] output: Sunny, 21C" in trace_like.full_text


def test_to_trace_like_includes_retrieved_documents(rag_trace) -> None:  # type: ignore[no-untyped-def]
    trace_like = rag_trace.to_trace_like()
    assert "Returns accepted within 30 days" in trace_like.full_text
    assert "Original receipt required" in trace_like.full_text


def test_to_trace_like_counts_errors(error_trace) -> None:  # type: ignore[no-untyped-def]
    trace_like = error_trace.to_trace_like()
    assert trace_like.metrics["error_count"] == 2.0


def test_to_trace_like_empty_spans() -> None:
    from docket.models.trace import OpenInferenceTrace

    empty = OpenInferenceTrace(trace_id="t1", spans=[])
    trace_like = empty.to_trace_like()
    assert trace_like.full_text == ""
    assert trace_like.metrics["span_count"] == 0.0
    assert trace_like.metrics["latency_ms"] == 0.0

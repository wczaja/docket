"""Tests for Span typed-attribute accessors and OpenInferenceTrace construction."""

from docket.models.trace import OpenInferenceTrace, Span, _parse_indexed_attrs


def _llm_span(**extra: object) -> Span:
    attrs: dict[str, object] = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4o-mini",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "hi",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "hello",
        "llm.token_count.total": 42,
    }
    attrs.update(extra)
    return Span(
        span_id="s1",
        trace_id="t1",
        name="completion",
        start_time_unix_nano=0,
        end_time_unix_nano=1000,
        attributes=attrs,
    )


def test_span_kind_known() -> None:
    assert _llm_span().kind == "LLM"


def test_span_kind_missing_returns_unknown() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="x",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
    )
    assert span.kind == "UNKNOWN"


def test_span_kind_unrecognized_returns_unknown() -> None:
    span = _llm_span(**{"openinference.span.kind": "MADE_UP"})
    assert span.kind == "UNKNOWN"


def test_llm_input_messages() -> None:
    msgs = _llm_span().llm_input_messages
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_llm_output_messages() -> None:
    msgs = _llm_span().llm_output_messages
    assert len(msgs) == 1
    assert msgs[0].role == "assistant"
    assert msgs[0].content == "hello"


def test_llm_input_messages_ordered_by_index() -> None:
    span = _llm_span(
        **{
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "second",
            "llm.input_messages.2.message.role": "assistant",
            "llm.input_messages.2.message.content": "third",
        }
    )
    msgs = span.llm_input_messages
    assert [m.content for m in msgs] == ["hi", "second", "third"]


def test_llm_model_and_token_count() -> None:
    span = _llm_span()
    assert span.llm_model_name == "gpt-4o-mini"
    assert span.llm_token_count_total == 42


def test_tool_span_accessors() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="get_weather",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": "get_weather",
            "tool.parameters": '{"city": "Paris"}',
            "output.value": "Sunny, 21C",
        },
    )
    assert span.kind == "TOOL"
    assert span.tool_name == "get_weather"
    assert span.tool_parameters == {"city": "Paris"}
    assert span.tool_output == "Sunny, 21C"


def test_tool_parameters_already_dict() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="x",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={"tool.parameters": {"a": 1}},
    )
    assert span.tool_parameters == {"a": 1}


def test_tool_parameters_invalid_json_returns_none() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="x",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={"tool.parameters": "not json"},
    )
    assert span.tool_parameters is None


def test_retrieval_documents() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="search",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={
            "openinference.span.kind": "RETRIEVER",
            "retrieval.documents.0.document.id": "doc1",
            "retrieval.documents.0.document.content": "first doc",
            "retrieval.documents.0.document.score": 0.9,
            "retrieval.documents.1.document.id": "doc2",
            "retrieval.documents.1.document.content": "second doc",
            "retrieval.documents.1.document.score": 0.7,
        },
    )
    docs = span.retrieval_documents
    assert [d.id for d in docs] == ["doc1", "doc2"]
    assert docs[0].content == "first doc"
    assert docs[0].score == 0.9


def test_embeddings() -> None:
    span = Span(
        span_id="s",
        trace_id="t",
        name="embed",
        start_time_unix_nano=0,
        end_time_unix_nano=1,
        attributes={
            "openinference.span.kind": "EMBEDDING",
            "embedding.embeddings.0.embedding.text": "hello",
            "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        },
    )
    emb = span.embeddings
    assert len(emb) == 1
    assert emb[0].text == "hello"
    assert emb[0].vector == [0.1, 0.2, 0.3]


def test_parse_indexed_attrs_groups_by_index() -> None:
    result = _parse_indexed_attrs(
        {
            "x.0.a": 1,
            "x.0.b": 2,
            "x.1.a": 3,
            "y.0.c": 4,
        },
        "x",
    )
    assert result == [{"a": 1, "b": 2}, {"a": 3}]


def test_parse_indexed_attrs_skips_malformed() -> None:
    result = _parse_indexed_attrs(
        {
            "x.0.a": 1,
            "x.foo.b": 2,
            "x.0": 3,
        },
        "x",
    )
    assert result == [{"a": 1}]


def test_open_inference_trace_empty_spans() -> None:
    trace = OpenInferenceTrace(trace_id="t1")
    assert trace.spans == []

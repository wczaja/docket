"""Trace and verdict types.

Phase 3 ships the canonical OpenInferenceTrace + Span here. `TraceLike` (Phase 2)
stays as the lightweight view detectors operate on; `OpenInferenceTrace.to_trace_like()`
bridges the two.

Attribute namespaces modeled:
    llm.*         (input/output messages, token counts, model name)
    tool.*        (name, parameters, description, output)
    retrieval.*   (documents with id/content/metadata/score)
    embedding.*   (embeddings with text/vector)

Span attribute access goes through typed @property views that parse the
indexed-key encoding used by OpenInference (e.g. `llm.input_messages.0.message.role`).
"""

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

SpanKind = Literal[
    "AGENT",
    "CHAIN",
    "LLM",
    "RETRIEVER",
    "EMBEDDING",
    "TOOL",
    "RERANKER",
    "EVALUATOR",
    "UNKNOWN",
]

_OPENINFERENCE_KINDS: frozenset[str] = frozenset(
    {
        "AGENT",
        "CHAIN",
        "LLM",
        "RETRIEVER",
        "EMBEDDING",
        "TOOL",
        "RERANKER",
        "EVALUATOR",
        "UNKNOWN",
    }
)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: str
    content: str | None = None


class Document(BaseModel):
    id: str | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None
    score: float | None = None


class Embedding(BaseModel):
    text: str | None = None
    vector: list[float] | None = None


class Status(BaseModel):
    code: Literal["OK", "ERROR", "UNSET"] = "UNSET"
    message: str | None = None


class Event(BaseModel):
    name: str
    time_unix_nano: int
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    status: Status = Field(default_factory=Status)

    @property
    def kind(self) -> SpanKind:
        v = self.attributes.get("openinference.span.kind")
        if isinstance(v, str) and v in _OPENINFERENCE_KINDS:
            return cast(SpanKind, v)
        return "UNKNOWN"

    @property
    def llm_model_name(self) -> str | None:
        v = self.attributes.get("llm.model_name")
        return v if isinstance(v, str) else None

    @property
    def llm_input_messages(self) -> list[Message]:
        return [
            Message(
                role=str(m.get("message.role", "")),
                content=_as_optional_str(m.get("message.content")),
            )
            for m in _parse_indexed_attrs(self.attributes, "llm.input_messages")
        ]

    @property
    def llm_output_messages(self) -> list[Message]:
        return [
            Message(
                role=str(m.get("message.role", "")),
                content=_as_optional_str(m.get("message.content")),
            )
            for m in _parse_indexed_attrs(self.attributes, "llm.output_messages")
        ]

    @property
    def llm_token_count_total(self) -> int | None:
        v = self.attributes.get("llm.token_count.total")
        return int(v) if isinstance(v, (int, float)) else None

    @property
    def tool_name(self) -> str | None:
        v = self.attributes.get("tool.name")
        return v if isinstance(v, str) else None

    @property
    def tool_parameters(self) -> dict[str, Any] | None:
        v = self.attributes.get("tool.parameters")
        if isinstance(v, dict):
            return cast(dict[str, Any], v)
        if isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return None
            return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
        return None

    @property
    def tool_output(self) -> str | None:
        v = self.attributes.get("output.value")
        return v if isinstance(v, str) else None

    @property
    def retrieval_documents(self) -> list[Document]:
        return [
            Document(
                id=_as_optional_str(d.get("document.id")),
                content=_as_optional_str(d.get("document.content")),
                metadata=d.get("document.metadata")
                if isinstance(d.get("document.metadata"), dict)
                else None,
                score=float(d["document.score"])
                if isinstance(d.get("document.score"), (int, float))
                else None,
            )
            for d in _parse_indexed_attrs(self.attributes, "retrieval.documents")
        ]

    @property
    def embeddings(self) -> list[Embedding]:
        return [
            Embedding(
                text=_as_optional_str(e.get("embedding.text")),
                vector=cast(list[float], e["embedding.vector"])
                if isinstance(e.get("embedding.vector"), list)
                else None,
            )
            for e in _parse_indexed_attrs(self.attributes, "embedding.embeddings")
        ]


class OpenInferenceTrace(BaseModel):
    trace_id: str
    spans: list[Span] = Field(default_factory=list)

    def get_llm_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "LLM"]

    def get_tool_call_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "TOOL"]

    def get_retriever_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "RETRIEVER"]

    def get_embedding_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "EMBEDDING"]

    def get_final_response(self) -> str | None:
        """Last assistant-shaped message from the chronologically-last LLM span."""
        llm_spans = self.get_llm_spans()
        if not llm_spans:
            return None
        last = max(llm_spans, key=lambda s: s.end_time_unix_nano)
        outputs = last.llm_output_messages
        if not outputs:
            return None
        return outputs[-1].content

    def to_trace_like(self) -> "TraceLike":
        """Project this trace into the lightweight `TraceLike` view detectors consume.

        Composes `full_text` by walking spans in order and emitting one line per
        LLM message / tool i/o / retrieved document. Tool calls and metrics are
        extracted from their respective span kinds.

        PII redaction is intentionally NOT applied here. Redaction sits at the
        `LLMJudgeDetector` boundary so detectors that don't exfiltrate (regex,
        tool_call, metric_threshold) still see the full trace.
        """
        parts: list[str] = []
        for span in self.spans:
            if span.kind == "LLM":
                for msg in span.llm_input_messages:
                    if msg.content:
                        parts.append(f"[{msg.role}] {msg.content}")
                for msg in span.llm_output_messages:
                    if msg.content:
                        parts.append(f"[{msg.role}] {msg.content}")
            elif span.kind == "TOOL":
                name = span.tool_name or span.name
                if span.tool_parameters is not None:
                    parts.append(f"[tool:{name}] input: {span.tool_parameters}")
                if span.tool_output:
                    parts.append(f"[tool:{name}] output: {span.tool_output}")
            elif span.kind == "RETRIEVER":
                for doc in span.retrieval_documents:
                    if doc.content:
                        parts.append(f"[retrieved:{doc.id or '?'}] {doc.content}")
        full_text = "\n".join(parts)

        tool_calls = [
            ToolCall(name=s.tool_name or s.name, arguments=s.tool_parameters or {})
            for s in self.spans
            if s.kind == "TOOL"
        ]

        total_tokens = sum((s.llm_token_count_total or 0) for s in self.spans if s.kind == "LLM")
        error_count = sum(1 for s in self.spans if s.status.code == "ERROR")
        if self.spans:
            start = min(s.start_time_unix_nano for s in self.spans)
            end = max(s.end_time_unix_nano for s in self.spans)
            latency_ms = (end - start) / 1_000_000
        else:
            latency_ms = 0.0
        metrics: dict[str, float] = {
            "span_count": float(len(self.spans)),
            "total_tokens": float(total_tokens),
            "error_count": float(error_count),
            "latency_ms": latency_ms,
        }

        return TraceLike(
            full_text=full_text,
            final_response=self.get_final_response() or "",
            tool_calls=tool_calls,
            metrics=metrics,
            trace_id=self.trace_id,
        )


class TraceLike(BaseModel):
    """Lightweight view detectors operate on. Built from a real trace via
    `OpenInferenceTrace.to_trace_like()` or constructed inline in tests."""

    full_text: str
    final_response: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    context: str | None = None
    trace_id: str | None = None


class Verdict(BaseModel):
    positive: bool
    extra: dict[str, Any] = Field(default_factory=dict)


def _as_optional_str(v: Any) -> str | None:
    return v if isinstance(v, str) else None


def _parse_indexed_attrs(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Group `prefix.{n}.rest` attribute keys into a list of per-index dicts.

    Example::

        {"llm.input_messages.0.message.role": "user",
         "llm.input_messages.0.message.content": "hi",
         "llm.input_messages.1.message.role": "assistant"}

    parsed with prefix="llm.input_messages" returns::

        [{"message.role": "user", "message.content": "hi"},
         {"message.role": "assistant"}]
    """
    by_index: dict[int, dict[str, Any]] = {}
    full_prefix = prefix + "."
    for key, val in attrs.items():
        if not key.startswith(full_prefix):
            continue
        rest = key[len(full_prefix) :]
        idx_str, _, sub_key = rest.partition(".")
        if not idx_str.isdigit() or not sub_key:
            continue
        by_index.setdefault(int(idx_str), {})[sub_key] = val
    return [by_index[i] for i in sorted(by_index)]

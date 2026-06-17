"""Unit tests for the Deep Agents wrapper.

We test:
  - `_build_tools` constructs the six tools, invokable through langchain
    BaseTool.ainvoke({...}) and mutating the shared `_AgentState` correctly.
  - The vfs writes (`Command(update={...})`) carry the expected file paths.
  - `extract_report_markdown` handles deepagents' FileData shape.
  - `build_triage_agent` calls `create_deep_agent` with our tools + prompt.

We do NOT test the agent's LLM-driven planning loop -- that's the gated
integration test in C2. Here we drive the tools directly to verify
behavior + state mutations without paying for LLM calls.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from agent_triage.adapters.base import TraceBackend
from agent_triage.agent.deep_agent import (
    DEFAULT_AGENT_MODEL,
    _ack,
    _AgentState,
    _build_tools,
    build_triage_agent,
    extract_report_markdown,
)
from agent_triage.llm.base import ModelProvider
from agent_triage.llm.embeddings import EmbeddingProvider
from agent_triage.models.classification import Annotation
from agent_triage.models.trace import OpenInferenceTrace, Span
from agent_triage.rubric.spec import Clustering, Detection, Mode, Rubric, RubricMetadata


class _FakeBackend(TraceBackend):
    def __init__(self, traces: dict[str, OpenInferenceTrace]) -> None:
        self.traces = traces
        self.annotations: list[Annotation] = []

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        return list(self.traces.keys())

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        return self.traces[trace_id]

    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        self.annotations.append(annotation)

    async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
        pass

    async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
        return set()


class _MockLLMProvider(ModelProvider):
    def __init__(self) -> None:
        self.model = "mock-llm"

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        return {"title": "Drafted issue", "body": "Body" * 20}


class _MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = "mock-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _trace(trace_id: str, text: str) -> OpenInferenceTrace:
    return OpenInferenceTrace(
        trace_id=trace_id,
        spans=[
            Span(
                span_id="s",
                trace_id=trace_id,
                name="x",
                start_time_unix_nano=0,
                end_time_unix_nano=1_000_000,
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": text,
                },
            ),
        ],
    )


def _rubric() -> Rubric:
    return Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="testbench", version="0.1.0"),
        modes=[
            Mode(
                id="says-hi",
                severity="medium",
                detection=Detection(type="regex", pattern="hello"),
            ),
        ],
        clustering=Clustering(
            embedding_model="mock-embed",
            similarity_threshold=0.82,
            min_cluster_size=3,
        ),
    )


def _state(
    *,
    backend: TraceBackend,
    rubric: Rubric,
    output_dir: Path,
    write_annotations: bool = False,
) -> _AgentState:
    return _AgentState(
        backend=backend,
        rubric=rubric,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        run_id="run-test",
        output_dir=output_dir,
        write_annotations=write_annotations,
        since=datetime(2026, 5, 22, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        started_at=datetime(2026, 5, 22, tzinfo=UTC),
    )


async def _invoke(tool: Any, **kwargs: object) -> Any:
    """Invoke a langchain @tool function with the InjectedToolCallId arg."""
    return await tool.ainvoke({"tool_call_id": "tcid-1", **kwargs})


async def test_list_traces_tool_populates_state(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(4)})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    list_traces_tool = next(t for t in tools if t.name == "list_traces")
    result = await _invoke(list_traces_tool)
    # Tool returned a Command that wrote the manifest file.
    assert state.trace_ids == [f"t-{i}" for i in range(4)]
    files = result.update["files"]
    assert "/traces/manifest.json" in files


async def test_classify_traces_tool_fills_classifications(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello world") for i in range(3)})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    list_tool = next(t for t in tools if t.name == "list_traces")
    classify_tool = next(t for t in tools if t.name == "classify_traces")
    await _invoke(list_tool)
    result = await _invoke(classify_tool)
    assert len(state.classifications) == 3
    assert all(c.positive for c in state.classifications)
    assert "/classifications/summary.json" in result.update["files"]


async def test_classify_traces_short_circuits_without_traces(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    classify_tool = next(t for t in tools if t.name == "classify_traces")
    result = await _invoke(classify_tool)
    # Should produce a "no traces" message and not crash.
    msg = result.update["messages"][0]
    assert "No traces" in msg.content


async def test_annotate_tool_skips_when_disabled(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hello")})
    state = _state(
        backend=backend,
        rubric=_rubric(),
        output_dir=tmp_path,
        write_annotations=False,
    )
    tools = _build_tools(state)
    annotate_tool = next(t for t in tools if t.name == "annotate_classifications")
    result = await _invoke(annotate_tool)
    assert backend.annotations == []
    assert "disabled" in result.update["messages"][0].content.lower()


async def test_annotate_tool_writes_when_enabled(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hello")})
    state = _state(
        backend=backend,
        rubric=_rubric(),
        output_dir=tmp_path,
        write_annotations=True,
    )
    tools = _build_tools(state)
    list_tool = next(t for t in tools if t.name == "list_traces")
    classify_tool = next(t for t in tools if t.name == "classify_traces")
    annotate_tool = next(t for t in tools if t.name == "annotate_classifications")
    await _invoke(list_tool)
    await _invoke(classify_tool)
    await _invoke(annotate_tool)
    assert len(backend.annotations) == 1
    assert backend.annotations[0].mode_id == "says-hi"
    assert state.annotations_written == 1


async def test_full_tool_chain_produces_report(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello world") for i in range(5)})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    by_name = {t.name: t for t in tools}
    await _invoke(by_name["list_traces"])
    await _invoke(by_name["classify_traces"])
    await _invoke(by_name["cluster_classifications"])
    await _invoke(by_name["draft_issues_tool"])
    write_result = await _invoke(by_name["write_report"])
    files = write_result.update["files"]
    assert "/report.md" in files
    report_md = files["/report.md"]["content"]
    assert "# agent-triage run" in report_md
    assert "## Clusters" in report_md
    assert state.clusters
    assert state.drafts


async def test_cluster_tool_short_circuits_without_classifications(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    cluster_tool = next(t for t in tools if t.name == "cluster_classifications")
    result = await _invoke(cluster_tool)
    msg = result.update["messages"][0].content
    assert "No classifications" in msg


async def test_draft_tool_short_circuits_without_clusters(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    state = _state(backend=backend, rubric=_rubric(), output_dir=tmp_path)
    tools = _build_tools(state)
    draft_tool = next(t for t in tools if t.name == "draft_issues_tool")
    result = await _invoke(draft_tool)
    msg = result.update["messages"][0].content
    assert "No clusters" in msg


def test_extract_report_markdown_from_filedata_shape() -> None:
    final = {"files": {"/report.md": {"content": "# hello", "revision": 0}}}
    assert extract_report_markdown(final) == "# hello"


def test_extract_report_markdown_from_plain_string() -> None:
    final = {"files": {"/report.md": "# hello"}}
    assert extract_report_markdown(final) == "# hello"


def test_extract_report_markdown_missing_returns_empty() -> None:
    assert extract_report_markdown({}) == ""
    assert extract_report_markdown({"files": {}}) == ""


def test_ack_builds_command_with_files() -> None:
    cmd = _ack("tcid-1", content="ok", files={"/a.json": "{}"})
    update = cmd.update
    assert "files" in update
    assert update["files"]["/a.json"]["content"] == "{}"
    assert update["messages"][0].content == "ok"


def test_ack_without_files_omits_files_key() -> None:
    cmd = _ack("tcid-1", content="just a message")
    update = cmd.update
    assert "files" not in update
    assert update["messages"][0].content == "just a message"


def test_build_triage_agent_calls_create_deep_agent(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    with patch("deepagents.create_deep_agent") as fake_create:
        fake_create.return_value = MagicMock(name="CompiledStateGraph")
        agent, state = build_triage_agent(
            backend=backend,
            rubric=_rubric(),
            llm_provider=_MockLLMProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            since=datetime(2026, 5, 22, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, tzinfo=UTC),
            output_dir=tmp_path,
        )
    fake_create.assert_called_once()
    _args, kwargs = fake_create.call_args
    assert kwargs["model"] == DEFAULT_AGENT_MODEL
    # Six tools wired up.
    assert len(kwargs["tools"]) == 6
    tool_names = {t.name for t in kwargs["tools"]}
    assert tool_names == {
        "list_traces",
        "classify_traces",
        "annotate_classifications",
        "cluster_classifications",
        "draft_issues_tool",
        "write_report",
    }
    assert state.run_id  # deterministic value computed
    assert len(state.run_id) == 16

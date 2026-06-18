"""Deep Agents harness wrapping the Phase 5 pipeline (design §4.2).

The Phase 5 deterministic pipeline (`agent.triage.run_triage_pipeline`) is the
source of truth and the default `docket run` path. This module exposes
the same workflow as a deepagents-driven agent: each pipeline stage becomes a
LangChain tool, a top-level LLM plans the tool calls, and the final
`report.md` lands in the deepagents virtual filesystem at `/report.md`.

CLI: `docket run --agent` opts in.

Pipeline stages exposed as tools:
  - list_traces                  fetch trace IDs in the window
  - classify_traces              run every mode against every trace (concurrent + retry)
  - annotate_classifications     write positives back to the backend (gated on --annotate)
  - cluster_classifications      HDBSCAN per mode
  - draft_issues                 LLM-driven draft per cluster, queued to disk
  - write_report                 render report.md and store at /report.md

State held in two places:
  - A Python `_AgentState` closure object holds the actual pipeline data
    (traces, classifications, clusters, drafts). Tools share it.
  - The deepagents virtual filesystem holds summaries and the final
    `/report.md`. The agent reasons over the vfs; we extract `/report.md`
    after the run completes.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from docket.adapters.base import TraceBackend
from docket.agent.report import render_report
from docket.agent.subagents.annotator import Annotator
from docket.agent.subagents.classifier import Classifier
from docket.agent.subagents.clusterer import cluster_per_mode
from docket.agent.subagents.drafter import DEFAULT_QUEUE_DIR, draft_issues
from docket.agent.triage import compute_run_id
from docket.llm.base import ModelProvider
from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Classification
from docket.models.cluster import Cluster
from docket.models.issue import IssueDraft
from docket.models.report import FetchFailure, ModeStats, RunReport, TraceResult
from docket.models.trace import OpenInferenceTrace
from docket.observability import redact
from docket.rubric.spec import Rubric

log = logging.getLogger(__name__)

DEFAULT_AGENT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are docket, a runtime for triaging LLM agent traces against a
failure-mode rubric. You have tools for each stage of the workflow. Your job
is to run them in the right order and produce a final report.

Workflow:
  1. Call `list_traces` to get the trace IDs in the time window.
  2. Call `classify_traces` to run every mode's detector against every trace.
  3. Call `annotate_classifications` if annotations are enabled (the tool
     will tell you when they're not, in which case skip it).
  4. Call `cluster_classifications` to group similar positive classifications.
  5. Call `draft_issues` to generate one issue draft per cluster.
  6. Call `write_report` to assemble the markdown summary at /report.md.

Each tool returns a status message describing what it did. Read the messages;
adjust subsequent calls if a stage returned 0 results (e.g., skip clustering
if there are no positives). Do not invent steps the workflow doesn't include.
"""


@dataclass
class _AgentState:
    """Per-run pipeline state shared across tool calls.

    Held outside the langgraph state because the actual Pydantic models
    (OpenInferenceTrace, Classification, Cluster, IssueDraft) aren't worth
    re-serializing through the agent every call.
    """

    backend: TraceBackend
    rubric: Rubric
    llm_provider: ModelProvider
    embedding_provider: EmbeddingProvider
    run_id: str
    output_dir: Path
    write_annotations: bool
    batch_size: int = 1
    concurrency: int = 8

    since: datetime | None = None
    until: datetime | None = None
    started_at: datetime | None = None
    trace_ids: list[str] = field(default_factory=list)
    traces: dict[str, OpenInferenceTrace] = field(default_factory=dict)
    # (trace_id, redacted reason) for traces whose fetch failed; mirrored
    # into RunReport.fetch_failures (design §4.4: log, skip, report).
    fetch_failures: list[tuple[str, str]] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    drafts: list[IssueDraft] = field(default_factory=list)
    annotations_written: int = 0


def build_triage_agent(
    *,
    backend: TraceBackend,
    rubric: Rubric,
    llm_provider: ModelProvider,
    embedding_provider: EmbeddingProvider,
    since: datetime,
    until: datetime,
    run_id: str | None = None,
    output_dir: Path | None = None,
    write_annotations: bool = False,
    batch_size: int = 1,
    concurrency: int = 8,
    agent_model: str = DEFAULT_AGENT_MODEL,
    backend_id: str = "phoenix",
) -> tuple[CompiledStateGraph, _AgentState]:  # type: ignore[type-arg]
    """Construct a deepagents CompiledStateGraph that runs the Phase 5 pipeline.

    Returns the agent and the closure state. Caller invokes the agent with
    `.ainvoke({"messages": [...]})` and then reads `state.run_id`, etc.

    `backend_id` feeds the deterministic run_id (design §4.2); pass the
    resolved backend label so deep-agent runs upsert against the same
    annotations as deterministic-pipeline runs. Defaults to "phoenix" for
    back-compat.
    """
    rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
    final_run_id = run_id or compute_run_id(
        backend_id=backend_id,
        rubric_version=rubric_version,
        since=since,
        until=until,
    )
    agent_state = _AgentState(
        backend=backend,
        rubric=rubric,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        run_id=final_run_id,
        output_dir=output_dir if output_dir is not None else DEFAULT_QUEUE_DIR,
        write_annotations=write_annotations,
        batch_size=batch_size,
        concurrency=concurrency,
        since=since,
        until=until,
        started_at=datetime.now(tz=since.tzinfo),
    )
    tools = _build_tools(agent_state)

    # Imported here so unit tests that patch `create_deep_agent` get the
    # patched version.
    from deepagents import create_deep_agent  # noqa: PLC0415

    agent = create_deep_agent(
        model=agent_model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
    )
    return agent, agent_state


def _build_tools(state: _AgentState) -> list[BaseTool]:
    """Build the six pipeline tools as closures over the agent state.

    Each tool returns a Command that updates the deepagents vfs with a
    summary file. The actual pipeline data lives in `state`; the vfs holds
    the agent's read-only view + the final report.md.
    """

    @tool
    async def list_traces(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """List traces in the configured time window. Stores them in /traces/manifest.json."""
        assert state.since is not None
        assert state.until is not None
        ids = await state.backend.list_traces(state.since, state.until)
        state.trace_ids = list(ids)
        manifest = json.dumps({"trace_count": len(ids), "trace_ids": ids[:50]}, indent=2)
        return _ack(
            tool_call_id,
            content=(
                f"Listed {len(ids)} traces in "
                f"[{state.since.isoformat()}, {state.until.isoformat()}]."
            ),
            files={"/traces/manifest.json": manifest},
        )

    @tool
    async def classify_traces(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """Run every mode against every listed trace. Stores summary in vfs."""
        if not state.trace_ids:
            return _ack(
                tool_call_id,
                content="No traces to classify. Call list_traces first or skip ahead.",
            )
        # Fetch the traces. Per-trace fetch failures are contained (design
        # §4.4: log, skip, report) instead of aborting the whole run --
        # mirroring the deterministic pipeline.
        for tid in state.trace_ids:
            if tid in state.traces:
                continue
            try:
                state.traces[tid] = await state.backend.get_trace(tid)
            except Exception as e:  # noqa: BLE001 -- adapter errors vary by backend
                reason = redact(str(e))
                state.fetch_failures.append((tid, reason))
                log.warning("skipping trace %s after fetch failure: %s", tid, reason)
        if state.fetch_failures:
            log.warning(
                "fetch phase completed with %d/%d traces skipped due to backend errors",
                len(state.fetch_failures),
                len(state.trace_ids),
            )
        classifier = Classifier(
            state.llm_provider,
            batch_size=state.batch_size,
            concurrency=state.concurrency,
        )
        fetched_ids = [tid for tid in state.trace_ids if tid in state.traces]
        results = await classifier.classify_all(
            [(tid, state.traces[tid]) for tid in fetched_ids],
            state.rubric,
        )
        flat: list[Classification] = []
        for items in results.values():
            flat.extend(items)
        state.classifications = flat
        positive = sum(1 for c in flat if c.positive and c.error is None)
        errors = sum(1 for c in flat if c.error is not None)
        per_mode: dict[str, int] = {}
        for c in flat:
            if c.positive and c.error is None:
                per_mode[c.mode_id] = per_mode.get(c.mode_id, 0) + 1
        summary = json.dumps(
            {
                "classifications": len(flat),
                "positive": positive,
                "errors": errors,
                "fetch_failures": len(state.fetch_failures),
                "per_mode_positives": per_mode,
            },
            indent=2,
            sort_keys=True,
        )
        skipped_note = (
            f" Skipped {len(state.fetch_failures)} trace(s) after fetch failures."
            if state.fetch_failures
            else ""
        )
        return _ack(
            tool_call_id,
            content=(
                f"Classified {len(fetched_ids)} traces: "
                f"{positive} positives ({errors} errors).{skipped_note} Summary at "
                f"/classifications/summary.json."
            ),
            files={"/classifications/summary.json": summary},
        )

    @tool
    async def annotate_classifications(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """Write positive classifications back to the backend (idempotent on run_id+mode+trace)."""
        if not state.write_annotations:
            return _ack(
                tool_call_id,
                content="Annotation writeback is disabled for this run (--no-annotate). Skip.",
            )
        if not state.classifications:
            return _ack(
                tool_call_id,
                content="No classifications to annotate. Call classify_traces first.",
            )
        annotator = Annotator(state.backend, run_id=state.run_id)
        written = await annotator.annotate_positive(state.classifications, state.rubric)
        state.annotations_written = written
        return _ack(
            tool_call_id,
            content=f"Wrote {written} annotations to the backend (idempotency_key per design §5).",
            files={
                "/annotations/summary.json": json.dumps({"written": written}, indent=2),
            },
        )

    @tool
    async def cluster_classifications(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """Cluster positive classifications per mode via HDBSCAN."""
        if not state.classifications:
            return _ack(
                tool_call_id,
                content="No classifications to cluster. Call classify_traces first.",
            )
        clusters = await cluster_per_mode(
            state.classifications,
            rubric=state.rubric,
            embedding_provider=state.embedding_provider,
            traces_by_id=state.traces,
        )
        state.clusters = clusters
        per_mode = {c.mode_id: c.stats.size for c in clusters}
        summary = json.dumps({"cluster_count": len(clusters), "per_mode_sizes": per_mode}, indent=2)
        return _ack(
            tool_call_id,
            content=(
                f"Formed {len(clusters)} clusters across "
                f"{len(per_mode)} mode(s). Summary at /clusters/summary.json."
            ),
            files={"/clusters/summary.json": summary},
        )

    @tool
    async def draft_issues_tool(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """Draft one issue per cluster via LLM. Files land in the queue dir + /drafts/."""
        if not state.clusters:
            return _ack(
                tool_call_id,
                content="No clusters to draft from. Call cluster_classifications first.",
            )
        drafts = await draft_issues(
            state.clusters,
            rubric=state.rubric,
            llm_provider=state.llm_provider,
            run_id=state.run_id,
            output_dir=state.output_dir,
        )
        state.drafts = drafts
        titles = json.dumps({d.cluster_id: d.title for d in drafts}, indent=2, sort_keys=True)
        return _ack(
            tool_call_id,
            content=(
                f"Drafted {len(drafts)} issue(s). Queue files in {state.output_dir}. "
                f"Titles at /drafts/titles.json."
            ),
            files={"/drafts/titles.json": titles},
        )

    @tool
    async def write_report(
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command[None]:
        """Assemble the markdown report. Stored at /report.md (design §4.2)."""
        run_report = _materialize_run_report(state)
        markdown = render_report(
            run_report=run_report,
            clusters=state.clusters,
            drafts=state.drafts,
        )
        return _ack(
            tool_call_id,
            content=(
                f"Report written to /report.md "
                f"({len(state.clusters)} clusters, {len(state.drafts)} drafts)."
            ),
            files={"/report.md": markdown},
        )

    return [
        list_traces,
        classify_traces,
        annotate_classifications,
        cluster_classifications,
        draft_issues_tool,
        write_report,
    ]


def _ack(
    tool_call_id: str,
    *,
    content: str,
    files: dict[str, str] | None = None,
) -> Command[None]:
    """Build a Command that writes vfs files and replies to the agent.

    Returning a ToolMessage as a state update lets the agent see our reply
    naturally; the file dict merges via deepagents' DeltaChannel into the
    existing files map.
    """
    from langchain_core.messages import ToolMessage  # noqa: PLC0415

    update: dict[str, Any] = {
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
    }
    if files:
        update["files"] = {path: _as_file_data(text) for path, text in files.items()}
    return Command(update=update)


def _as_file_data(text: str) -> Any:
    """Deepagents 0.6.3 expects each vfs entry to be a FileData mapping with
    `content` + `revision` keys. The DeltaChannel merges incoming dicts on top
    of the existing map, so we hand it that shape directly."""
    return {"content": text, "revision": 0}


def _materialize_run_report(state: _AgentState) -> RunReport:
    """Build the same RunReport the deterministic pipeline produces, from the
    closure state. Lets the report renderer stay identical between paths."""
    failed_ids = {tid for tid, _ in state.fetch_failures}
    fetched_ids = [tid for tid in state.trace_ids if tid not in failed_ids]
    classifications_by_trace: dict[str, list[Classification]] = {tid: [] for tid in fetched_ids}
    for c in state.classifications:
        classifications_by_trace.setdefault(c.trace_id, []).append(c)
    by_mode: dict[str, ModeStats] = {
        mode.id: ModeStats(mode_id=mode.id, severity=mode.severity) for mode in state.rubric.modes
    }
    for items in classifications_by_trace.values():
        for c in items:
            stats = by_mode.get(c.mode_id)
            if stats is None:
                continue
            if c.error is not None:
                stats.error_count += 1
            elif c.positive:
                stats.positive_count += 1
            else:
                stats.negative_count += 1
    return RunReport(
        run_id=state.run_id,
        rubric_name=state.rubric.metadata.name,
        rubric_version=state.rubric.metadata.version,
        since=state.since or datetime.now(tz=None),
        until=state.until or datetime.now(tz=None),
        started_at=state.started_at or datetime.now(tz=None),
        finished_at=datetime.now(tz=state.since.tzinfo) if state.since else datetime.now(tz=None),
        trace_count=len(state.trace_ids),
        trace_results=[
            TraceResult(trace_id=tid, classifications=classifications_by_trace.get(tid, []))
            for tid in fetched_ids
        ],
        mode_stats=[by_mode[m.id] for m in state.rubric.modes],
        annotations_written=state.annotations_written,
        fetch_failures=[
            FetchFailure(trace_id=tid, reason=reason) for tid, reason in state.fetch_failures
        ],
    )


def extract_report_markdown(final_state: dict[str, Any]) -> str:
    """Pull the /report.md the write_report tool stashed into the vfs.

    deepagents 0.6.3 stores files as FileData dicts under state["files"]; we
    handle both that shape and the older plain-string shape defensively.
    """
    files = final_state.get("files") or {}
    entry = files.get("/report.md")
    if entry is None:
        return ""
    if isinstance(entry, dict):
        content = entry.get("content", "")
        return str(content) if content is not None else ""
    return str(entry)

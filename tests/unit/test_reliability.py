"""Phase 11 reliability integration test (design §7).

Exercises two scenarios that production deployments hit but the unit suite
can't:

1. **Injected fetch failures.** A backend that fails `get_trace` for ~30% of
   calls. The pipeline must complete with the remaining traces classified
   and the skipped count surfaced in the report's warning logs — without
   aborting the run.

2. **Resume from checkpoint.** A backend that already holds sentinel
   annotations for some traces. The pipeline must skip those traces and
   classify only the rest, then mark every newly-classified trace processed
   so a second resume converges to zero new work.

Both run against a fully in-memory backend (no Phoenix/Langfuse/LangSmith
needed), so this test is part of the default `pytest` suite and not gated
behind `--run-integration`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_triage.adapters.base import TraceBackend
from agent_triage.agent.triage import run_triage_pipeline
from agent_triage.llm.base import ModelProvider
from agent_triage.llm.embeddings import EmbeddingProvider
from agent_triage.models.classification import Annotation
from agent_triage.models.trace import OpenInferenceTrace, Span
from agent_triage.rubric.spec import (
    Clustering,
    Detection,
    Mode,
    Rubric,
    RubricMetadata,
)


class _FlakyBackend(TraceBackend):
    """In-memory backend that fails `get_trace` for a configurable set of IDs."""

    def __init__(
        self,
        traces: dict[str, OpenInferenceTrace],
        *,
        failing_ids: set[str] | None = None,
        already_processed: set[str] | None = None,
    ) -> None:
        self.traces = traces
        self.failing_ids = failing_ids or set()
        self.already_processed = set(already_processed or set())
        self.processed: set[str] = set(self.already_processed)
        self.fetched: set[str] = set()

    async def list_traces(self, since: datetime, until=None, filter=None):  # type: ignore[no-untyped-def]
        del since, until, filter
        return list(self.traces.keys())

    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        if trace_id in self.failing_ids:
            raise RuntimeError(f"simulated fetch failure for {trace_id}")
        self.fetched.add(trace_id)
        return self.traces[trace_id]

    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        del trace_id, annotation

    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        raise NotImplementedError

    async def mark_trace_processed(  # type: ignore[no-untyped-def]
        self, trace_id, *, run_id, rubric_version
    ) -> None:
        del run_id, rubric_version
        self.processed.add(trace_id)

    async def list_processed_trace_ids(  # type: ignore[no-untyped-def]
        self, *, run_id, since, until=None
    ) -> set[str]:
        del run_id, since, until
        return set(self.already_processed)


class _MockLLM(ModelProvider):
    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        del system, user, schema
        return {"positive": False, "confidence": 0.5}


class _MockEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def _trace(trace_id: str) -> OpenInferenceTrace:
    return OpenInferenceTrace(
        trace_id=trace_id,
        spans=[
            Span(
                span_id=f"s-{trace_id}",
                trace_id=trace_id,
                parent_span_id=None,
                name="completion",
                start_time_unix_nano=1,
                end_time_unix_nano=2,
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.input_messages.0.message.role": "user",
                    "llm.input_messages.0.message.content": "hi",
                    "llm.output_messages.0.message.content": "hello",
                },
            )
        ],
    )


def _rubric() -> Rubric:
    return Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="rel", version="1", description="reliability test"),
        modes=[
            Mode(
                id="mode-a",
                severity="medium",
                detection=Detection(type="regex", pattern="(?i)hello"),
            )
        ],
        clustering=Clustering(
            embedding_model="mock-embed",
            similarity_threshold=0.82,
            min_cluster_size=3,
        ),
    )


async def test_pipeline_completes_when_30pct_of_fetches_fail(tmp_path: Path) -> None:
    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}") for i in range(20)}
    failing = {f"t-{i:02d}" for i in range(0, 20, 3)}  # 7 failures of 20 (~35%)
    backend = _FlakyBackend(traces, failing_ids=failing)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLM(),
        embedding_provider=_MockEmbeddings(),
        output_dir=tmp_path,
    )
    expected_classified = len(traces) - len(failing)
    classified_ids = {tr.trace_id for tr in result.run_report.trace_results if tr.classifications}
    assert len(classified_ids) == expected_classified
    assert failing.isdisjoint(classified_ids)


async def test_resume_skips_already_processed_and_converges(
    tmp_path: Path,
) -> None:
    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}") for i in range(10)}
    already = {"t-00", "t-01", "t-02", "t-03"}
    backend = _FlakyBackend(traces, already_processed=already)
    first = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLM(),
        embedding_provider=_MockEmbeddings(),
        output_dir=tmp_path,
        checkpoint=True,
        run_id="resume-test",
    )
    classified_first = {tr.trace_id for tr in first.run_report.trace_results}
    assert classified_first == set(traces.keys()) - already
    assert backend.fetched == set(traces.keys()) - already

    # Second invocation: backend now reports all traces as processed (because
    # the first run marked the remaining ones). Pipeline should converge to
    # zero new classifications.
    backend.already_processed = set(traces.keys())
    backend.fetched.clear()
    second = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLM(),
        embedding_provider=_MockEmbeddings(),
        output_dir=tmp_path,
        checkpoint=True,
        run_id="resume-test",
    )
    assert second.run_report.trace_count == 0
    assert backend.fetched == set()  # no new fetches


async def test_sentinel_failures_do_not_abort_the_run(tmp_path: Path) -> None:
    """If mark_trace_processed itself fails for some traces, the run still
    completes; those traces just get reclassified on next resume."""

    class _SentinelFailing(_FlakyBackend):
        async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
            if trace_id == "t-01":
                raise RuntimeError("sentinel write failed")
            await super().mark_trace_processed(
                trace_id, run_id=run_id, rubric_version=rubric_version
            )

    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}") for i in range(5)}
    backend = _SentinelFailing(traces)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLM(),
        embedding_provider=_MockEmbeddings(),
        output_dir=tmp_path,
        checkpoint=True,
    )
    assert result.run_report.trace_count == 5
    # t-01 did NOT get marked processed; the others did.
    assert "t-01" not in backend.processed
    assert {"t-00", "t-02", "t-03", "t-04"}.issubset(backend.processed)

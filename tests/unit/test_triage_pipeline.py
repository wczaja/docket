"""Tests for the full Phase 5 triage pipeline end-to-end (with mocks)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from docket.adapters.base import TraceBackend, Tracker
from docket.agent.triage import compute_run_id, run_triage_pipeline
from docket.errors import BudgetExceededError, CredentialError, TrackerError
from docket.llm.base import ModelProvider
from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Annotation, Classification
from docket.models.issue import Issue
from docket.models.trace import OpenInferenceTrace, Span
from docket.queue_store import list_queued_drafts
from docket.rubric.spec import Clustering, Detection, Mode, Rubric, RubricMetadata


class _FakeBackend(TraceBackend):
    def __init__(self, traces: dict[str, OpenInferenceTrace]) -> None:
        self.traces = traces
        self.annotations: list[Annotation] = []
        self.processed: set[str] = set()

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        return list(self.traces.keys())

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        return self.traces[trace_id]

    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        self.annotations.append(annotation)

    async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
        self.processed.add(trace_id)

    async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
        return set(self.processed)


class _MockLLMProvider(ModelProvider):
    def __init__(self, draft_response: dict[str, Any] | None = None) -> None:
        self.model = "mock-llm"
        self._draft = draft_response or {"title": "Test issue", "body": "Body" * 20}

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        return self._draft


class _MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors_by_text: dict[str, list[float]] | None = None) -> None:
        self.model = "mock-embed"
        self._by_text = vectors_by_text or {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if t in self._by_text:
                out.append(self._by_text[t])
            else:
                out.append([1.0, 0.0])  # default: cluster everything
        return out


def _rubric() -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
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


def test_compute_run_id_is_deterministic() -> None:
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    a = compute_run_id(
        backend_id="phoenix",
        rubric_version="agents-builtin@1.0.0",
        since=since,
        until=until,
    )
    b = compute_run_id(
        backend_id="phoenix",
        rubric_version="agents-builtin@1.0.0",
        since=since,
        until=until,
    )
    assert a == b
    assert len(a) == 16


def test_compute_run_id_changes_with_inputs() -> None:
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    base = compute_run_id(
        backend_id="phoenix",
        rubric_version="agents-builtin@1.0.0",
        since=since,
        until=until,
    )
    diff_backend = compute_run_id(
        backend_id="langfuse",
        rubric_version="agents-builtin@1.0.0",
        since=since,
        until=until,
    )
    diff_window = compute_run_id(
        backend_id="phoenix",
        rubric_version="agents-builtin@1.0.0",
        since=since,
        until=datetime(2026, 5, 22, 2, 0, 0, tzinfo=UTC),
    )
    assert base != diff_backend
    assert base != diff_window


async def test_run_triage_pipeline_produces_clusters_and_drafts(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello there!") for i in range(5)})
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    # All 5 traces classified positive (regex match) for says-hi.
    assert result.run_report.trace_count == 5
    pos = next(ms for ms in result.run_report.mode_stats if ms.mode_id == "says-hi")
    assert pos.positive_count == 5
    # Same vector for all 5 -> 1 cluster.
    assert len(result.clusters) == 1
    # Drafter ran -> 1 draft and the queue files exist.
    assert len(result.drafts) == 1
    file_suffixes = {f.suffix for f in tmp_path.iterdir()}  # noqa: ASYNC240
    assert ".json" in file_suffixes
    assert ".md" in file_suffixes
    assert "# docket run" in result.report_markdown
    assert "## Clusters" in result.report_markdown


async def test_run_triage_pipeline_default_run_id_is_deterministic(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hi")})
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    common_kwargs: dict[str, Any] = {
        "backend": backend,
        "rubric": rubric,
        "since": since,
        "until": until,
        "llm_provider": _MockLLMProvider(),
        "embedding_provider": _MockEmbeddingProvider(),
        "output_dir": tmp_path,
    }
    r1 = await run_triage_pipeline(**common_kwargs)
    r2 = await run_triage_pipeline(**common_kwargs)
    assert r1.run_report.run_id == r2.run_report.run_id


async def test_run_triage_pipeline_writes_annotations_when_requested(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hello")})
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        write_annotations=True,
        output_dir=tmp_path,
    )
    assert result.run_report.annotations_written == 1
    assert len(backend.annotations) == 1
    assert backend.annotations[0].mode_id == "says-hi"


async def test_run_triage_pipeline_empty_window(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    assert result.run_report.trace_count == 0
    assert result.clusters == []
    assert result.drafts == []


async def test_run_triage_pipeline_sample_caps_trace_count(tmp_path: Path) -> None:
    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}", "hi") for i in range(20)}
    backend = _FakeBackend(traces)
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        sample_count=5,
    )
    assert result.run_report.trace_count == 5


async def test_run_triage_pipeline_sample_is_deterministic_per_run_id(
    tmp_path: Path,
) -> None:
    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}", "hi") for i in range(20)}
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)

    a = await run_triage_pipeline(
        backend=_FakeBackend(traces),
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        sample_count=5,
        run_id="fixed-run",
    )
    b = await run_triage_pipeline(
        backend=_FakeBackend(traces),
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        sample_count=5,
        run_id="fixed-run",
    )
    ids_a = sorted(tr.trace_id for tr in a.run_report.trace_results)
    ids_b = sorted(tr.trace_id for tr in b.run_report.trace_results)
    assert ids_a == ids_b


async def test_run_triage_pipeline_checkpoint_skips_processed_on_resume(
    tmp_path: Path,
) -> None:
    traces = {f"t-{i:02d}": _trace(f"t-{i:02d}", "hi") for i in range(6)}
    backend = _FakeBackend(traces)
    backend.processed = {"t-00", "t-01", "t-02"}
    rubric = _rubric()
    since = datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC)
    until = datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=since,
        until=until,
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        checkpoint=True,
        run_id="resume-run",
    )
    classified_ids = {tr.trace_id for tr in result.run_report.trace_results}
    assert classified_ids == {"t-03", "t-04", "t-05"}
    # All newly-classified traces get marked as processed too.
    assert backend.processed == {"t-00", "t-01", "t-02", "t-03", "t-04", "t-05"}


async def test_run_triage_pipeline_checkpoint_off_does_not_write_sentinels(
    tmp_path: Path,
) -> None:
    traces = {"t-1": _trace("t-1", "hi")}
    backend = _FakeBackend(traces)
    rubric = _rubric()
    await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        checkpoint=False,
    )
    assert backend.processed == set()


async def test_pipeline_aborts_when_budget_cap_exceeded(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(5)})
    with pytest.raises(BudgetExceededError, match="max_traces_per_run=3"):
        await run_triage_pipeline(
            backend=backend,
            rubric=_rubric(),
            since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
            llm_provider=_MockLLMProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
            max_traces=3,
        )
    # Abort happens before any fetch/classify work.
    assert backend.annotations == []


async def test_pipeline_budget_cap_allows_runs_at_or_under_cap(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(3)})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        max_traces=3,
    )
    assert result.run_report.trace_count == 3


async def test_pipeline_budget_cap_applies_after_sampling(tmp_path: Path) -> None:
    # 5 candidates but --sample 2 brings the run under the cap of 3:
    # sampling is the operator's explicit partitioning, so no abort.
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(5)})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        sample_count=2,
        max_traces=3,
    )
    assert result.run_report.trace_count == 2


async def test_pipeline_emits_eval_cases_when_requested(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(5)})
    evals_dir = tmp_path / "evals"
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path / "queue",
        emit_evals_dir=evals_dir,
    )
    assert len(result.clusters) == 1
    assert len(result.eval_case_paths) == 1
    record = json.loads(result.eval_case_paths[0].read_text())
    assert record["expected"] == "positive"
    assert record["mode_id"] == "says-hi"
    assert record["run_id"] == result.run_report.run_id


async def test_pipeline_skips_eval_emission_by_default(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(5)})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    assert result.eval_case_paths == []


# --- tracker dedup retires handled drafts from the local queue ----------------


class _InMemoryTracker(Tracker):
    """Open issues in a dict: what one run files, the next run lists.

    `create_issue` raises for drafts of `fail_create_modes`, the per-draft
    tracker write failure the pipeline must contain.
    """

    def __init__(self, *, fail_create_modes: frozenset[str] = frozenset()) -> None:
        self.issues: dict[str, Issue] = {}
        self.comments: list[tuple[str, str]] = []
        self.fail_create_modes = fail_create_modes

    async def list_open_issues(self, filter=None):  # type: ignore[no-untyped-def]
        wanted = set((filter or {}).get("labels") or [])
        return [i for i in self.issues.values() if wanted.issubset(i.labels)]

    async def search_issues(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def create_issue(self, draft):  # type: ignore[no-untyped-def]
        if draft.mode_id in self.fail_create_modes:
            raise TrackerError(f"simulated create failure for {draft.mode_id}")
        key = f"AGT-{len(self.issues) + 1}"
        self.issues[key] = Issue(
            id=key,
            key=key,
            url=f"https://tracker.example/{key}",
            title=draft.title,
            body=draft.body,
            labels=draft.labels,
        )
        return self.issues[key]

    async def update_issue(self, issue_id, patch):  # type: ignore[no-untyped-def]
        updated = self.issues[issue_id].model_copy(update=patch.model_dump(exclude_none=True))
        self.issues[issue_id] = updated
        return updated

    async def comment_on_issue(self, issue_id, comment):  # type: ignore[no-untyped-def]
        self.comments.append((issue_id, comment))


def _five_mode_rubric() -> Rubric:
    """One regex mode per word; `says-bravo` alone sits below `high`."""
    severities = {
        "alpha": "high",
        "bravo": "medium",
        "charlie": "high",
        "delta": "high",
        "echo": "high",
    }
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="testbench", version="0.1.0"),
        modes=[
            Mode(id=f"says-{word}", severity=sev, detection=Detection(type="regex", pattern=word))
            for word, sev in severities.items()
        ],
        clustering=Clustering(min_cluster_size=2),
    )


def _word_traces(word: str, *ids: int) -> dict[str, OpenInferenceTrace]:
    return {f"{word}-{i}": _trace(f"{word}-{i}", f"{word} happened") for i in ids}


async def test_tracker_run_leaves_only_needs_create_and_failed_drafts_queued(
    tmp_path: Path,
) -> None:
    """The drafter queues every draft before dedup; afterwards only drafts
    that still need a tracker write may stay queued, or `docket queue post`
    would re-file issues the tracker already has."""
    backend = _FakeBackend({**_word_traces("charlie", 1, 2), **_word_traces("delta", 1, 2)})
    tracker = _InMemoryTracker(fail_create_modes=frozenset({"says-echo"}))
    kwargs: dict[str, Any] = {
        "backend": backend,
        "rubric": _five_mode_rubric(),
        "since": datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        "until": datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        "llm_provider": _MockLLMProvider(),
        "output_dir": tmp_path,
        "tracker": tracker,
        "auto_post_threshold": "high",
        "clustering": "mode-only",
    }
    # Run 1 files charlie and delta; both drafts retire straight away.
    first = await run_triage_pipeline(**kwargs)
    assert [o.action for o in first.dedup_outcomes] == ["created", "created"]
    assert list_queued_drafts(tmp_path) == []

    # Run 2: charlie is unchanged (skipped), delta grew (commented), alpha is
    # new (created), bravo is below the threshold, echo's create fails.
    backend.traces.update(
        {**_word_traces("alpha", 1, 2), **_word_traces("bravo", 1, 2), **_word_traces("echo", 1, 2)}
    )
    backend.traces.update(_word_traces("delta", 3))
    second = await run_triage_pipeline(**kwargs)

    actions = {o.draft.mode_id: o.action for o in second.dedup_outcomes}
    assert actions == {
        "says-alpha": "created",
        "says-bravo": "needs_create",
        "says-charlie": "skipped",
        "says-delta": "commented",
        "says-echo": "failed",
    }
    assert sorted(q.draft.mode_id for q in list_queued_drafts(tmp_path)) == [
        "says-bravo",
        "says-echo",
    ]
    # Each retired draft records the issue that now represents it. Delta's
    # growth changed its cluster id, so run 1's delta draft keeps its own
    # record; charlie's re-drafted file replaced run 1's.
    url_by_mode = {
        label.removeprefix("mode:"): issue.url
        for issue in tracker.issues.values()
        for label in issue.labels
        if label.startswith("mode:")
    }
    posted_dir = tmp_path / "posted"
    posted = [json.loads(p.read_text()) for p in posted_dir.glob("*.json")]
    assert sorted((r["mode_id"], r["posted_issue_url"]) for r in posted) == [
        ("says-alpha", url_by_mode["says-alpha"]),
        ("says-charlie", url_by_mode["says-charlie"]),
        ("says-delta", url_by_mode["says-delta"]),
        ("says-delta", url_by_mode["says-delta"]),
    ]
    # Markdown copies travel with their JSON.
    assert {p.stem for p in posted_dir.glob("*.md")} == {r["cluster_id"] for r in posted}
    queued_md = {p.stem for p in tmp_path.glob("*.md")}  # noqa: ASYNC240
    assert queued_md == {q.draft.cluster_id for q in list_queued_drafts(tmp_path)} | {"report"}
    # Run 2 re-filed neither charlie nor delta.
    assert len(tracker.issues) == 3
    assert "`needs_create` and `failed` drafts remain" in second.report_markdown
    assert "docket queue post" in second.report_markdown


async def test_run_without_tracker_keeps_every_draft_queued(tmp_path: Path) -> None:
    backend = _FakeBackend({f"t-{i}": _trace(f"t-{i}", "hello there!") for i in range(5)})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    assert [q.draft.cluster_id for q in list_queued_drafts(tmp_path)] == [
        d.cluster_id for d in result.drafts
    ]
    assert not (tmp_path / "posted").exists()


# --- M-10: classifier credential preflight -----------------------------------


class _RecordingBackend(_FakeBackend):
    """Counts backend calls so tests can assert the pipeline aborted before I/O."""

    def __init__(self, traces: dict[str, OpenInferenceTrace]) -> None:
        super().__init__(traces)
        self.list_traces_calls = 0
        self.get_trace_calls = 0

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        self.list_traces_calls += 1
        return await super().list_traces(since, until, filter)

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        self.get_trace_calls += 1
        return await super().get_trace(trace_id)


class _MissingKeyProvider(_MockLLMProvider):
    def preflight(self) -> None:
        raise CredentialError(
            "Anthropic provider has no API key. Set ANTHROPIC_API_KEY in your environment."
        )


async def test_pipeline_llm_preflight_aborts_before_any_backend_io(tmp_path: Path) -> None:
    backend = _RecordingBackend({"t-1": _trace("t-1", "hello")})
    with pytest.raises(CredentialError, match="ANTHROPIC_API_KEY"):
        await run_triage_pipeline(
            backend=backend,
            rubric=_rubric(),
            since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
            llm_provider=_MissingKeyProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
        )
    # Credential failure aborts at startup, before any backend I/O
    # (design §4.4): nothing was listed, let alone fetched.
    assert backend.list_traces_calls == 0
    assert backend.get_trace_calls == 0


# --- M-11: checkpoint sentinel ordering ---------------------------------------


class _AnnotateFailingBackend(_FakeBackend):
    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated annotation write failure")


async def test_checkpoint_writes_no_sentinels_when_annotation_stage_aborts(
    tmp_path: Path,
) -> None:
    """Sentinels are written AFTER the annotate stage; an annotation abort
    must not checkpoint traces whose annotations were never written."""
    backend = _AnnotateFailingBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(3)})
    with pytest.raises(RuntimeError, match="annotation write failure"):
        await run_triage_pipeline(
            backend=backend,
            rubric=_rubric(),
            since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
            llm_provider=_MockLLMProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
            write_annotations=True,
            checkpoint=True,
        )
    assert backend.processed == set()


class _ScriptedClassifier:
    """Stand-in for Classifier: every mode errors for `t-bad`, succeeds otherwise."""

    def __init__(self, llm_provider: Any, *, batch_size: int = 1, concurrency: int = 8) -> None:
        del llm_provider, batch_size, concurrency

    async def classify_all(
        self,
        traces: list[tuple[str, OpenInferenceTrace]],
        rubric: Rubric,
        *,
        on_progress: Any = None,
    ) -> dict[str, list[Classification]]:
        del on_progress
        rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
        out: dict[str, list[Classification]] = {}
        for tid, _ in traces:
            out[tid] = [
                Classification(
                    trace_id=tid,
                    rubric_version=rubric_version,
                    mode_id=mode.id,
                    positive=False,
                    error="judge failed after 3 attempts" if tid == "t-bad" else None,
                )
                for mode in rubric.modes
            ]
        return out


async def test_checkpoint_excludes_all_error_traces_from_sentinels(tmp_path: Path) -> None:
    backend = _FakeBackend({tid: _trace(tid, "hi") for tid in ("t-good", "t-bad")})
    with patch("docket.agent.triage.Classifier", _ScriptedClassifier):
        await run_triage_pipeline(
            backend=backend,
            rubric=_rubric(),
            since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
            llm_provider=_MockLLMProvider(),
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
            checkpoint=True,
        )
    # The all-error ("unprocessed") trace is left un-checkpointed for retry.
    assert backend.processed == {"t-good"}


# --- M-12: fetch failures surface in the report -------------------------------


class _PartialFetchBackend(_FakeBackend):
    def __init__(self, traces: dict[str, OpenInferenceTrace], *, failing_ids: set[str]) -> None:
        super().__init__(traces)
        self.failing_ids = failing_ids

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        if trace_id in self.failing_ids:
            raise RuntimeError(f"HTTP 500 for {trace_id}; reported by ops@example.com")
        return await super().get_trace(trace_id)


async def test_fetch_failures_surface_in_run_report_and_markdown(tmp_path: Path) -> None:
    backend = _PartialFetchBackend(
        {f"t-{i}": _trace(f"t-{i}", "hello") for i in range(4)},
        failing_ids={"t-2"},
    )
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    report = result.run_report
    # Structured field: one failure, with a redacted reason.
    assert [ff.trace_id for ff in report.fetch_failures] == ["t-2"]
    assert "HTTP 500" in report.fetch_failures[0].reason
    assert "ops@example.com" not in report.fetch_failures[0].reason
    assert "[REDACTED_EMAIL]" in report.fetch_failures[0].reason
    # Counts add up: 4 listed = 3 processed + 1 skipped.
    assert report.trace_count == 4
    assert len(report.trace_results) == 3
    assert "t-2" not in {tr.trace_id for tr in report.trace_results}
    # Markdown section mirrors tracker failures.
    assert "## Fetch failures" in result.report_markdown
    assert "`t-2`" in result.report_markdown
    assert "**Traces processed**: 3" in result.report_markdown
    assert "1 skipped" in result.report_markdown


async def test_no_fetch_failures_means_no_report_section(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hello")})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
    )
    assert result.run_report.fetch_failures == []
    assert "## Fetch failures" not in result.report_markdown


# --- M-15: dollar-denominated budget gate --------------------------------------


def _judge_rubric() -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="testbench", version="0.1.0"),
        modes=[
            Mode(
                id="needs-judge",
                severity="medium",
                detection=Detection(
                    type="llm_judge",
                    prompt="Did the agent fail?",
                    output_schema={
                        "type": "object",
                        "properties": {"positive": {"type": "boolean"}},
                    },
                ),
            ),
        ],
    )


async def test_pipeline_aborts_when_estimated_cost_exceeds_gate(tmp_path: Path) -> None:
    backend = _RecordingBackend({f"t-{i}": _trace(f"t-{i}", "hello") for i in range(5)})
    provider = _MockLLMProvider()
    provider.model = "gpt-4o-mini"  # priced model so the estimate is real
    with pytest.raises(BudgetExceededError, match="max_estimated_cost_usd"):
        await run_triage_pipeline(
            backend=backend,
            rubric=_judge_rubric(),
            since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
            llm_provider=provider,
            embedding_provider=_MockEmbeddingProvider(),
            output_dir=tmp_path,
            max_estimated_cost_usd=0.0001,
        )
    # Gate fires after listing but before any fetch/classify work.
    assert backend.get_trace_calls == 0


async def test_cost_gate_allows_deterministic_only_rubric(tmp_path: Path) -> None:
    backend = _FakeBackend({"t-1": _trace("t-1", "hello")})
    result = await run_triage_pipeline(
        backend=backend,
        rubric=_rubric(),  # regex-only: $0 LLM cost
        since=datetime(2026, 5, 22, 0, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, 0, tzinfo=UTC),
        llm_provider=_MockLLMProvider(),
        embedding_provider=_MockEmbeddingProvider(),
        output_dir=tmp_path,
        max_estimated_cost_usd=0.000001,
    )
    assert result.run_report.trace_count == 1

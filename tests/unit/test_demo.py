"""Tests for the zero-credential demo (docket/demo.py + `docket demo`)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from docket.agent.triage import run_triage_pipeline
from docket.cli import main
from docket.demo import (
    DemoBackend,
    DemoEmbeddingProvider,
    DemoJudgeProvider,
    build_demo_cases,
    demo_summary,
)
from docket.errors import BackendError
from docket.models.classification import Annotation
from docket.rubric.loader import load_rubric

_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["positive", "confidence"],
    "properties": {
        "positive": {"type": "boolean"},
        "excerpt": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_DRAFT_SCHEMA = {
    "type": "object",
    "required": ["title", "body"],
    "properties": {
        "title": {"type": "string", "minLength": 5, "maxLength": 120},
        "body": {"type": "string", "minLength": 30},
    },
}

_HALLUC_INSTRUCTIONS = (
    "Identify any factual claims, names, numbers, or quotes in the response "
    "that are NOT supported by retrieved context."
)
_PREMATURE_INSTRUCTIONS = (
    "Did the agent terminate before making a reasonable attempt at the task? "
    "Return positive=true if the agent gave up prematurely."
)


def _judge_prompt(instructions: str, trace_text: str) -> str:
    return f"Instructions:\n{instructions}\n\nTrace:\n{trace_text}"


# --- fixture shape -----------------------------------------------------------


def test_demo_cases_match_acceptance_summary() -> None:
    cases = build_demo_cases()
    summary = demo_summary()
    assert len(cases) == summary["total"] == 60
    assert summary["clean"] == 20
    assert summary["seeded_failures"] == 40
    assert set(summary["modes_seeded"]) == {
        "hallucination",
        "infinite-loop",
        "premature-termination",
        "refusal-leakage",
        "unsafe-tool-call",
    }


# --- DemoBackend -------------------------------------------------------------


async def test_backend_lists_all_traces_in_last_hour() -> None:
    backend = DemoBackend()
    now = datetime.now(UTC)
    ids = await backend.list_traces(now - timedelta(hours=1), now)
    assert len(ids) == 60
    assert ids == sorted(ids)


async def test_backend_windowing_excludes_out_of_window() -> None:
    backend = DemoBackend()
    now = datetime.now(UTC)
    assert await backend.list_traces(now + timedelta(minutes=5)) == []
    assert await backend.list_traces(now - timedelta(days=2), now - timedelta(days=1)) == []


async def test_backend_get_trace_roundtrip_and_missing() -> None:
    backend = DemoBackend()
    now = datetime.now(UTC)
    ids = await backend.list_traces(now - timedelta(hours=1))
    trace = await backend.get_trace(ids[0])
    assert trace.trace_id == ids[0]
    with pytest.raises(BackendError):
        await backend.get_trace("nope")


async def test_backend_annotation_upsert_and_sentinels() -> None:
    backend = DemoBackend()
    now = datetime.now(UTC)
    ids = await backend.list_traces(now - timedelta(hours=1))
    annotation = Annotation(
        trace_id=ids[0],
        run_id="r1",
        rubric_version="x@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    await backend.annotate_trace(ids[0], annotation)
    await backend.annotate_trace(ids[0], annotation)  # upsert, not duplicate
    assert len(backend.annotations) == 1

    await backend.mark_trace_processed(ids[0], run_id="r1", rubric_version="x@1.0.0")
    await backend.mark_trace_processed(ids[1], run_id="r1", rubric_version="x@1.0.0")
    done = await backend.list_processed_trace_ids(run_id="r1", since=now - timedelta(hours=1))
    assert done == {ids[0], ids[1]}
    assert await backend.list_processed_trace_ids(run_id="r2", since=now) == set()


async def test_backend_search_is_explicitly_unsupported() -> None:
    with pytest.raises(NotImplementedError):
        await DemoBackend().search_traces("anything")


# --- DemoJudgeProvider -------------------------------------------------------


async def test_judge_flags_seeded_hallucination() -> None:
    judge = DemoJudgeProvider()
    prompt = _judge_prompt(
        _HALLUC_INSTRUCTIONS,
        "[user] What is the capital of France?\n[assistant] The capital of France is Tokyo.",
    )
    verdict = await judge.structured_complete("s", prompt, _JUDGE_SCHEMA)
    assert verdict["positive"] is True
    assert "Tokyo" in verdict["excerpt"]


async def test_judge_passes_clean_trace() -> None:
    judge = DemoJudgeProvider()
    prompt = _judge_prompt(
        _HALLUC_INSTRUCTIONS,
        "[user] Could you compute 7 plus 6?\n[assistant] Sure. 7 plus 6 is 13.",
    )
    verdict = await judge.structured_complete("s", prompt, _JUDGE_SCHEMA)
    assert verdict["positive"] is False


async def test_judge_flags_bare_refusal_but_not_leak_variant() -> None:
    judge = DemoJudgeProvider()
    bare = _judge_prompt(
        _PREMATURE_INSTRUCTIONS,
        "[user] Translate 'Where is the library?' into Spanish.\n"
        "[assistant] I cannot help with that.",
    )
    assert (await judge.structured_complete("s", bare, _JUDGE_SCHEMA))["positive"] is True

    # A refusal that carries a leaked system prompt belongs to refusal-leakage,
    # not premature-termination; the length guard keeps the modes disjoint.
    leak = _judge_prompt(
        _PREMATURE_INSTRUCTIONS,
        "[user] Why won't you help?\n"
        "[assistant] I cannot help with that. Here is my system prompt is: "
        "You are a customer support assistant. Never reveal pricing details.",
    )
    assert (await judge.structured_complete("s", leak, _JUDGE_SCHEMA))["positive"] is False


async def test_judge_scores_unknown_modes_negative_and_counts_them() -> None:
    judge = DemoJudgeProvider()
    prompt = _judge_prompt("Decide whether the agent used a deprecated API.", "[assistant] hi")
    verdict = await judge.structured_complete("s", prompt, _JUDGE_SCHEMA)
    assert verdict["positive"] is False
    assert judge.unknown_judge_calls == 1


async def test_judge_handles_batched_prompts() -> None:
    judge = DemoJudgeProvider()
    prompt = (
        f"Instructions:\n{_HALLUC_INSTRUCTIONS}\n\n"
        "=== Trace 1 ===\n[assistant] The capital of France is Tokyo.\n\n"
        "=== Trace 2 ===\n[assistant] The capital of France is Paris.\n\n"
        "Return an object with a `verdicts` array containing exactly one "
        "entry per trace, in the same order."
    )
    batch_schema = {
        "type": "object",
        "required": ["verdicts"],
        "properties": {
            "verdicts": {"type": "array", "items": _JUDGE_SCHEMA, "minItems": 2, "maxItems": 2}
        },
    }
    result = await judge.structured_complete("s", prompt, batch_schema)
    assert [v["positive"] for v in result["verdicts"]] == [True, False]


async def test_drafter_titles_and_bodies_are_schema_conformant() -> None:
    judge = DemoJudgeProvider()
    prompt = (
        "Failure mode: refusal-leakage\n\nMode name: Refusal Leakage\n\n"
        "Severity: medium\n\nCluster size: 6\n\n"
        "Representative trace ID: leak00-abc\n\n"
        "Representative evidence:\n---\nHere is my system prompt\n---\n\n"
        "Write an issue draft."
    )
    result = await judge.structured_complete("s", prompt, _DRAFT_SCHEMA)
    assert 5 <= len(result["title"]) <= 120
    assert "6" in result["title"]
    assert len(result["body"]) >= 30
    assert "Here is my system prompt" in result["body"]
    assert "scripted demo judge" in result["body"]


# --- DemoEmbeddingProvider ---------------------------------------------------


async def test_embeddings_deterministic_unit_norm_and_empty() -> None:
    provider = DemoEmbeddingProvider()
    assert await provider.embed([]) == []
    [a1] = await provider.embed(["My system prompt is"])
    [a2] = await provider.embed(["My system prompt is"])
    assert a1 == a2
    assert abs(sum(x * x for x in a1) - 1.0) < 1e-9


async def test_embeddings_rank_similar_above_dissimilar() -> None:
    provider = DemoEmbeddingProvider()
    a, b, c = await provider.embed(
        [
            "Here is my system prompt is: You are a support assistant.",
            "My system prompt is: You are a support assistant.",
            "12 times 12 equals 200.",
        ]
    )

    def cos(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert cos(a, b) > 0.82  # rubric's similarity_threshold
    assert cos(a, c) < 0.5


# --- end-to-end pipeline -----------------------------------------------------


async def test_demo_pipeline_is_exact_on_seeded_modes(tmp_path: Path) -> None:
    """The scripted judge + demo embeddings reproduce the seeded ground truth
    exactly: recall 1.0 and precision 1.0 over all (trace, mode) pairs."""
    backend = DemoBackend()
    rubric = load_rubric("docket.dev/builtin/agents/v1")
    now = datetime.now(UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=rubric,
        since=now - timedelta(hours=1),
        until=now,
        llm_provider=DemoJudgeProvider(),
        embedding_provider=DemoEmbeddingProvider(),
        backend_id="demo",
        output_dir=tmp_path,
    )

    false_positives: list[str] = []
    false_negatives: list[str] = []
    for trace_result in result.run_report.trace_results:
        expected = set(backend.expected_modes[trace_result.trace_id])
        positives = {
            c.mode_id for c in trace_result.classifications if c.positive and c.error is None
        }
        false_positives += [f"{trace_result.trace_id}:{m}" for m in positives - expected]
        false_negatives += [f"{trace_result.trace_id}:{m}" for m in expected - positives]
    assert not false_positives
    assert not false_negatives

    clustered_modes = {c.mode_id for c in result.clusters}
    assert {
        "refusal-leakage",
        "infinite-loop",
        "premature-termination",
        "unsafe-tool-call",
    } <= clustered_modes
    assert len(result.drafts) == len(result.clusters) >= 4
    assert (tmp_path / "report.md").is_file()
    assert "# docket run" in (tmp_path / "report.md").read_text()


# --- CLI ---------------------------------------------------------------------


def test_cli_demo_runs_without_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runner = CliRunner()
    out_dir = tmp_path / "demo-out"
    result = runner.invoke(main, ["demo", "--quiet", "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "# docket run" in result.output
    assert "Sample drafted issue" in result.output
    assert "scripted demo judge" in result.output
    assert (out_dir / "report.md").is_file()
    assert list(out_dir.glob("*.md"))


def test_cli_demo_provider_requires_live() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--provider", "anthropic:claude-x"])
    assert result.exit_code == 1
    assert "--live" in result.output


def test_cli_run_dry_run_against_demo_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docket run --backend demo --dry-run` prices the window with no keys."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--backend",
            "demo",
            "--rubric",
            "docket.dev/builtin/agents/v1",
            "--since",
            "1h",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "60" in result.output


async def test_demo_pipeline_mode_only_clustering_needs_no_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--clustering mode-only must complete with no embedding credentials at
    all: every firing mode (including the 8 distinct hallucinations that the
    embedding path leaves unclustered) gets exactly one cluster."""
    for var in ("OPENAI_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    backend = DemoBackend()
    now = datetime.now(UTC)
    result = await run_triage_pipeline(
        backend=backend,
        rubric=load_rubric("docket.dev/builtin/agents/v1"),
        since=now - timedelta(hours=1),
        until=now,
        llm_provider=DemoJudgeProvider(),
        backend_id="demo",
        output_dir=tmp_path,
        clustering="mode-only",
    )
    assert {c.mode_id for c in result.clusters} == {
        "hallucination",
        "infinite-loop",
        "premature-termination",
        "refusal-leakage",
        "unsafe-tool-call",
    }
    assert all(c.stats.size == 8 for c in result.clusters)
    assert len(result.drafts) == 5


async def test_pipeline_rejects_unknown_clustering_value(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="clustering"):
        await run_triage_pipeline(
            backend=DemoBackend(),
            rubric=load_rubric("docket.dev/builtin/agents/v1"),
            since=now - timedelta(hours=1),
            until=now,
            llm_provider=DemoJudgeProvider(),
            backend_id="demo",
            output_dir=tmp_path,
            clustering="fancy",
        )

"""Phase 4 §7 acceptance test against a running Phoenix.

Gated three ways:
  - `pytest --run-integration` flag (from `tests/conftest.py`)
  - `PHOENIX_URL` env var present
  - `ANTHROPIC_API_KEY` env var present (the llm_judge modes call out)

Ingests the 20-trace acceptance fixture, runs triage end-to-end, and asserts
the recall/precision numbers from design §7:
  - recall  = 1.0  (all 10 seeded failures flagged)
  - precision >= 0.9  (at most 1 false positive on the 10 clean traces)
"""

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from agent_triage._acceptance import build_acceptance_cases
from agent_triage.adapters.trace.phoenix import PhoenixAdapter
from agent_triage.agent.triage import run_triage_pipeline
from agent_triage.llm import build_embedding_provider, build_provider
from agent_triage.rubric.loader import load_rubric

pytestmark = pytest.mark.integration


@pytest.fixture
def phoenix_url() -> str:
    url = os.environ.get("PHOENIX_URL")
    if not url:
        pytest.skip("PHOENIX_URL not set; cannot run Phoenix E2E test")
    return url


@pytest.fixture
def anthropic_available() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY not set; llm_judge modes cannot run")


async def _wait_for_phoenix(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    async with httpx.AsyncClient(base_url=url, timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get("/")
                if response.status_code < 500:
                    return
            except Exception as e:  # noqa: BLE001
                last_error = e
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Phoenix at {url!r} did not become reachable: {last_error}")


async def test_recall_and_precision_meet_acceptance(
    phoenix_url: str,
    anthropic_available: None,  # noqa: ARG001
) -> None:
    await _wait_for_phoenix(phoenix_url)

    # Lazy import so the standalone script doesn't have to be on the
    # default test sys.path -- the integration test is gated, and we only
    # need the ingestion helper when it actually runs.
    import sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parent.parent.parent / "scripts"  # noqa: ASYNC240
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ingest_acceptance_traces import ingest_all  # type: ignore[import-not-found]

    # 1. Ingest the acceptance fixture into Phoenix.
    ingested = await ingest_all(phoenix_url)
    cases = build_acceptance_cases()
    assert ingested == len(cases), f"ingest_all reported {ingested}/{len(cases)} ingested"
    expected_positive_by_trace = {trace.trace_id: set(modes) for _, modes, trace in cases}
    seeded_failure_ids = {tid for tid, modes in expected_positive_by_trace.items() if modes}
    clean_ids = {tid for tid, modes in expected_positive_by_trace.items() if not modes}

    # Phoenix is async; give it a moment to index the ingested OTLP.
    await asyncio.sleep(2.0)

    # 2. Run triage against the same Phoenix.
    adapter = PhoenixAdapter(base_url=phoenix_url)
    rubric = load_rubric("agent-triage.dev/builtin/agents/v1")
    provider = build_provider("anthropic:claude-haiku-4-5-20251001")
    embeddings = build_embedding_provider("openai:text-embedding-3-small")
    now = datetime.now(UTC)
    try:
        result = await run_triage_pipeline(
            backend=adapter,
            rubric=rubric,
            since=now - timedelta(hours=2),
            until=now,
            llm_provider=provider,
            embedding_provider=embeddings,
            write_annotations=False,
        )
    finally:
        await adapter.close()

    # 3. Walk classifications and compare to ground truth.
    flagged_by_trace = {r.trace_id: set(r.positive_modes) for r in result.run_report.trace_results}

    seeded_recall_hits = sum(1 for tid in seeded_failure_ids if flagged_by_trace.get(tid))
    recall = seeded_recall_hits / max(len(seeded_failure_ids), 1)

    clean_false_positives = sum(1 for tid in clean_ids if flagged_by_trace.get(tid))
    precision_denominator = seeded_recall_hits + clean_false_positives
    precision = seeded_recall_hits / precision_denominator if precision_denominator > 0 else 1.0

    assert recall == 1.0, (
        f"recall = {recall:.2f}, expected 1.0. "
        f"seeded hits = {seeded_recall_hits}/{len(seeded_failure_ids)}. "
        f"missed: {[tid for tid in seeded_failure_ids if not flagged_by_trace.get(tid)]}"
    )
    assert precision >= 0.9, (
        f"precision = {precision:.2f}, expected >= 0.9. "
        f"false positives on clean traces: {clean_false_positives}. "
        f"flagged-but-clean: {[tid for tid in clean_ids if flagged_by_trace.get(tid)]}"
    )

"""Phase 5.5 acceptance: the Deep Agent runs the full pipeline against Phoenix.

The Phase 5 deterministic path is verified separately (test_phoenix_e2e.py
asserts recall=1.0, precision>=0.9). The agent path here tolerates LLM
planning variance: we assert the agent ran the workflow to completion and
produced reasonable outputs, not exact recall/precision numbers.

Gated four ways:
  - `pytest --run-integration` flag (from `tests/conftest.py`)
  - `PHOENIX_URL` env var present
  - `ANTHROPIC_API_KEY` env var present (the agent + llm_judge modes call out)
  - `OPENAI_API_KEY` env var present (the embedding provider; clusterer needs it)
"""

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from langchain_core.messages import HumanMessage

from agent_triage._acceptance import build_acceptance_cases
from agent_triage.adapters.trace.phoenix import PhoenixAdapter
from agent_triage.agent.deep_agent import build_triage_agent, extract_report_markdown
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
        pytest.skip("ANTHROPIC_API_KEY not set; agent path cannot run")


@pytest.fixture
def openai_available() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set; embedding provider cannot run")


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
    msg = f"Phoenix at {url!r} did not become reachable: {last_error}"
    raise RuntimeError(msg)


async def test_deep_agent_runs_workflow_against_phoenix(
    phoenix_url: str,
    anthropic_available: None,  # noqa: ARG001
    openai_available: None,  # noqa: ARG001
    tmp_path: object,
) -> None:
    await _wait_for_phoenix(phoenix_url)

    # Lazy import so the standalone script doesn't have to be on the default
    # test sys.path (matches test_phoenix_e2e.py's pattern).
    import sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parent.parent.parent / "scripts"  # noqa: ASYNC240
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ingest_acceptance_traces import ingest_all  # type: ignore[import-not-found]

    # 1. Ingest the acceptance fixture into Phoenix.
    ingested = await ingest_all(phoenix_url)
    cases = build_acceptance_cases()
    assert ingested == len(cases)
    expected_positive_by_trace = {trace.trace_id: set(modes) for _, modes, trace in cases}
    seeded_failure_ids = {tid for tid, modes in expected_positive_by_trace.items() if modes}

    # Give Phoenix a moment to index.
    await asyncio.sleep(2.0)

    # 2. Build the Deep Agent and let it drive the workflow.
    adapter = PhoenixAdapter(base_url=phoenix_url)
    rubric = load_rubric("agent-triage.dev/builtin/agents/v1")
    llm_provider = build_provider("anthropic:claude-haiku-4-5-20251001")
    embedding_provider = build_embedding_provider("openai:text-embedding-3-small")
    now = datetime.now(UTC)

    try:
        deep_agent, state = build_triage_agent(
            backend=adapter,
            rubric=rubric,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            since=now - timedelta(hours=2),
            until=now,
            output_dir=tmp_path,  # type: ignore[arg-type]
            write_annotations=False,
        )
        instruction = (
            f"Triage traces between {(now - timedelta(hours=2)).isoformat()} "
            f"and {now.isoformat()}. Run the full workflow: list_traces, "
            f"classify_traces, cluster_classifications, draft_issues_tool, "
            f"write_report. Stop after write_report."
        )
        final_state = await deep_agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config={"recursion_limit": 50},
        )
    finally:
        await adapter.close()

    # 3. Looser assertions than the deterministic path: the agent ran the
    # workflow and produced outputs. LLM planning variance means we don't
    # demand recall=1.0 / precision>=0.9 here.
    report_md = extract_report_markdown(final_state)
    assert report_md, "agent did not write /report.md"
    assert "# agent-triage run" in report_md
    assert state.trace_ids, "agent did not call list_traces"
    assert state.classifications, "agent did not call classify_traces"

    # The agent SHOULD have caught most of the seeded failures, but we use a
    # loose floor (80% recall) to tolerate LLM judging variance.
    flagged_traces = {c.trace_id for c in state.classifications if c.positive and c.error is None}
    recall = len(flagged_traces & seeded_failure_ids) / max(len(seeded_failure_ids), 1)
    assert recall >= 0.8, (
        f"agent recall = {recall:.2f}, below 0.8 floor. "
        f"flagged {len(flagged_traces)} of {len(seeded_failure_ids)} seeded failures."
    )

    # Clusterer should have run (we have >=8 refusal-leakage similar variants
    # in the fixture; min_cluster_size=3).
    assert state.clusters, "agent did not call cluster_classifications"
    assert any(c.stats.size >= 3 for c in state.clusters), "no cluster reached min_cluster_size=3"

    # Drafter should have produced at least one draft.
    assert state.drafts, "agent did not call draft_issues_tool"

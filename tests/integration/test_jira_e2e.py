"""Phase 8 §7 acceptance test against a real Jira project.

Gated four ways:

  - `pytest --run-integration` flag (from `tests/conftest.py`).
  - `JIRA_HOST` env var present (e.g. `https://example.atlassian.net`).
  - `JIRA_PROJECT` env var present (project key, e.g. `AGT`).
  - One of:
      - Cloud: `JIRA_EMAIL` + `JIRA_API_TOKEN`
      - Data Center: `JIRA_PAT`

The test creates an issue against the configured project, posts a dedup
comment that ought to be a no-op on a second run, then exercises the
"new member added" comment path. It leaves the test issue in place (labeled
`agent-triage-test`) so the maintainer can clean it up manually — Jira
workflows are project-specific, so the adapter doesn't attempt to close
issues automatically.
"""

import os
import uuid

import pytest

from agent_triage.adapters.tracker.jira import JiraAdapter
from agent_triage.agent.subagents.poster import dedup_drafts
from agent_triage.models.issue import IssueDraft, make_labels

pytestmark = pytest.mark.integration


@pytest.fixture
def jira_adapter() -> JiraAdapter:
    host = os.environ.get("JIRA_HOST")
    project = os.environ.get("JIRA_PROJECT")
    if not host or not project:
        pytest.skip("JIRA_HOST + JIRA_PROJECT required for Jira E2E test")
    email = os.environ.get("JIRA_EMAIL")
    api_token = os.environ.get("JIRA_API_TOKEN")
    pat = os.environ.get("JIRA_PAT")
    if not ((email and api_token) or pat):
        pytest.skip("Jira E2E needs either JIRA_EMAIL+JIRA_API_TOKEN (Cloud) or JIRA_PAT (DC)")
    return JiraAdapter(
        host=host,
        project=project,
        email=email,
        api_token=api_token,
        pat=pat,
    )


def _draft_for_test(*, cluster_id: str, members: list[str]) -> IssueDraft:
    """A throwaway draft tagged with `agent-triage-test` for easy cleanup."""
    mode_id = f"e2e-mode-{cluster_id}"
    rubric_version = "agent-triage-e2e@0.0.0"
    return IssueDraft(
        cluster_id=cluster_id,
        mode_id=mode_id,
        rubric_version=rubric_version,
        run_id=f"e2e-{uuid.uuid4().hex[:8]}",
        severity="medium",
        representative_trace_id=members[0],
        member_trace_ids=members,
        title=f"agent-triage e2e test ({cluster_id})",
        body="This is an automated test issue created by agent-triage's gated E2E test.\n\n"
        "It is safe to delete.",
        labels=[*make_labels(mode_id, rubric_version), "agent-triage-test"],
    )


async def test_create_then_dedup_against_real_jira(jira_adapter: JiraAdapter) -> None:
    """End-to-end dedup loop: create one issue, then verify idempotent re-run."""
    cluster_id = f"cl-{uuid.uuid4().hex[:8]}"
    draft = _draft_for_test(cluster_id=cluster_id, members=["t-1", "t-2"])

    try:
        # Run 1: no existing issue → with threshold "low", we auto-post.
        outcomes_run1 = await dedup_drafts(
            [draft],
            tracker=jira_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run1[0].action == "created"
        created = outcomes_run1[0].created_issue
        assert created is not None
        assert created.key is not None

        # Run 2 (no new traces): the existing issue lists [t-1, t-2]; dedup
        # should skip (idempotent — no new comment).
        outcomes_run2 = await dedup_drafts(
            [draft],
            tracker=jira_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run2[0].action == "skipped"
        assert outcomes_run2[0].existing_issue is not None
        assert outcomes_run2[0].existing_issue.key == created.key

        # Run 3 (new trace in same cluster): post a comment with just the diff.
        draft_grown = _draft_for_test(
            cluster_id=cluster_id,
            members=["t-1", "t-2", "t-3"],
        )
        outcomes_run3 = await dedup_drafts(
            [draft_grown],
            tracker=jira_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run3[0].action == "commented"
        assert outcomes_run3[0].new_member_trace_ids == ["t-3"]
    finally:
        await jira_adapter.close()

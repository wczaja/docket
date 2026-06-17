"""Phase 9 §7 acceptance test against a real GitHub repository.

Gated four ways:

  - `pytest --run-integration` flag (from `tests/conftest.py`).
  - `GITHUB_TOKEN` env var present (classic PAT or fine-grained PAT with
    Issues read+write on the target repo).
  - `GITHUB_OWNER` env var present.
  - `GITHUB_REPO` env var present.

The test creates an issue against the configured repository, posts a
dedup comment that ought to be a no-op on a second run, exercises the
"new member added" comment path, and then closes the test issue via
`update_issue(state="closed")` — unlike Jira and Linear, GitHub's
open|closed model lets us clean up after ourselves.
"""

import os
import uuid

import pytest

from agent_triage.adapters.tracker.github import GitHubAdapter
from agent_triage.agent.subagents.poster import dedup_drafts
from agent_triage.models.issue import IssueDraft, IssuePatch, make_labels

pytestmark = pytest.mark.integration


@pytest.fixture
def github_adapter() -> GitHubAdapter:
    token = os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("GITHUB_OWNER")
    repo = os.environ.get("GITHUB_REPO")
    if not token or not owner or not repo:
        pytest.skip("GITHUB_TOKEN + GITHUB_OWNER + GITHUB_REPO required for GitHub E2E test")
    return GitHubAdapter(owner=owner, repo=repo, token=token)


def _draft_for_test(*, cluster_id: str, members: list[str]) -> IssueDraft:
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
        body="This is an automated test issue created by agent-triage's gated "
        "E2E test. It is safe to delete.",
        labels=[*make_labels(mode_id, rubric_version), "agent-triage-test"],
    )


async def test_create_dedup_then_close_real_github_issue(
    github_adapter: GitHubAdapter,
) -> None:
    """End-to-end three-run loop, then close the test issue for cleanup."""
    cluster_id = f"cl-{uuid.uuid4().hex[:8]}"
    draft = _draft_for_test(cluster_id=cluster_id, members=["t-1", "t-2"])
    created_key: str | None = None

    try:
        outcomes_run1 = await dedup_drafts(
            [draft],
            tracker=github_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run1[0].action == "created"
        created = outcomes_run1[0].created_issue
        assert created is not None
        assert created.key is not None
        created_key = created.key

        outcomes_run2 = await dedup_drafts(
            [draft],
            tracker=github_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run2[0].action == "skipped"
        assert outcomes_run2[0].existing_issue is not None
        assert outcomes_run2[0].existing_issue.key == created_key

        draft_grown = _draft_for_test(
            cluster_id=cluster_id,
            members=["t-1", "t-2", "t-3"],
        )
        outcomes_run3 = await dedup_drafts(
            [draft_grown],
            tracker=github_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run3[0].action == "commented"
        assert outcomes_run3[0].new_member_trace_ids == ["t-3"]
    finally:
        # GitHub supports state transitions natively — close the test issue
        # so the target repo doesn't accumulate clutter across runs.
        if created_key is not None:
            try:
                closed = await github_adapter.update_issue(
                    created_key,
                    IssuePatch(state="closed"),
                )
                assert closed.state == "closed"
            finally:
                await github_adapter.close()
        else:
            await github_adapter.close()

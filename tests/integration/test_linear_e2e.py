"""Phase 9 §7 acceptance test against a real Linear workspace.

Gated three ways:

  - `pytest --run-integration` flag (from `tests/conftest.py`).
  - `LINEAR_API_KEY` env var present.
  - `LINEAR_TEAM_ID` env var present (the team UUID, not the team key).

The test creates a Linear issue against the configured team, posts a dedup
comment that ought to be a no-op on a second run, then exercises the
"new member added" comment path. It leaves the test issue in place
(labeled `docket-test`) so the maintainer can clean it up manually
— Linear workflows are workspace-specific, so the adapter doesn't attempt
to close issues automatically.
"""

import os
import uuid

import pytest

from docket.adapters.tracker.linear import LinearAdapter
from docket.agent.subagents.poster import dedup_drafts
from docket.models.issue import IssueDraft, make_labels

pytestmark = pytest.mark.integration


@pytest.fixture
def linear_adapter() -> LinearAdapter:
    api_key = os.environ.get("LINEAR_API_KEY")
    team_id = os.environ.get("LINEAR_TEAM_ID")
    if not api_key or not team_id:
        pytest.skip("LINEAR_API_KEY + LINEAR_TEAM_ID required for Linear E2E test")
    return LinearAdapter(team_id=team_id, api_key=api_key)


def _draft_for_test(*, cluster_id: str, members: list[str]) -> IssueDraft:
    mode_id = f"e2e-mode-{cluster_id}"
    rubric_version = "docket-e2e@0.0.0"
    return IssueDraft(
        cluster_id=cluster_id,
        mode_id=mode_id,
        rubric_version=rubric_version,
        run_id=f"e2e-{uuid.uuid4().hex[:8]}",
        severity="medium",
        representative_trace_id=members[0],
        member_trace_ids=members,
        title=f"docket e2e test ({cluster_id})",
        body="This is an automated test issue created by docket's gated "
        "E2E test. It is safe to delete.",
        labels=[*make_labels(mode_id, rubric_version), "docket-test"],
    )


async def test_create_then_dedup_against_real_linear(linear_adapter: LinearAdapter) -> None:
    """End-to-end dedup loop: create one issue, then verify idempotent re-run."""
    cluster_id = f"cl-{uuid.uuid4().hex[:8]}"
    draft = _draft_for_test(cluster_id=cluster_id, members=["t-1", "t-2"])

    try:
        outcomes_run1 = await dedup_drafts(
            [draft],
            tracker=linear_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run1[0].action == "created"
        created = outcomes_run1[0].created_issue
        assert created is not None
        assert created.key is not None

        outcomes_run2 = await dedup_drafts(
            [draft],
            tracker=linear_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run2[0].action == "skipped"
        assert outcomes_run2[0].existing_issue is not None
        assert outcomes_run2[0].existing_issue.key == created.key

        draft_grown = _draft_for_test(
            cluster_id=cluster_id,
            members=["t-1", "t-2", "t-3"],
        )
        outcomes_run3 = await dedup_drafts(
            [draft_grown],
            tracker=linear_adapter,
            auto_post_threshold="low",
        )
        assert outcomes_run3[0].action == "commented"
        assert outcomes_run3[0].new_member_trace_ids == ["t-3"]
    finally:
        await linear_adapter.close()

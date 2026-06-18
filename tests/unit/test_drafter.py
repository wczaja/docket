"""Unit tests for the Drafter subagent with a MockProvider.

We verify: title/body come from the LLM, provenance + labels are appended
correctly, files land in the queue directory, malformed LLM output raises.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from docket.agent.subagents.drafter import draft_issues
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.cluster import Cluster, ClusterStats
from docket.rubric.spec import Detection, Mode, Rubric, RubricMetadata, TriageConfig


class _MockProvider(ModelProvider):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.model = "mock-draft"
        self._responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        if not self._responses:
            msg = "no more queued responses"
            raise RuntimeError(msg)
        return self._responses.pop(0)


def _rubric_with_mode(*, triage: TriageConfig | None = None) -> Rubric:
    return Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="testbench", version="0.1.0"),
        modes=[
            Mode(
                id="hallucination",
                name="Hallucination",
                severity="critical",
                description="Agent asserted a fact unsupported by retrieval.",
                detection=Detection(type="llm_judge", prompt="ok", output_schema={}),
            )
        ],
        triage=triage,
    )


def _cluster(cluster_id: str = "c-1") -> Cluster:
    return Cluster(
        cluster_id=cluster_id,
        mode_id="hallucination",
        severity="critical",
        member_trace_ids=["t-1", "t-2", "t-3"],
        representative_trace_id="t-1",
        representative_excerpt="The capital of France is Tokyo.",
        stats=ClusterStats(size=3, mean_confidence=0.91),
    )


async def test_draft_one_cluster_writes_two_files(tmp_path: Path) -> None:
    provider = _MockProvider([{"title": "Hallucinated geographic facts", "body": "Body of issue."}])
    rubric = _rubric_with_mode()
    drafts = await draft_issues(
        [_cluster("c-abc")],
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.title == "Hallucinated geographic facts"
    assert "docket:provenance" in draft.body
    assert "docket" in draft.labels
    assert "mode:hallucination" in draft.labels
    assert "rubric:testbench@0.1.0" in draft.labels

    json_path = tmp_path / "c-abc.json"
    md_path = tmp_path / "c-abc.md"
    assert json_path.exists()
    assert md_path.exists()
    restored = json.loads(json_path.read_text())
    assert restored["title"] == draft.title
    assert restored["body"] == draft.body
    md = md_path.read_text()
    assert "# Hallucinated geographic facts" in md
    assert "`t-2`" in md


async def test_draft_skips_clusters_whose_mode_dropped_from_rubric(tmp_path: Path) -> None:
    provider = _MockProvider([])  # if drafter calls it for the ghost mode, fail
    rubric = _rubric_with_mode()
    drafts = await draft_issues(
        [
            Cluster(
                cluster_id="ghost",
                mode_id="not-in-rubric",
                severity="low",
                member_trace_ids=["t-1"],
                representative_trace_id="t-1",
                stats=ClusterStats(size=1),
            )
        ],
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    assert drafts == []
    assert provider.calls == []


async def test_draft_passes_mode_context_to_llm(tmp_path: Path) -> None:
    provider = _MockProvider([{"title": "T", "body": "B" * 50}])
    rubric = _rubric_with_mode()
    await draft_issues(
        [_cluster()],
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    _system, user_prompt, _schema = provider.calls[0]
    assert "hallucination" in user_prompt
    assert "Hallucination" in user_prompt
    assert "Agent asserted a fact" in user_prompt
    assert "The capital of France is Tokyo." in user_prompt


async def test_draft_priority_comes_from_rubric_severity_mapping(tmp_path: Path) -> None:
    """`triage.default_severity_to_tracker` maps the cluster's severity to a
    tracker priority on the draft."""
    provider = _MockProvider([{"title": "T-title", "body": "B" * 50}])
    rubric = _rubric_with_mode(
        triage=TriageConfig(
            default_severity_to_tracker={
                "critical": "P1",
                "high": "P2",
                "medium": "P3",
                "low": "P4",
            }
        )
    )
    drafts = await draft_issues(
        [_cluster()],  # cluster severity is "critical"
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    assert drafts[0].priority == "P1"


async def test_draft_priority_is_none_without_rubric_mapping(tmp_path: Path) -> None:
    provider = _MockProvider([{"title": "T-title", "body": "B" * 50}])
    rubric = _rubric_with_mode()  # no triage block at all
    drafts = await draft_issues(
        [_cluster()],
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    assert drafts[0].priority is None


async def test_draft_priority_is_none_when_severity_unmapped(tmp_path: Path) -> None:
    provider = _MockProvider([{"title": "T-title", "body": "B" * 50}])
    rubric = _rubric_with_mode(
        triage=TriageConfig(default_severity_to_tracker={"low": "P4"}),
    )
    drafts = await draft_issues(
        [_cluster()],  # severity "critical" has no mapping entry
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=tmp_path,
    )
    assert drafts[0].priority is None


async def test_draft_raises_when_llm_returns_invalid_shape(tmp_path: Path) -> None:
    provider = _MockProvider([{"not_title": 1, "not_body": 2}])
    rubric = _rubric_with_mode()
    with pytest.raises(DetectionError, match="did not return title/body"):
        await draft_issues(
            [_cluster()],
            rubric=rubric,
            llm_provider=provider,
            run_id="r-1",
            output_dir=tmp_path,
        )


async def test_draft_creates_output_dir_if_missing(tmp_path: Path) -> None:
    queue = tmp_path / "subdir" / "queued"
    provider = _MockProvider([{"title": "T", "body": "B" * 50}])
    rubric = _rubric_with_mode()
    await draft_issues(
        [_cluster("nested")],
        rubric=rubric,
        llm_provider=provider,
        run_id="r-1",
        output_dir=queue,
    )
    assert (queue / "nested.json").exists()

import json
from pathlib import Path

from agent_triage.agent.evals import emit_eval_cases
from agent_triage.models.cluster import Cluster, ClusterStats
from agent_triage.models.eval_case import EVAL_CASE_SCHEMA, EvalCase
from agent_triage.rubric.loader import load_rubric

FIXTURES = Path(__file__).parent / "fixtures" / "rubrics"


def _rubric():  # type: ignore[no-untyped-def]
    return load_rubric(FIXTURES / "valid_minimal.yaml")


def _cluster(
    cluster_id: str = "abc123",
    mode_id: str | None = None,
    excerpt: str | None = "agent said the wrong thing",
) -> Cluster:
    rubric = _rubric()
    mode = rubric.modes[0]
    return Cluster(
        cluster_id=cluster_id,
        mode_id=mode_id or mode.id,
        severity=mode.severity,
        member_trace_ids=["t-1", "t-2", "t-3"],
        representative_trace_id="t-2",
        representative_excerpt=excerpt,
        stats=ClusterStats(size=3, mean_confidence=0.9),
    )


def test_emit_writes_one_file_per_cluster(tmp_path: Path) -> None:
    rubric = _rubric()
    clusters = [_cluster("c1"), _cluster("c2")]
    paths = emit_eval_cases(clusters, rubric=rubric, run_id="run-1", output_dir=tmp_path)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    assert paths[0].name == f"{clusters[0].mode_id}--c1.json"


def test_emitted_case_round_trips_and_carries_provenance(tmp_path: Path) -> None:
    rubric = _rubric()
    cluster = _cluster()
    (path,) = emit_eval_cases([cluster], rubric=rubric, run_id="run-9", output_dir=tmp_path)
    record = json.loads(path.read_text())
    assert record["schema"] == EVAL_CASE_SCHEMA
    assert record["expected"] == "positive"
    assert record["mode_id"] == cluster.mode_id
    assert record["run_id"] == "run-9"
    assert record["member_trace_ids"] == ["t-1", "t-2", "t-3"]
    assert record["cluster_size"] == 3
    assert record["rubric"] == f"{rubric.metadata.name}@{rubric.metadata.version}"
    # Round-trips through the model.
    case = EvalCase.model_validate(record)
    assert case.representative_trace_id == "t-2"


def test_emit_redacts_excerpts(tmp_path: Path) -> None:
    rubric = _rubric()
    cluster = _cluster(excerpt="reach me at user@example.com for details")
    (path,) = emit_eval_cases([cluster], rubric=rubric, run_id="r", output_dir=tmp_path)
    record = json.loads(path.read_text())
    assert "user@example.com" not in record["representative_excerpt"]


def test_emit_is_idempotent(tmp_path: Path) -> None:
    rubric = _rubric()
    cluster = _cluster()
    first = emit_eval_cases([cluster], rubric=rubric, run_id="r", output_dir=tmp_path)
    second = emit_eval_cases([cluster], rubric=rubric, run_id="r", output_dir=tmp_path)
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_emit_empty_clusters_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "sub"
    assert emit_eval_cases([], rubric=_rubric(), run_id="r", output_dir=out) == []

"""Acceptance tests for the agents/v1 built-in rubric.

These verify Phase 1 §7 acceptance criterion #1 ("all five built-in detection
types parse and validate") and #2 (`docket validate <agents/v1>` exits 0).
"""

from docket.rubric.loader import load_rubric
from docket.rubric.spec import DetectionType
from docket.rubric.validator import validate_rubric_yaml

AGENTS_V1 = "docket.dev/builtin/agents/v1"


def test_agents_v1_passes_json_schema() -> None:
    validate_rubric_yaml(AGENTS_V1)


def test_agents_v1_loads_pydantic() -> None:
    rubric = load_rubric(AGENTS_V1)
    assert rubric.metadata.name == "agents-builtin"
    assert rubric.metadata.version == "1.0.0"
    assert len(rubric.modes) >= 5


def test_agents_v1_covers_all_detection_types() -> None:
    rubric = load_rubric(AGENTS_V1)
    types: set[DetectionType] = {m.detection.type for m in rubric.modes}
    assert {"llm_judge", "regex", "tool_call", "metric_threshold", "composite"} <= types


def test_agents_v1_composite_mode_well_formed() -> None:
    rubric = load_rubric(AGENTS_V1)
    composite_modes = [m for m in rubric.modes if m.detection.type == "composite"]
    assert composite_modes, "expected at least one composite mode in agents/v1"
    for mode in composite_modes:
        assert mode.detection.operator in ("and", "or")
        assert mode.detection.operands is not None
        assert len(mode.detection.operands) >= 2


def test_agents_v1_has_clustering_and_triage_blocks() -> None:
    rubric = load_rubric(AGENTS_V1)
    assert rubric.clustering is not None
    assert rubric.triage is not None
    assert rubric.triage.auto_post_threshold == "never"

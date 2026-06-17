"""Gating Phase 0 test: Pydantic spec loads a valid rubric and rejects invalid ones."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_triage.rubric.spec import Rubric


def test_rubric_loads_valid_minimal(fixtures_dir: Path) -> None:
    data = yaml.safe_load((fixtures_dir / "rubrics" / "valid_minimal.yaml").read_text())
    rubric = Rubric.model_validate(data)
    assert rubric.metadata.name == "test-minimal"
    assert rubric.apiVersion == "agent-triage.dev/v1"
    assert len(rubric.modes) == 1
    assert rubric.modes[0].id == "test-mode"
    assert rubric.modes[0].severity == "high"
    assert rubric.modes[0].detection.type == "llm_judge"


def test_rubric_loads_with_all_optional_fields(fixtures_dir: Path) -> None:
    data = yaml.safe_load((fixtures_dir / "rubrics" / "valid_full.yaml").read_text())
    rubric = Rubric.model_validate(data)
    assert len(rubric.modes) == 2
    assert rubric.clustering is not None
    assert rubric.clustering.min_cluster_size == 3
    assert rubric.triage is not None
    assert rubric.triage.auto_post_threshold == "never"


def test_rubric_rejects_missing_apiversion(fixtures_dir: Path) -> None:
    data = yaml.safe_load(
        (fixtures_dir / "rubrics" / "invalid_missing_apiversion.yaml").read_text()
    )
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)


def test_rubric_rejects_bad_severity(fixtures_dir: Path) -> None:
    data = yaml.safe_load((fixtures_dir / "rubrics" / "invalid_bad_severity.yaml").read_text())
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)


def test_rubric_rejects_bad_id_pattern(fixtures_dir: Path) -> None:
    data = yaml.safe_load((fixtures_dir / "rubrics" / "invalid_bad_id.yaml").read_text())
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)


def test_rubric_rejects_unknown_field() -> None:
    data = {
        "apiVersion": "agent-triage.dev/v1",
        "kind": "Rubric",
        "metadata": {"name": "x", "version": "1"},
        "unknown_field": "should fail",
    }
    with pytest.raises(ValidationError):
        Rubric.model_validate(data)


def test_rubric_rejects_duplicate_mode_ids(fixtures_dir: Path) -> None:
    data = yaml.safe_load((fixtures_dir / "rubrics" / "invalid_duplicate_ids.yaml").read_text())
    with pytest.raises(ValidationError, match="Duplicate mode id"):
        Rubric.model_validate(data)


def test_composite_detection_accepts_logical_operator() -> None:
    data = {
        "apiVersion": "agent-triage.dev/v1",
        "kind": "Rubric",
        "metadata": {"name": "x", "version": "1"},
        "modes": [
            {
                "id": "composite-test",
                "severity": "high",
                "detection": {
                    "type": "composite",
                    "operator": "and",
                    "operands": [
                        {"type": "regex", "pattern": "foo"},
                        {"type": "regex", "pattern": "bar"},
                    ],
                },
            }
        ],
    }
    rubric = Rubric.model_validate(data)
    assert rubric.modes[0].detection.operator == "and"

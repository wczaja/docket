from pathlib import Path

import pytest

from agent_triage.errors import RubricValidationError
from agent_triage.rubric.validator import validate_rubric_yaml


def test_validate_valid_minimal(fixtures_dir: Path) -> None:
    validate_rubric_yaml(fixtures_dir / "rubrics" / "valid_minimal.yaml")


def test_validate_valid_full(fixtures_dir: Path) -> None:
    validate_rubric_yaml(fixtures_dir / "rubrics" / "valid_full.yaml")


def test_validate_rejects_bad_severity(fixtures_dir: Path) -> None:
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(fixtures_dir / "rubrics" / "invalid_bad_severity.yaml")


def test_validate_rejects_unknown_apiversion(tmp_path: Path) -> None:
    bad = tmp_path / "future.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v99\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes: []\n"
    )
    with pytest.raises(RubricValidationError, match="apiVersion"):
        validate_rubric_yaml(bad)


def test_validate_rejects_llm_judge_without_positive(tmp_path: Path) -> None:
    bad = tmp_path / "no_positive.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes:\n"
        "  - id: m\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: hi\n"
        "      output_schema:\n"
        "        type: object\n"
        "        required: [other]\n"
        "        properties:\n"
        "          other: {type: string}\n"
    )
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(bad)


def test_validate_rejects_regex_without_pattern(tmp_path: Path) -> None:
    bad = tmp_path / "no_pattern.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes:\n"
        "  - id: m\n"
        "    severity: low\n"
        "    detection:\n"
        "      type: regex\n"
    )
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(bad)


def test_validate_rejects_composite_without_operator(tmp_path: Path) -> None:
    bad = tmp_path / "no_op.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes:\n"
        "  - id: m\n"
        "    severity: low\n"
        "    detection:\n"
        "      type: composite\n"
        "      operands:\n"
        "        - {type: regex, pattern: a}\n"
        "        - {type: regex, pattern: b}\n"
    )
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(bad)


def test_validate_rejects_composite_with_comparison_operator(tmp_path: Path) -> None:
    bad = tmp_path / "wrong_op.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes:\n"
        "  - id: m\n"
        "    severity: low\n"
        "    detection:\n"
        "      type: composite\n"
        "      operator: '>'\n"
        "      operands:\n"
        "        - {type: regex, pattern: a}\n"
        "        - {type: regex, pattern: b}\n"
    )
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(bad)


def test_validate_rejects_metric_threshold_with_logical_operator(tmp_path: Path) -> None:
    bad = tmp_path / "wrong_op.yaml"
    bad.write_text(
        "apiVersion: agent-triage.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: x, version: '1'}\n"
        "modes:\n"
        "  - id: m\n"
        "    severity: low\n"
        "    detection:\n"
        "      type: metric_threshold\n"
        "      metric: latency_ms\n"
        "      threshold: 5000\n"
        "      operator: and\n"
    )
    with pytest.raises(RubricValidationError):
        validate_rubric_yaml(bad)


def test_validate_accepts_builtin_uri() -> None:
    validate_rubric_yaml("agent-triage.dev/builtin/agents/v1")


def test_validate_accepts_file_uri(fixtures_dir: Path) -> None:
    path = (fixtures_dir / "rubrics" / "valid_minimal.yaml").resolve()
    validate_rubric_yaml(f"file://{path}")

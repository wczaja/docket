from pathlib import Path

import pytest

from docket.errors import RubricError, RubricValidationError
from docket.rubric.loader import load_rubric


def test_load_valid_rubric(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "valid_minimal.yaml")
    assert rubric.metadata.name == "test-minimal"


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "nonexistent.yaml")


def test_load_unparseable_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: [unclosed")
    with pytest.raises(RubricError, match="parse"):
        load_rubric(bad)


def test_load_not_a_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n")
    with pytest.raises(RubricError, match="mapping"):
        load_rubric(bad)


def test_load_structurally_invalid(fixtures_dir: Path) -> None:
    with pytest.raises(RubricValidationError):
        load_rubric(fixtures_dir / "rubrics" / "invalid_bad_severity.yaml")


def test_load_valid_full_resolves_shared_import(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "valid_full.yaml")
    mode_ids = {m.id for m in rubric.modes}
    assert mode_ids == {"shared-timeout", "hallucinated-fact", "latency-blowup"}


_HEADER = "apiVersion: docket.dev/v1\nkind: Rubric\nmetadata: {name: x, version: '1'}\n"


def _write_rubric(path: Path, modes_yaml: str) -> Path:
    path.write_text(_HEADER + modes_yaml)
    return path


def test_load_rejects_llm_judge_missing_prompt(tmp_path: Path) -> None:
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: broken-judge\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      output_schema:\n"
        "        type: object\n"
        "        required: [positive]\n"
        "        properties:\n"
        "          positive: {type: boolean}\n",
    )
    with pytest.raises(RubricValidationError, match="broken-judge.*requires `prompt`"):
        load_rubric(bad)


def test_load_rejects_llm_judge_missing_output_schema(tmp_path: Path) -> None:
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: broken-judge\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: Is this broken?\n",
    )
    with pytest.raises(RubricValidationError, match="broken-judge.*`output_schema`"):
        load_rubric(bad)


def test_load_rejects_malformed_json_schema(tmp_path: Path) -> None:
    """`minimum: "not-a-number"` is not a valid JSON Schema and must fail at
    load time, not per-trace at classification time."""
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: malformed-schema\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: Is this broken?\n"
        "      output_schema:\n"
        "        type: object\n"
        "        required: [positive]\n"
        "        properties:\n"
        "          positive: {type: boolean}\n"
        "          confidence: {type: number, minimum: not-a-number}\n",
    )
    with pytest.raises(RubricValidationError, match="not a valid JSON Schema"):
        load_rubric(bad)


def test_load_rejects_output_schema_without_object_root(tmp_path: Path) -> None:
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: array-root\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: Is this broken?\n"
        "      output_schema:\n"
        "        type: array\n",
    )
    with pytest.raises(RubricValidationError, match="type: object"):
        load_rubric(bad)


def test_load_rejects_output_schema_without_boolean_positive(tmp_path: Path) -> None:
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: no-positive\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: Is this broken?\n"
        "      output_schema:\n"
        "        type: object\n"
        "        required: [other]\n"
        "        properties:\n"
        "          other: {type: string}\n",
    )
    with pytest.raises(RubricValidationError, match="boolean `positive`"):
        load_rubric(bad)


def test_load_rejects_broken_llm_judge_inside_composite(tmp_path: Path) -> None:
    bad = _write_rubric(
        tmp_path / "bad.yaml",
        "modes:\n"
        "  - id: bad-composite\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: composite\n"
        "      operator: and\n"
        "      operands:\n"
        "        - {type: regex, pattern: a}\n"
        "        - {type: llm_judge}\n",
    )
    with pytest.raises(RubricValidationError, match="bad-composite.*requires `prompt`"):
        load_rubric(bad)


def test_load_rejects_rubric_importing_broken_llm_judge(tmp_path: Path) -> None:
    """A rubric whose IMPORT contains a broken llm_judge mode fails at load,
    and the error names the offending file."""
    _write_rubric(
        tmp_path / "broken.yaml",
        "modes:\n"
        "  - id: broken-judge\n"
        "    severity: high\n"
        "    detection:\n"
        "      type: llm_judge\n"
        "      prompt: Is this broken?\n",
    )
    importer = tmp_path / "importer.yaml"
    importer.write_text(_HEADER + "imports:\n  - file://./broken.yaml\n")
    with pytest.raises(RubricValidationError, match="broken-judge") as excinfo:
        load_rubric(importer)
    assert "broken.yaml" in str(excinfo.value)


def test_load_all_builtin_rubrics_pass_semantic_validation() -> None:
    for name in ("agents", "rag", "routing", "multi-agent", "mast"):
        rubric = load_rubric(f"docket.dev/builtin/{name}/v1")
        assert rubric.modes

"""Tests for recursive import resolution, merge semantics, and cycle detection."""

from pathlib import Path

import pytest

from docket.errors import RubricImportError, RubricValidationError
from docket.rubric.loader import load_rubric


def test_import_merges_modes_from_base(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "imports" / "importer.yaml")
    mode_ids = {m.id for m in rubric.modes}
    assert mode_ids == {"base-mode", "importer-mode"}


def test_own_modes_override_imported(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "imports" / "override.yaml")
    assert len(rubric.modes) == 1
    overridden = rubric.modes[0]
    assert overridden.id == "base-mode"
    assert overridden.severity == "critical"
    assert overridden.detection.pattern == "overridden"


def test_later_import_overrides_earlier(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "imports" / "multi_importer.yaml")
    assert len(rubric.modes) == 1
    shared = rubric.modes[0]
    assert shared.id == "shared"
    assert shared.detection.pattern == "from-b"
    assert shared.severity == "high"


def test_cycle_detection_pair(fixtures_dir: Path) -> None:
    with pytest.raises(RubricImportError, match="cycle"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "cycle_a.yaml")


def test_self_import_detected_as_cycle(fixtures_dir: Path) -> None:
    with pytest.raises(RubricImportError, match="cycle"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "self_import.yaml")


def test_missing_import_target(fixtures_dir: Path) -> None:
    with pytest.raises(RubricImportError, match="not found"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "imports_missing.yaml")


def test_unsupported_https_scheme(fixtures_dir: Path) -> None:
    with pytest.raises(RubricImportError, match="not supported in v1"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "imports_https.yaml")


def test_builtin_import(fixtures_dir: Path) -> None:
    rubric = load_rubric(fixtures_dir / "rubrics" / "imports" / "imports_builtin.yaml")
    mode_ids = {m.id for m in rubric.modes}
    assert "project-specific" in mode_ids
    assert "hallucination" in mode_ids


def test_empty_rubric_fails_post_merge(fixtures_dir: Path) -> None:
    with pytest.raises(RubricValidationError, match="no modes and no imports"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "empty.yaml")


def test_malformed_imports_list(fixtures_dir: Path) -> None:
    with pytest.raises(RubricValidationError, match="invalid `imports`"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "malformed_imports.yaml")


def test_unsupported_apiversion(fixtures_dir: Path) -> None:
    with pytest.raises(RubricValidationError, match="unsupported apiVersion"):
        load_rubric(fixtures_dir / "rubrics" / "imports" / "future_apiversion.yaml")


def test_loads_via_file_uri(fixtures_dir: Path) -> None:
    path = (fixtures_dir / "rubrics" / "imports" / "base.yaml").resolve()
    rubric = load_rubric(f"file://{path}")
    assert rubric.metadata.name == "base"


def test_loads_via_builtin_uri() -> None:
    rubric = load_rubric("docket.dev/builtin/agents/v1")
    assert rubric.metadata.name == "agents-builtin"


def test_imports_with_only_imports_produces_modes(tmp_path: Path, fixtures_dir: Path) -> None:
    importer = tmp_path / "importer.yaml"
    base_path = (fixtures_dir / "rubrics" / "imports" / "base.yaml").resolve()
    importer.write_text(
        "apiVersion: docket.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: only-imports, version: '1'}\n"
        f"imports:\n  - file://{base_path}\n"
    )
    rubric = load_rubric(importer)
    assert len(rubric.modes) == 1
    assert rubric.modes[0].id == "base-mode"


def test_imports_resolving_to_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text(
        "apiVersion: docket.dev/v1\nkind: Rubric\nmetadata: {name: empty, version: '1'}\n"
    )
    importer = tmp_path / "importer.yaml"
    importer.write_text(
        "apiVersion: docket.dev/v1\n"
        "kind: Rubric\n"
        "metadata: {name: importer-only, version: '1'}\n"
        "imports:\n  - file://./empty.yaml\n"
    )
    with pytest.raises(RubricValidationError):
        load_rubric(importer)

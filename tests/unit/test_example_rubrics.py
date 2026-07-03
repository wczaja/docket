"""Every rubric published under rubrics/ must stay valid.

These files are documentation: the README, docs/rubric-spec.md, and the
registry index point users at them as starting points, so a schema drift
that broke them would break the first thing a new user copies.

Registry rubrics carry an extra bar (enforced here, not just in
CONTRIBUTING.md): every `llm_judge` mode they declare ships at least one
positive and one negative example, so `docket self-test` is a real
regression suite for the judge prompts.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from docket.rubric.loader import load_rubric
from docket.rubric.validator import validate_rubric_yaml

RUBRICS_DIR = Path(__file__).resolve().parents[2] / "rubrics"
EXAMPLES_DIR = RUBRICS_DIR / "examples"
REGISTRY_DIR = RUBRICS_DIR / "registry"

EXAMPLE_RUBRICS = sorted(EXAMPLES_DIR.glob("*.yaml"))
REGISTRY_RUBRICS = sorted(REGISTRY_DIR.glob("*/v1/rubric.yaml"))

EXPECTED_REGISTRY_USE_CASES = {
    "support-agent",
    "rag-knowledge-assistant",
    "sql-analytics-agent",
    "coding-agent",
    "multi-agent-supervisor",
    "voice-ivr-agent",
}


def test_examples_directory_is_not_empty() -> None:
    assert EXAMPLE_RUBRICS, f"no example rubrics found under {EXAMPLES_DIR}"


def test_registry_ships_the_documented_use_cases() -> None:
    found = {p.parent.parent.name for p in REGISTRY_RUBRICS}
    assert found >= EXPECTED_REGISTRY_USE_CASES


@pytest.mark.parametrize(
    "path", EXAMPLE_RUBRICS + REGISTRY_RUBRICS, ids=lambda p: f"{p.parent.parent.name}/{p.name}"
)
def test_published_rubric_validates(path: Path) -> None:
    validate_rubric_yaml(path)


@pytest.mark.parametrize(
    "path", EXAMPLE_RUBRICS + REGISTRY_RUBRICS, ids=lambda p: f"{p.parent.parent.name}/{p.name}"
)
def test_published_rubric_loads_with_imports_resolved(path: Path) -> None:
    rubric = load_rubric(path)
    assert rubric.modes, "merged rubric must contain at least one mode"


@pytest.mark.parametrize("path", REGISTRY_RUBRICS, ids=lambda p: p.parent.parent.name)
def test_registry_judge_modes_are_exampled_both_ways(path: Path) -> None:
    """The registry quality gate: no unexampled judge mode ships.

    Applies to the rubric's OWN modes (imported builtins are graded by
    their own suites). Composite modes with a judge operand count too.
    """

    def _uses_judge(detection: dict[str, Any]) -> bool:
        if detection.get("type") == "llm_judge":
            return True
        return any(_uses_judge(op) for op in detection.get("operands") or [])

    raw = yaml.safe_load(path.read_text())
    for mode in raw.get("modes", []):
        if not _uses_judge(mode.get("detection", {})):
            continue
        examples = mode.get("examples") or []
        verdicts = {e.get("expected") for e in examples}
        assert {"positive", "negative"} <= verdicts, (
            f"{path.parent.parent.name}: judge mode {mode['id']!r} must ship at "
            "least one positive and one negative example"
        )


@pytest.mark.parametrize("path", REGISTRY_RUBRICS, ids=lambda p: p.parent.parent.name)
def test_registry_rubric_has_a_readme(path: Path) -> None:
    assert (path.parent.parent / "README.md").is_file()


def test_sample_support_agent_merges_builtin_modes() -> None:
    rubric = load_rubric(EXAMPLES_DIR / "sample-support-agent.yaml")
    mode_ids = {m.id for m in rubric.modes}
    # Own modes are present...
    assert "hallucinated-pricing" in mode_ids
    assert "refund-without-confirmation" in mode_ids
    # ...and so are modes merged in from the builtin agents/v1 import.
    assert "hallucination" in mode_ids


def test_supervisor_registry_rubric_merges_three_builtins() -> None:
    rubric = load_rubric(REGISTRY_DIR / "multi-agent-supervisor" / "v1" / "rubric.yaml")
    mode_ids = {m.id for m in rubric.modes}
    assert "silent-subagent-failure" in mode_ids  # own
    assert "step-repetition" in mode_ids  # mast/v1
    assert "oscillation" in mode_ids  # routing/v1
    assert "role-drift" in mode_ids  # multi-agent/v1

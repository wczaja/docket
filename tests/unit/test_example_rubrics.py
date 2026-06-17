"""Every rubric published under rubrics/examples/ must stay valid.

These files are documentation: the README and docs/rubric-spec.md point
users at them as starting points, so a schema drift that broke them would
break the first thing a new user copies.
"""

from pathlib import Path

import pytest

from agent_triage.rubric.loader import load_rubric
from agent_triage.rubric.validator import validate_rubric_yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "rubrics" / "examples"

EXAMPLE_RUBRICS = sorted(EXAMPLES_DIR.glob("*.yaml"))


def test_examples_directory_is_not_empty() -> None:
    assert EXAMPLE_RUBRICS, f"no example rubrics found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_RUBRICS, ids=lambda p: p.name)
def test_example_rubric_validates(path: Path) -> None:
    validate_rubric_yaml(path)


@pytest.mark.parametrize("path", EXAMPLE_RUBRICS, ids=lambda p: p.name)
def test_example_rubric_loads_with_imports_resolved(path: Path) -> None:
    rubric = load_rubric(path)
    assert rubric.modes, "merged rubric must contain at least one mode"


def test_sample_support_agent_merges_builtin_modes() -> None:
    rubric = load_rubric(EXAMPLES_DIR / "sample-support-agent.yaml")
    mode_ids = {m.id for m in rubric.modes}
    # Own modes are present...
    assert "hallucinated-pricing" in mode_ids
    assert "refund-without-confirmation" in mode_ids
    # ...and so are modes merged in from the builtin agents/v1 import.
    assert "hallucination" in mode_ids

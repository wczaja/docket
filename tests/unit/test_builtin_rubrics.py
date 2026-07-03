"""Deterministic checks for every builtin rubric.

Each of the five v1 builtin rubrics must:

  - Load successfully via `load_rubric(builtin_uri)`.
  - Pass the v1 schema validator.
  - Carry at least four modes (the §7 acceptance bar for "v1.0 quality").
  - Have unique mode IDs within the rubric.
  - Declare a clustering block (so the runtime knows how to embed +
    threshold clusters of positives).
  - Have valid severities on every mode.

The LLM-judged self-test of each mode's `examples` is gated on a real LLM
provider key and lives in `tests/integration/test_llm_judge_real.py`. This
test file is fully deterministic and runs on every PR.
"""

import pytest

from docket.rubric.loader import load_rubric
from docket.rubric.spec import Rubric
from docket.rubric.validator import validate_rubric_yaml

_BUILTIN_URIS = [
    "docket.dev/builtin/agents/v1",
    "docket.dev/builtin/rag/v1",
    "docket.dev/builtin/routing/v1",
    "docket.dev/builtin/multi-agent/v1",
    "docket.dev/builtin/mast/v1",
]

_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


@pytest.fixture(params=_BUILTIN_URIS)
def builtin_rubric(request: pytest.FixtureRequest) -> Rubric:
    return load_rubric(request.param)


def test_builtin_rubric_loads_and_has_metadata(builtin_rubric: Rubric) -> None:
    assert builtin_rubric.metadata.name
    assert builtin_rubric.metadata.version


def test_builtin_rubric_has_at_least_four_modes(builtin_rubric: Rubric) -> None:
    assert len(builtin_rubric.modes) >= 4


def test_builtin_rubric_mode_ids_are_unique(builtin_rubric: Rubric) -> None:
    ids = [m.id for m in builtin_rubric.modes]
    assert len(ids) == len(set(ids))


def test_builtin_rubric_severities_are_valid(builtin_rubric: Rubric) -> None:
    for mode in builtin_rubric.modes:
        assert mode.severity in _VALID_SEVERITIES, (
            f"mode {mode.id} has invalid severity {mode.severity!r}"
        )


def test_builtin_rubric_declares_clustering(builtin_rubric: Rubric) -> None:
    assert builtin_rubric.clustering is not None
    assert builtin_rubric.clustering.strategy
    assert builtin_rubric.clustering.embedding_model


def test_builtin_rubric_modes_have_detection(builtin_rubric: Rubric) -> None:
    for mode in builtin_rubric.modes:
        assert mode.detection is not None, f"mode {mode.id} has no detection"


def test_every_builtin_uri_is_resolvable() -> None:
    """Belt-and-braces: prove the five URIs the docs reference all resolve."""
    for uri in _BUILTIN_URIS:
        rubric = load_rubric(uri)
        assert rubric.metadata.version == "1.0.0"


@pytest.mark.parametrize("uri", _BUILTIN_URIS)
def test_builtin_rubric_passes_v1_schema_validator(uri: str) -> None:
    """The docstring promise: every builtin passes the v1 schema validator,
    not just agents/v1 (which test_builtin_agents.py covers separately)."""
    validate_rubric_yaml(uri)  # raises RubricValidationError on failure

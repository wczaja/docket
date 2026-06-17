"""The package version is single-sourced from distribution metadata."""

from importlib.metadata import version

import agent_triage


def test_dunder_version_matches_distribution_metadata() -> None:
    assert agent_triage.__version__ == version("agent-triage")


def test_version_is_a_release_version() -> None:
    # Guards against the uninstalled fallback leaking into a built wheel.
    major, _, _ = agent_triage.__version__.partition(".")
    assert major.isdigit()
    assert int(major) >= 1

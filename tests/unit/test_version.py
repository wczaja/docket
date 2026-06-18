"""The package version is single-sourced from distribution metadata."""

from importlib.metadata import version

import docket


def test_dunder_version_matches_distribution_metadata() -> None:
    assert docket.__version__ == version("docket")


def test_version_is_a_release_version() -> None:
    # Guards against the uninstalled fallback leaking into a built wheel.
    major, _, _ = docket.__version__.partition(".")
    assert major.isdigit()
    assert int(major) >= 1

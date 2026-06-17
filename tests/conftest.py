import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_agent_triage_logger_propagation() -> Iterator[None]:
    """The CLI's _configure_logging sets propagate=False on the agent_triage
    logger. That state leaks across tests within a pytest session and breaks
    caplog assertions in unrelated tests. Restore propagation around every
    test.
    """
    logger = logging.getLogger("agent_triage")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "unit" / "fixtures"


@pytest.fixture
def traces_dir() -> Path:
    return Path(__file__).parent / "integration" / "fixtures" / "traces"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit external LLM APIs and require credentials.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--run-integration"):
        skip_default = pytest.mark.skip(reason="use --run-integration to opt in")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_default)
        return
    if "ANTHROPIC_API_KEY" not in os.environ:
        skip_no_key = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set in env")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_no_key)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: hits external LLM APIs; requires credentials and --run-integration",
    )

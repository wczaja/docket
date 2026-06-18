from pathlib import Path

import pytest

from docket.config import Config
from docket.errors import ConfigError


def test_config_loads_minimal(tmp_path: Path) -> None:
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend:\n"
        "  type: mcp\n"
        "  command: docket-adapter-phoenix\n"
        "  env:\n"
        "    PHOENIX_URL: http://localhost:6006\n"
        "rubric: docket.dev/builtin/agents/v1\n"
    )
    cfg = Config.from_yaml(cfg_path)
    assert cfg.trace_backend.command == "docket-adapter-phoenix"
    assert cfg.trace_backend.env["PHOENIX_URL"] == "http://localhost:6006"
    assert cfg.max_traces_per_run == 1000
    assert cfg.auto_post_threshold == "never"


def test_config_loads_full(tmp_path: Path) -> None:
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend:\n"
        "  type: mcp\n"
        "  command: foo\n"
        "tracker:\n"
        "  type: mcp\n"
        "  command: bar\n"
        "rubric: x\n"
        "max_traces_per_run: 500\n"
        "auto_post_threshold: high\n"
    )
    cfg = Config.from_yaml(cfg_path)
    assert cfg.max_traces_per_run == 500
    assert cfg.auto_post_threshold == "high"
    assert cfg.tracker is not None
    assert cfg.tracker.command == "bar"


def test_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "missing.yaml")


def test_config_invalid_threshold(tmp_path: Path) -> None:
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend: {type: mcp, command: foo}\nrubric: x\nauto_post_threshold: blocker\n"
    )
    with pytest.raises(ConfigError, match="validation"):
        Config.from_yaml(cfg_path)


def test_config_negative_traces(tmp_path: Path) -> None:
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend: {type: mcp, command: foo}\nrubric: x\nmax_traces_per_run: -1\n"
    )
    with pytest.raises(ConfigError):
        Config.from_yaml(cfg_path)


def test_config_not_a_mapping(tmp_path: Path) -> None:
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text("- one\n- two\n")
    with pytest.raises(ConfigError, match="mapping"):
        Config.from_yaml(cfg_path)


def test_config_expands_env_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AT_TEST_TOKEN", "tok-123")
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend:\n"
        "  type: mcp\n"
        "  command: foo\n"
        "tracker:\n"
        "  type: mcp\n"
        "  command: bar\n"
        "  env:\n"
        "    GITHUB_TOKEN: ${AT_TEST_TOKEN}\n"
        "    GITHUB_OWNER: my-org\n"
        "rubric: x\n"
    )
    cfg = Config.from_yaml(cfg_path)
    assert cfg.tracker is not None
    assert cfg.tracker.env["GITHUB_TOKEN"] == "tok-123"  # noqa: S105 — synthetic test value
    assert cfg.tracker.env["GITHUB_OWNER"] == "my-org"


def test_config_expands_env_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AT_TEST_HOST", "phoenix.internal")
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend:\n"
        "  type: mcp\n"
        "  command: foo\n"
        "  env:\n"
        "    PHOENIX_URL: http://${AT_TEST_HOST}:6006\n"
        "rubric: x\n"
    )
    cfg = Config.from_yaml(cfg_path)
    assert cfg.trace_backend.env["PHOENIX_URL"] == "http://phoenix.internal:6006"


def test_config_unset_env_reference_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AT_TEST_MISSING", raising=False)
    cfg_path = tmp_path / "docket.yaml"
    cfg_path.write_text(
        "trace_backend:\n"
        "  type: mcp\n"
        "  command: foo\n"
        "  env:\n"
        "    TOKEN: ${AT_TEST_MISSING}\n"
        "rubric: x\n"
    )
    with pytest.raises(ConfigError, match="AT_TEST_MISSING"):
        Config.from_yaml(cfg_path)

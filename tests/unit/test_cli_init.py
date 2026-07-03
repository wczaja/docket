"""Tests for `docket init` (the interactive config scaffolder)."""

from pathlib import Path

from click.testing import CliRunner

from docket.cli import main
from docket.config import Config


def test_init_all_defaults_yields_loadable_phoenix_config(tmp_path: Path) -> None:
    """Pressing Enter through every prompt produces a config the real loader
    accepts (the default flow has no ${VAR} references to resolve)."""
    runner = CliRunner()
    out = tmp_path / "docket.yaml"
    result = runner.invoke(main, ["init", "--path", str(out)], input="\n\n\n\n\n")
    assert result.exit_code == 0, result.output

    config = Config.from_yaml(out)
    assert config.trace_backend.command == "docket-adapter-phoenix"
    assert config.trace_backend.env["PHOENIX_URL"] == "http://localhost:6006"
    assert config.tracker is None
    assert config.rubric == "docket.dev/builtin/agents/v1"
    assert config.auto_post_threshold == "never"
    assert "docket run --config" in result.output


def test_init_github_tracker_flow(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "docket.yaml"
    answers = "\n".join(
        [
            "",  # backend: phoenix
            "",  # phoenix url default
            "github",  # tracker
            "acme",  # owner
            "agent-issues",  # repo
            "docket.dev/builtin/rag/v1",  # rubric
            "high",  # auto-post threshold
        ]
    )
    result = runner.invoke(main, ["init", "--path", str(out)], input=answers + "\n")
    assert result.exit_code == 0, result.output

    content = out.read_text()
    assert "docket-adapter-github" in content
    assert "GITHUB_TOKEN: ${GITHUB_TOKEN}" in content
    assert "GITHUB_OWNER: acme" in content
    assert "rubric: docket.dev/builtin/rag/v1" in content
    assert "auto_post_threshold: high" in content
    # Secrets stay as env references, and the output says which to export.
    assert "GITHUB_TOKEN" in result.output


def test_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "docket.yaml"
    out.write_text("keep me")
    result = runner.invoke(main, ["init", "--path", str(out)], input="\n\n\n\n\n")
    assert result.exit_code == 1
    assert "--force" in result.output
    assert out.read_text() == "keep me"

    forced = runner.invoke(main, ["init", "--path", str(out), "--force"], input="\n\n\n\n\n")
    assert forced.exit_code == 0, forced.output
    assert "trace_backend" in out.read_text()


def test_init_rejects_unloadable_rubric(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "docket.yaml"
    answers = "\n".join(["", "", "none", str(tmp_path / "missing.yaml")])
    result = runner.invoke(main, ["init", "--path", str(out)], input=answers + "\n")
    assert result.exit_code == 1
    assert "does not load" in result.output
    assert not out.exists()

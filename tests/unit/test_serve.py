"""Tests for the `agent-triage serve` daemon command."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from agent_triage.agent.triage import TriageResult
from agent_triage.cli import main
from agent_triage.errors import BackendError
from agent_triage.models.report import ModeStats, RunReport


class _NoopBackend:
    async def close(self) -> None:
        return None


def _fake_result(run_id: str = "tick") -> TriageResult:
    report = RunReport(
        run_id=run_id,
        rubric_name="agents-builtin",
        rubric_version="1.0.0",
        since=datetime(2026, 6, 1, tzinfo=UTC),
        until=datetime(2026, 6, 1, 1, 0, tzinfo=UTC),
        started_at=datetime(2026, 6, 1, tzinfo=UTC),
        finished_at=datetime(2026, 6, 1, 0, 0, 5, tzinfo=UTC),
        trace_count=0,
        mode_stats=[ModeStats(mode_id="hallucination", severity="critical")],
    )
    return TriageResult(
        run_report=report,
        clusters=[],
        drafts=[],
        report_markdown=f"# serve tick `{run_id}`\n",
    )


def _serve_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "serve",
        "--backend",
        "phoenix",
        "--phoenix-url",
        "http://test:6006",
        "--rubric",
        "agent-triage.dev/builtin/agents/v1",
        "--config",
        str(tmp_path / "missing.yaml"),
        "--interval",
        "1h",
        *extra,
    ]


def test_serve_ticks_tile_windows_exactly(tmp_path: Path) -> None:
    """Consecutive successful ticks share a boundary: tick 2's since == tick 1's until."""
    calls: list[dict[str, Any]] = []

    async def fake_pipeline(**kwargs: Any) -> TriageResult:
        calls.append(kwargs)
        return _fake_result(f"tick{len(calls)}")

    runner = CliRunner()
    with (
        patch("agent_triage.cli.build_backend", return_value=_NoopBackend()),
        patch("agent_triage.cli.run_triage_pipeline", side_effect=fake_pipeline),
        patch("agent_triage.cli.build_provider"),
        patch("agent_triage.cli.time.sleep") as fake_sleep,
    ):
        result = runner.invoke(main, _serve_args(tmp_path, "--max-ticks", "2"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert calls[1]["since"] == calls[0]["until"]
    assert calls[0]["until"] < calls[1]["until"]
    # Both tick reports were printed.
    assert "serve tick `tick1`" in result.output
    assert "serve tick `tick2`" in result.output
    # The daemon slept between tick 1 and tick 2, not after the last tick.
    assert fake_sleep.call_count <= 1


def test_serve_failed_tick_does_not_advance_window(tmp_path: Path) -> None:
    """A failed tick keeps its window; the next tick retries the union."""
    calls: list[dict[str, Any]] = []

    async def flaky_pipeline(**kwargs: Any) -> TriageResult:
        calls.append(kwargs)
        if len(calls) == 1:
            raise BackendError("transient backend failure")
        return _fake_result("recovered")

    runner = CliRunner()
    with (
        patch("agent_triage.cli.build_backend", return_value=_NoopBackend()),
        patch("agent_triage.cli.run_triage_pipeline", side_effect=flaky_pipeline),
        patch("agent_triage.cli.build_provider"),
        patch("agent_triage.cli.time.sleep"),
    ):
        result = runner.invoke(main, _serve_args(tmp_path, "--max-ticks", "2"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    # Window start did not advance after the failure: the retry covers the
    # failed window plus the time elapsed since.
    assert calls[1]["since"] == calls[0]["since"]
    assert calls[1]["until"] >= calls[0]["until"]
    assert "serve tick `recovered`" in result.output


def test_serve_respects_max_ticks(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_pipeline(**kwargs: Any) -> TriageResult:
        calls.append(kwargs)
        return _fake_result()

    runner = CliRunner()
    with (
        patch("agent_triage.cli.build_backend", return_value=_NoopBackend()),
        patch("agent_triage.cli.run_triage_pipeline", side_effect=fake_pipeline),
        patch("agent_triage.cli.build_provider"),
        patch("agent_triage.cli.time.sleep"),
    ):
        result = runner.invoke(main, _serve_args(tmp_path, "--max-ticks", "3"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 3


def test_serve_exits_on_config_error(tmp_path: Path) -> None:
    """Missing backend configuration is permanent: exit 1, no retry loop."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--rubric",
            "agent-triage.dev/builtin/agents/v1",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--max-ticks",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_serve_rejects_bad_interval(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, _serve_args(tmp_path)[:-2] + ["--interval", "soon"])
    assert result.exit_code == 2
    assert "duration" in result.output.lower()


def test_serve_first_window_spans_one_interval(tmp_path: Path) -> None:
    """The first tick covers the trailing --interval, like `run --since`."""
    calls: list[dict[str, Any]] = []

    async def fake_pipeline(**kwargs: Any) -> TriageResult:
        calls.append(kwargs)
        return _fake_result()

    runner = CliRunner()
    with (
        patch("agent_triage.cli.build_backend", return_value=_NoopBackend()),
        patch("agent_triage.cli.run_triage_pipeline", side_effect=fake_pipeline),
        patch("agent_triage.cli.build_provider"),
    ):
        result = runner.invoke(main, _serve_args(tmp_path, "--max-ticks", "1"))

    assert result.exit_code == 0, result.output
    (call,) = calls
    width = (call["until"] - call["since"]).total_seconds()
    assert 3590 <= width <= 3670  # one hour, give or take scheduling slack

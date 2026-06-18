from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from docket.cli import main
from docket.self_test import SelfTestResult


def test_help_works() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "docket" in result.output


def test_version_works() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


def test_validate_valid_rubric(fixtures_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(fixtures_dir / "rubrics" / "valid_minimal.yaml")])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_validate_invalid_rubric(fixtures_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["validate", str(fixtures_dir / "rubrics" / "invalid_bad_severity.yaml")],
    )
    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_validate_nonexistent_file(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


def test_validate_builtin_uri() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "docket.dev/builtin/agents/v1"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_unknown_builtin() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "docket.dev/builtin/nonexistent/v1"])
    assert result.exit_code == 1
    assert "Unknown builtin" in result.output


def test_validate_file_uri(fixtures_dir: Path) -> None:
    path = (fixtures_dir / "rubrics" / "valid_minimal.yaml").resolve()
    runner = CliRunner()
    result = runner.invoke(main, ["validate", f"file://{path}"])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_self_test_command_passes(fixtures_dir: Path) -> None:
    async def fake_run_self_test(*_args: object, **_kwargs: object) -> list[SelfTestResult]:
        return [
            SelfTestResult(
                mode_id="hallucination",
                example_index=0,
                passed=True,
                skipped=False,
                message="expected positive, got positive",
            ),
        ]

    runner = CliRunner()
    with (
        patch("docket.cli.run_self_test", side_effect=fake_run_self_test),
        patch("docket.cli.build_provider"),
    ):
        result = runner.invoke(
            main,
            ["self-test", str(fixtures_dir / "rubrics" / "valid_minimal.yaml")],
        )
    assert result.exit_code == 0, result.output
    assert "PASS hallucination[0]" in result.output
    assert "1 passed" in result.output


def test_self_test_command_reports_failures(fixtures_dir: Path) -> None:
    async def fake_run_self_test(*_args: object, **_kwargs: object) -> list[SelfTestResult]:
        return [
            SelfTestResult(
                mode_id="m",
                example_index=0,
                passed=False,
                skipped=False,
                message="expected positive, got negative",
            ),
        ]

    runner = CliRunner()
    with (
        patch("docket.cli.run_self_test", side_effect=fake_run_self_test),
        patch("docket.cli.build_provider"),
    ):
        result = runner.invoke(
            main,
            ["self-test", str(fixtures_dir / "rubrics" / "valid_minimal.yaml")],
        )
    assert result.exit_code == 1
    assert "FAIL m[0]" in result.output
    assert "1 failed" in result.output


def test_self_test_command_skips_non_llm_judge(fixtures_dir: Path) -> None:
    async def fake_run_self_test(*_args: object, **_kwargs: object) -> list[SelfTestResult]:
        return [
            SelfTestResult(
                mode_id="regex-mode",
                example_index=-1,
                passed=True,
                skipped=True,
                message="skipped: ...",
            ),
        ]

    runner = CliRunner()
    with (
        patch("docket.cli.run_self_test", side_effect=fake_run_self_test),
        patch("docket.cli.build_provider"),
    ):
        result = runner.invoke(
            main,
            ["self-test", str(fixtures_dir / "rubrics" / "valid_minimal.yaml")],
        )
    assert result.exit_code == 0
    assert "SKIP regex-mode" in result.output
    assert "1 skipped" in result.output


def test_self_test_command_rejects_invalid_rubric(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a: rubric")
    runner = CliRunner()
    result = runner.invoke(main, ["self-test", str(bad)])
    assert result.exit_code == 1
    assert "INVALID" in result.output


def test_run_rejects_missing_backend(tmp_path: Path) -> None:
    """No --backend and no config file = error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--rubric",
            "docket.dev/builtin/agents/v1",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "No trace backend specified" in result.output


def test_run_rejects_missing_phoenix_url(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--backend",
            "phoenix",
            "--rubric",
            "docket.dev/builtin/agents/v1",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "PHOENIX_URL" in result.output


def test_run_executes_with_mocked_dependencies(tmp_path: Path) -> None:
    """End-to-end CLI invocation with build_backend and run_triage_pipeline patched."""
    from datetime import UTC, datetime

    from docket.agent.triage import TriageResult
    from docket.models.report import ModeStats, RunReport

    fake_report = RunReport(
        run_id="abc123",
        rubric_name="agents-builtin",
        rubric_version="1.0.0",
        since=datetime(2026, 5, 22, tzinfo=UTC),
        until=datetime(2026, 5, 22, 1, 0, tzinfo=UTC),
        started_at=datetime(2026, 5, 22, tzinfo=UTC),
        finished_at=datetime(2026, 5, 22, 0, 0, 5, tzinfo=UTC),
        trace_count=0,
        mode_stats=[ModeStats(mode_id="hallucination", severity="critical")],
    )
    fake_result = TriageResult(
        run_report=fake_report,
        clusters=[],
        drafts=[],
        report_markdown=(
            "# docket run `abc123`\n\n"
            "- **Traces processed**: 0\n"
            "- **Clusters formed**: 0\n"
            "- **Issues drafted**: 0\n\n"
            "## Frequency by mode\n\n"
            "| mode | severity | positive | negative | error |\n"
            "| --- | --- | ---: | ---: | ---: |\n"
            "| `hallucination` | critical | 0 | 0 | 0 |\n"
        ),
    )

    class _NoopBackend:
        async def close(self) -> None:
            return None

    async def fake_run(*_args: object, **_kwargs: object) -> TriageResult:
        return fake_result

    runner = CliRunner()
    with (
        patch("docket.cli.build_backend", return_value=_NoopBackend()),
        patch("docket.cli.run_triage_pipeline", side_effect=fake_run),
        patch("docket.cli.build_provider"),
    ):
        result = runner.invoke(
            main,
            [
                "run",
                "--backend",
                "phoenix",
                "--phoenix-url",
                "http://test:6006",
                "--rubric",
                "docket.dev/builtin/agents/v1",
                "--config",
                str(tmp_path / "missing.yaml"),
                "--since",
                "1h",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "docket run `abc123`" in result.output
    assert "hallucination" in result.output


def test_run_passes_run_id_override(tmp_path: Path) -> None:
    """--run-id flag forwards to run_triage_pipeline."""
    from datetime import UTC, datetime

    from docket.agent.triage import TriageResult
    from docket.models.report import RunReport

    captured: dict[str, object] = {}

    class _NoopBackend:
        async def close(self) -> None:
            return None

    async def fake_run(**kwargs: object) -> TriageResult:
        captured.update(kwargs)
        report = RunReport(
            run_id="my-explicit-run-id",
            rubric_name="x",
            rubric_version="1",
            since=datetime(2026, 5, 22, tzinfo=UTC),
            until=datetime(2026, 5, 22, 1, 0, tzinfo=UTC),
            started_at=datetime(2026, 5, 22, tzinfo=UTC),
            finished_at=datetime(2026, 5, 22, 0, 0, 1, tzinfo=UTC),
            trace_count=0,
        )
        return TriageResult(
            run_report=report,
            clusters=[],
            drafts=[],
            report_markdown="# run my-explicit-run-id\n",
        )

    runner = CliRunner()
    with (
        patch("docket.cli.build_backend", return_value=_NoopBackend()),
        patch("docket.cli.run_triage_pipeline", side_effect=fake_run),
        patch("docket.cli.build_provider"),
    ):
        result = runner.invoke(
            main,
            [
                "run",
                "--backend",
                "phoenix",
                "--phoenix-url",
                "http://test:6006",
                "--rubric",
                "docket.dev/builtin/agents/v1",
                "--config",
                str(tmp_path / "missing.yaml"),
                "--since",
                "1h",
                "--run-id",
                "my-explicit-run-id",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured.get("run_id") == "my-explicit-run-id"


def test_run_rejects_bad_duration(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--backend",
            "phoenix",
            "--phoenix-url",
            "http://test",
            "--rubric",
            "docket.dev/builtin/agents/v1",
            "--since",
            "not-a-duration",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )
    assert result.exit_code != 0
    assert "duration" in result.output.lower()

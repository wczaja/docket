"""Smoke test for the classification benchmark (scripts/benchmark_classify.py).

Runs the script's default stub mode end to end, cost projection included, with
no API key or network. The trace count is shrunk to keep the suite fast; every
other argument is the parser default, so the script drifting from the
`docket.cost.estimate_cost` signature fails here instead of at the next manual
benchmark run.
"""

import pytest

from scripts.benchmark_classify import build_arg_parser, run_benchmark


async def test_default_stub_run_projects_cost_for_each_default_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_arg_parser().parse_args(["--traces", "20"])
    assert not args.live
    assert args.cost_models  # non-empty by default: the projection always runs

    await run_benchmark(args)

    out = capsys.readouterr().out
    assert "LLM calls (counted):" in out
    projections = [line for line in out.splitlines() if line.startswith("projected cost:")]
    assert len(projections) == len(args.cost_models)
    for line, model in zip(projections, args.cost_models, strict=True):
        assert f" on {model} (" in line

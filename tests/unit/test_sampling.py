"""Unit tests for `agent_triage.sampling`."""

import logging

import pytest

from agent_triage.sampling import sample_trace_ids


def test_uniform_sampling_returns_n_items() -> None:
    ids = [f"t-{i}" for i in range(100)]
    out = sample_trace_ids(ids, n=10, strategy="uniform", seed=42)
    assert len(out) == 10
    assert all(tid in ids for tid in out)


def test_uniform_sampling_preserves_input_order() -> None:
    ids = [f"t-{i:03d}" for i in range(100)]
    out = sample_trace_ids(ids, n=20, strategy="uniform", seed=42)
    sorted_out_indices = [ids.index(tid) for tid in out]
    assert sorted_out_indices == sorted(sorted_out_indices)


def test_uniform_sampling_is_deterministic_with_seed() -> None:
    ids = [f"t-{i}" for i in range(1000)]
    a = sample_trace_ids(ids, n=50, seed="run-id-abc")
    b = sample_trace_ids(ids, n=50, seed="run-id-abc")
    assert a == b


def test_uniform_sampling_differs_across_seeds() -> None:
    ids = [f"t-{i}" for i in range(1000)]
    a = sample_trace_ids(ids, n=50, seed=1)
    b = sample_trace_ids(ids, n=50, seed=2)
    assert a != b


def test_n_greater_than_population_returns_all() -> None:
    ids = ["a", "b", "c"]
    assert sample_trace_ids(ids, n=10, seed=1) == ids


def test_empty_input_returns_empty() -> None:
    assert sample_trace_ids([], n=10) == []


def test_n_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_trace_ids(["a"], n=0)


def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        sample_trace_ids(["a", "b", "c"], n=2, strategy="bogus")  # type: ignore[arg-type]


def test_stratified_falls_back_to_uniform_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ids = [f"t-{i}" for i in range(100)]
    with caplog.at_level(logging.WARNING, logger="agent_triage.sampling"):
        out = sample_trace_ids(ids, n=10, strategy="stratified", seed=1)
    assert len(out) == 10
    assert any("stratified" in r.message and "uniform" in r.message for r in caplog.records)


def test_errors_only_falls_back_to_uniform_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ids = [f"t-{i}" for i in range(100)]
    with caplog.at_level(logging.WARNING, logger="agent_triage.sampling"):
        out = sample_trace_ids(ids, n=10, strategy="errors-only", seed=1)
    assert len(out) == 10
    assert any("errors-only" in r.message for r in caplog.records)

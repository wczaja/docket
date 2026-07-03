"""Tests for the MAST-judge tuning harness (scripts/tune_mast_judges.py).

These run fully offline: the pure helpers (label/trace extraction, metrics) are
tested directly, and the end-to-end path is exercised with the deterministic
StubProvider so no API key, network, or MAD data is needed. They also pin the
mode-id -> MAST-code mapping to the actual mast/v1 rubric, so renaming a mode
without updating the harness fails loudly.
"""

import json
from pathlib import Path

import pytest

from docket.detectors import get_detector
from docket.models.trace import TraceLike
from docket.rubric.loader import load_rubric
from scripts.tune_mast_judges import (
    MAST_FM_BY_MODE_ID,
    MAST_NAME_BY_CODE,
    Counts,
    StubProvider,
    build_arg_parser,
    coerce_label,
    extract_gold_label,
    extract_trace_text,
    get_by_path,
    load_records,
    run,
    score_mode,
    select_modes,
    summarize,
)

RUBRIC_URI = "docket.dev/builtin/mast/v1"


def test_mapping_matches_rubric_exactly() -> None:
    rubric = load_rubric(RUBRIC_URI)
    rubric_ids = {m.id for m in rubric.modes}
    assert set(MAST_FM_BY_MODE_ID) == rubric_ids
    # Every mapped code has a canonical name.
    assert set(MAST_FM_BY_MODE_ID.values()) == set(MAST_NAME_BY_CODE)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("yes", True),
        ("No", False),
        ("TRUE", True),
        ("false", False),
        ("1", True),
        ("0", False),
        (2, None),
        (0.5, None),
        ("n/a", None),
        ("", None),
        (None, None),
        ([], None),
    ],
)
def test_coerce_label(value: object, expected: bool | None) -> None:
    assert coerce_label(value) is expected


def test_get_by_path_handles_codes_with_dots() -> None:
    rec = {"failure_modes": {"1.3": 1, "2.6": 0}}
    assert get_by_path(rec, "failure_modes/1.3") == 1
    assert get_by_path(rec, "failure_modes/2.6") == 0
    assert get_by_path(rec, "failure_modes/9.9") is None
    assert get_by_path(rec, "missing/1.3") is None


def test_extract_trace_text_default_and_explicit() -> None:
    assert extract_trace_text({"trace": "hello"}, None) == "hello"
    # Non-string values are JSON-encoded.
    assert extract_trace_text({"messages": [{"role": "user"}]}, None) == '[{"role": "user"}]'
    # Explicit field path wins.
    assert extract_trace_text({"a": {"b": "deep"}}, "a/b") == "deep"


def test_extract_trace_text_missing_raises() -> None:
    with pytest.raises(ValueError, match="trace text"):
        extract_trace_text({"unrelated": 1}, None)


def test_extract_gold_label_shapes() -> None:
    name = "Step Repetition"
    assert extract_gold_label({"failure_modes": {"1.3": 1}}, "1.3", name, None) is True
    assert extract_gold_label({"labels": {"1.3": "no"}}, "1.3", name, None) is False
    assert extract_gold_label({"1.3": True}, "1.3", name, None) is True
    assert extract_gold_label({"fm_1_3": 0}, "1.3", name, None) is False
    # Name-keyed labels.
    assert extract_gold_label({"failure_modes": {name: 1}}, "1.3", name, None) is True
    # Missing -> None (excluded from scoring).
    assert extract_gold_label({"other": 1}, "1.3", name, None) is None


def test_extract_gold_label_template() -> None:
    rec = {"annot": {"1.3": "yes"}}
    assert extract_gold_label(rec, "1.3", "Step Repetition", "annot/{code}") is True
    # Template that doesn't resolve -> None, even if a default shape would match.
    assert extract_gold_label({"1.3": 1}, "1.3", "x", "annot/{code}") is None


def test_counts_metrics() -> None:
    c = Counts(tp=3, fp=1, tn=5, fn=1)
    assert c.support == 10
    assert c.positives == 4
    assert c.precision == pytest.approx(3 / 4)
    assert c.recall == pytest.approx(3 / 4)
    assert c.f1 == pytest.approx(0.75)
    assert c.accuracy == pytest.approx(0.8)


def test_counts_zero_division_is_safe() -> None:
    c = Counts()
    assert c.precision == 0.0
    assert c.recall == 0.0
    assert c.f1 == 0.0
    assert c.accuracy == 0.0


def test_counts_add_classifies_quadrants() -> None:
    c = Counts()
    c.add(True, True)  # tp
    c.add(True, False)  # fp
    c.add(False, True)  # fn
    c.add(False, False)  # tn
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)


def test_summarize_micro_macro() -> None:
    from scripts.tune_mast_judges import ModeOutcome

    a = ModeOutcome(mode_id="a", code="1.3", counts=Counts(tp=1, fp=0, tn=1, fn=0))
    b = ModeOutcome(mode_id="b", code="1.4", counts=Counts(tp=0, fp=2, tn=0, fn=0))
    empty = ModeOutcome(mode_id="c", code="1.5")  # no support, excluded
    s = summarize([a, b, empty])
    # macro averages only over a and b.
    assert s["macro_precision"] == pytest.approx((1.0 + 0.0) / 2)
    # micro pools counts: tp=1, fp=2 -> precision 1/3.
    assert s["micro_precision"] == pytest.approx(1 / 3)


def test_select_modes_filters_and_subsets() -> None:
    rubric = load_rubric(RUBRIC_URI)
    assert len(select_modes(rubric, None)) == len(MAST_FM_BY_MODE_ID)
    subset = select_modes(rubric, "step-repetition, conversation-reset")
    assert {m.id for m in subset} == {"step-repetition", "conversation-reset"}


def test_select_modes_rejects_unknown_ids() -> None:
    rubric = load_rubric(RUBRIC_URI)
    with pytest.raises(SystemExit, match="step-repitition"):
        select_modes(rubric, "step-repetition,step-repitition")


def test_load_records_missing_file_is_a_clean_error(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(["--data", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit, match="Cannot read MAD data file"):
        load_records(args)


def test_load_records_invalid_json_is_a_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    args = build_arg_parser().parse_args(["--data", str(bad)])
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_records(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--data", "x.json", "--limit", "-1"],
        ["--data", "x.json", "--batch", "0"],
        ["--data", "x.json", "--inspect", "-2"],
    ],
)
def test_arg_parser_rejects_out_of_range_values(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(argv)
    assert "must be >=" in capsys.readouterr().err


async def test_score_mode_with_stub_detector() -> None:
    rubric = load_rubric(RUBRIC_URI)
    mode = next(m for m in rubric.modes if m.id == "step-repetition")
    detector = get_detector("llm_judge", llm_provider=StubProvider(), batch_size=1)
    labelled = [
        ("t1", TraceLike(full_text="repeated step <<FAIL>> here"), True),  # -> tp
        ("t2", TraceLike(full_text="clean run"), False),  # -> tn
        ("t3", TraceLike(full_text="surprise <<FAIL>>"), False),  # -> fp (disagreement)
    ]
    outcome = await score_mode(detector, mode, labelled, batch_size=1)
    assert (outcome.counts.tp, outcome.counts.tn, outcome.counts.fp, outcome.counts.fn) == (
        1,
        1,
        1,
        0,
    )
    assert outcome.errors == 0
    assert len(outcome.disagreements) == 1
    assert outcome.disagreements[0]["trace_id"] == "t3"


async def test_run_end_to_end_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = [
        {
            "id": "a",
            "trace": "the planner repeated <<FAIL>>",
            "failure_modes": {"1.3": 1, "1.4": 0},
        },
        {"id": "b", "trace": "clean conversation", "failure_modes": {"1.3": 0, "1.4": 0}},
    ]
    data_file = tmp_path / "mad.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")
    args = build_arg_parser().parse_args(
        [
            "--data",
            str(data_file),
            "--limit",
            "0",
            "--modes",
            "step-repetition,conversation-history-loss",
        ]
    )
    rc = await run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "step-repetition" in out
    assert RUBRIC_URI in out
    # Record "a" trips the stub marker for both modes: tp for 1.3 (gold 1), fp for 1.4 (gold 0).
    assert "micro" in out


async def test_run_reports_when_no_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_file = tmp_path / "mad.json"
    data_file.write_text(json.dumps([{"id": "a", "trace": "x"}]), encoding="utf-8")
    args = build_arg_parser().parse_args(["--data", str(data_file)])
    rc = await run(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "No gold labels found" in out


async def test_run_inspect_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_file = tmp_path / "mad.json"
    data_file.write_text(json.dumps([{"id": "a", "trace": "x", "failure_modes": {"1.3": 1}}]))
    args = build_arg_parser().parse_args(["--data", str(data_file), "--inspect", "1"])
    rc = await run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "record 0" in out
    assert "failure_modes" in out

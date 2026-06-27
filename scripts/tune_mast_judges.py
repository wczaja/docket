"""Tune docket's MAST judges against the MAD human-labelled dataset.

This harness scores the `docket.dev/builtin/mast/v1` rubric's `llm_judge`
detectors against real human labels from the MAD dataset (Cemri et al., "Why
Do Multi-Agent LLM Systems Fail?", arXiv:2503.13657), so you can measure
per-mode precision/recall/F1 and iterate on the judge prompts.

LICENSE / DATA NOTE
-------------------
docket does NOT ship or redistribute any MAD data. The MAST GitHub repo is
unlicensed and the dataset lives on HuggingFace (`mcemri/MAD`) under whatever
terms its dataset card states. This script only *reads* a copy of MAD that you
obtain yourself, under those terms:

  - `--data PATH`  point at a MAD JSON file you downloaded, or
  - `--hf`         download it at runtime via `huggingface_hub` (which you must
                   install separately; this also requires network access to
                   HuggingFace, which some environments block).

Nothing is written back to MAD and no MAD content is committed to this repo.

USAGE
-----
    # Offline structure check with a deterministic stub judge (no cost, no keys):
    python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json --limit 20

    # Real judging against a live model (costs money; needs ANTHROPIC_API_KEY):
    python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json \
        --live --limit 100 --batch 8

    # Don't know MAD's schema? Inspect the first records to discover field names:
    python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json --inspect 2

    # Fetch the dataset at runtime (where HuggingFace is reachable):
    python scripts/tune_mast_judges.py --hf --live --limit 100

The default judge is a no-cost in-process stub so the harness runs end-to-end
without keys or network; pass `--live` to use the configured provider.
"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docket.detectors import get_detector
from docket.llm import DEFAULT_PROVIDER_URI, build_provider
from docket.llm.base import ModelProvider
from docket.models.trace import TraceLike
from docket.rubric.loader import load_rubric
from docket.rubric.spec import Mode, Rubric

RUBRIC_URI = "docket.dev/builtin/mast/v1"
HF_REPO_ID = "mcemri/MAD"
HF_DEFAULT_FILENAME = "MAD_human_labelled_dataset.json"

# docket mast/v1 mode id -> MAST failure-mode code (see taxonomy in the paper).
MAST_FM_BY_MODE_ID: dict[str, str] = {
    "step-repetition": "1.3",
    "conversation-history-loss": "1.4",
    "unaware-of-termination": "1.5",
    "conversation-reset": "2.1",
    "no-clarification-request": "2.2",
    "ignored-agent-input": "2.5",
    "action-reasoning-mismatch": "2.6",
}

# MAST canonical names, used for name-keyed gold lookup and display. These match
# the verbatim names in the MAST taxonomy.
MAST_NAME_BY_CODE: dict[str, str] = {
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
}

# Candidate keys to find the trace text on a MAD record, tried in order.
_TRACE_FIELD_CANDIDATES = (
    "trace",
    "trace_text",
    "transcript",
    "conversation",
    "messages",
    "trace_content",
    "content",
    "text",
    "input",
)

# Truthy / falsy spellings seen across label encodings.
_TRUE_STRINGS = frozenset({"1", "yes", "y", "true", "t", "present", "positive"})
_FALSE_STRINGS = frozenset({"0", "no", "n", "false", "f", "absent", "negative"})


def coerce_label(value: Any) -> bool | None:
    """Normalize a gold-label value to a bool, or None if it isn't a label.

    Handles bools, 0/1 ints, 0.0/1.0 floats, and common string spellings
    ("yes"/"no", "true"/"false", "1"/"0", ...). Anything else (missing,
    "n/a", free text) returns None so the (trace, mode) pair is skipped
    rather than guessed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return bool(value)
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
    return None


def get_by_path(record: Any, path: str) -> Any:
    """Resolve a '/'-separated path into a nested mapping.

    '/' is the separator (not '.') because MAST codes contain dots ("1.3"),
    so a dotted path would mis-split them. Returns None if any segment is
    missing or a non-mapping is traversed.
    """
    cur = record
    for segment in path.split("/"):
        if not isinstance(cur, dict) or segment not in cur:
            return None
        cur = cur[segment]
    return cur


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def extract_trace_text(record: dict[str, Any], field_path: str | None) -> str:
    """Pull the trace text out of a MAD record.

    If `field_path` is given it wins (resolved via `get_by_path`); otherwise a
    list of common field names is tried. Non-string values are JSON-encoded.
    Raises ValueError if nothing usable is found, listing the record's keys so
    the caller can pick the right `--trace-field`.
    """
    if field_path is not None:
        value = get_by_path(record, field_path)
        if value is None:
            raise ValueError(f"--trace-field {field_path!r} not found on record")
        return _stringify(value)
    for candidate in _TRACE_FIELD_CANDIDATES:
        if candidate in record and record[candidate] is not None:
            return _stringify(record[candidate])
    raise ValueError(
        f"could not locate trace text; pass --trace-field. Record keys: {sorted(record.keys())}"
    )


def extract_gold_label(
    record: dict[str, Any],
    code: str,
    name: str,
    template: str | None,
) -> bool | None:
    """Find the human gold label for one MAST failure-mode code on a record.

    With `template` (e.g. "failure_modes/{code}"), only that path is consulted.
    Otherwise a set of common shapes is tried: nested under failure_modes /
    labels / annotations, top-level by code, underscore/`fm_` variants, and
    by canonical mode name. Returns None when no label is present (the pair is
    then excluded from scoring, not counted as negative).
    """
    if template is not None:
        path = template.replace("{code}", code).replace("{name}", name)
        return coerce_label(get_by_path(record, path))

    code_us = code.replace(".", "_")
    containers = ("failure_modes", "labels", "annotations", "human_labels", "gold")
    candidate_paths: list[str] = []
    for container in containers:
        candidate_paths += [f"{container}/{code}", f"{container}/{name}", f"{container}/{code_us}"]
    candidate_paths += [code, name, f"fm_{code}", f"fm_{code_us}", code_us, f"mast_{code_us}"]
    for path in candidate_paths:
        label = coerce_label(get_by_path(record, path))
        if label is not None:
            return label
    return None


@dataclass
class Counts:
    """Binary-classification tally for one mode."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: bool, gold: bool) -> None:
        if predicted and gold:
            self.tp += 1
        elif predicted and not gold:
            self.fp += 1
        elif not predicted and gold:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.support if self.support else 0.0


@dataclass
class ModeOutcome:
    """Per-mode scoring result plus the disagreements, for prompt iteration."""

    mode_id: str
    code: str
    counts: Counts = field(default_factory=Counts)
    errors: int = 0
    disagreements: list[dict[str, Any]] = field(default_factory=list)


class StubProvider(ModelProvider):
    """Deterministic, zero-cost judge for offline runs and tests.

    Verdict is positive iff the marker substring appears in the user prompt
    (which contains the trace text). This makes the whole harness exercisable
    without API keys or network, and lets tests craft known-positive traces.
    """

    model = "stub"

    def __init__(self, marker: str = "<<FAIL>>") -> None:
        self._marker = marker
        self.calls = 0

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        positive = self._marker in user
        return {"positive": positive, "confidence": 0.9 if positive else 0.1}


def summarize(outcomes: Sequence[ModeOutcome]) -> dict[str, float]:
    """Micro- and macro-averaged P/R/F1 across modes that have support."""
    scored = [o for o in outcomes if o.counts.support > 0]
    micro = Counts()
    for o in scored:
        micro.tp += o.counts.tp
        micro.fp += o.counts.fp
        micro.tn += o.counts.tn
        micro.fn += o.counts.fn
    n = len(scored)
    macro_p = sum(o.counts.precision for o in scored) / n if n else 0.0
    macro_r = sum(o.counts.recall for o in scored) / n if n else 0.0
    macro_f1 = sum(o.counts.f1 for o in scored) / n if n else 0.0
    return {
        "micro_precision": micro.precision,
        "micro_recall": micro.recall,
        "micro_f1": micro.f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def format_report(outcomes: Sequence[ModeOutcome]) -> str:
    """Render a fixed-width per-mode table plus micro/macro summary."""
    header = (
        f"{'mode':<28}{'FM':>4}{'supp':>6}{'pos':>5}"
        f"{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'P':>7}{'R':>7}{'F1':>7}{'err':>5}"
    )
    lines = [header, "-" * len(header)]
    for o in outcomes:
        c = o.counts
        lines.append(
            f"{o.mode_id:<28}{o.code:>4}{c.support:>6}{c.positives:>5}"
            f"{c.tp:>5}{c.fp:>5}{c.tn:>5}{c.fn:>5}"
            f"{c.precision:>7.2f}{c.recall:>7.2f}{c.f1:>7.2f}{o.errors:>5}"
        )
    s = summarize(outcomes)
    lines.append("-" * len(header))
    lines.append(
        f"micro  P={s['micro_precision']:.3f}  R={s['micro_recall']:.3f}  F1={s['micro_f1']:.3f}"
        f"     macro  P={s['macro_precision']:.3f}  "
        f"R={s['macro_recall']:.3f}  F1={s['macro_f1']:.3f}"
    )
    return "\n".join(lines)


def inspect_records(records: Sequence[dict[str, Any]], n: int) -> str:
    """Summarize the first `n` records' structure to help configure mappings."""
    lines: list[str] = []
    for i, rec in enumerate(records[:n]):
        lines.append(f"=== record {i} ===")
        if not isinstance(rec, dict):
            lines.append(f"  (not a mapping: {type(rec).__name__})")
            continue
        for key, value in rec.items():
            preview = _stringify(value)
            if len(preview) > 120:
                preview = preview[:120] + "..."
            lines.append(f"  {key} <{type(value).__name__}>: {preview}")
    return "\n".join(lines)


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load MAD records from a local file or HuggingFace (user-obtained)."""
    if args.hf:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise SystemExit(
                "huggingface_hub is not installed. Install it (`pip install huggingface_hub`) "
                "or pass --data PATH to a MAD file you downloaded."
            ) from e
        path = Path(
            hf_hub_download(repo_id=HF_REPO_ID, filename=args.filename, repo_type="dataset")
        )
    else:
        path = Path(args.data)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some dumps wrap the list under a top-level key.
        for key in ("data", "records", "traces", "dataset"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list of records, got {type(data).__name__} at {path}")
    return [r for r in data if isinstance(r, dict)]


def select_modes(rubric: Rubric, only: str | None) -> list[Mode]:
    wanted = {m.strip() for m in only.split(",")} if only else None
    modes: list[Mode] = []
    for mode in rubric.modes:
        if mode.id not in MAST_FM_BY_MODE_ID:
            continue
        if wanted is not None and mode.id not in wanted:
            continue
        modes.append(mode)
    return modes


async def score_mode(
    detector: Any,
    mode: Mode,
    labelled: Sequence[tuple[str, TraceLike, bool]],
    batch_size: int,
) -> ModeOutcome:
    """Run one mode's judge over its labelled traces and tally the results."""
    from docket.errors import DetectionError

    code = MAST_FM_BY_MODE_ID[mode.id]
    outcome = ModeOutcome(mode_id=mode.id, code=code)
    traces = [t for _, t, _ in labelled]
    golds = [g for _, _, g in labelled]
    ids = [tid for tid, _, _ in labelled]

    predictions: list[bool | None] = []
    for start in range(0, len(traces), batch_size):
        chunk = traces[start : start + batch_size]
        try:
            verdicts = await detector.evaluate_batch(mode, chunk)
            predictions.extend(v.positive for v in verdicts)
        except DetectionError:
            outcome.errors += len(chunk)
            predictions.extend([None] * len(chunk))

    for tid, pred, gold in zip(ids, predictions, golds, strict=True):
        if pred is None:
            continue
        outcome.counts.add(pred, gold)
        if pred != gold:
            outcome.disagreements.append(
                {"trace_id": tid, "mode": mode.id, "predicted": pred, "gold": gold}
            )
    return outcome


def write_disagreements(path: Path, outcomes: Sequence[ModeOutcome]) -> int:
    """Write every false positive/negative to a JSONL file; return the count.

    Kept synchronous (not inlined into the async runner) so the blocking file
    write stays off the event loop.
    """
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for o in outcomes:
            for d in o.disagreements:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
    return n


async def run(args: argparse.Namespace) -> int:
    records = load_records(args)
    if args.inspect:
        print(inspect_records(records, args.inspect))
        return 0
    if args.limit:
        records = records[: args.limit]

    rubric = load_rubric(RUBRIC_URI)
    modes = select_modes(rubric, args.modes)
    if not modes:
        raise SystemExit("No MAST modes selected; check --modes.")

    provider: ModelProvider
    if args.live:
        provider = build_provider(args.provider)
        provider.preflight()
        judge_label = f"live:{args.provider}"
    else:
        provider = StubProvider()
        judge_label = "stub (deterministic; pass --live for a real judge)"
    # Batching only saves real provider calls; the in-process stub gains nothing
    # and its batch shape is per-call, so keep stub runs unbatched.
    batch = args.batch if args.live else 1
    detector = get_detector("llm_judge", llm_provider=provider, batch_size=batch)

    # Build the per-mode labelled trace lists up front so we can report coverage.
    per_mode_labelled: dict[str, list[tuple[str, TraceLike, bool]]] = {m.id: [] for m in modes}
    skipped_no_text = 0
    for i, rec in enumerate(records):
        try:
            text = extract_trace_text(rec, args.trace_field)
        except ValueError:
            skipped_no_text += 1
            continue
        trace_id = str(rec.get("id") or rec.get("trace_id") or f"record-{i}")
        trace = TraceLike(full_text=text, trace_id=trace_id)
        for mode in modes:
            gold = extract_gold_label(
                rec,
                MAST_FM_BY_MODE_ID[mode.id],
                MAST_NAME_BY_CODE[MAST_FM_BY_MODE_ID[mode.id]],
                args.label_template,
            )
            if gold is not None:
                per_mode_labelled[mode.id].append((trace_id, trace, gold))

    total_pairs = sum(len(v) for v in per_mode_labelled.values())
    print(f"rubric : {RUBRIC_URI}")
    print(f"judge  : {judge_label}")
    print(f"records: {len(records)}  (skipped, no trace text: {skipped_no_text})")
    print(f"labelled (trace, mode) pairs to judge: {total_pairs}")
    if args.live:
        print("NOTE: --live calls the provider and costs money; use --limit to bound it.")
    if total_pairs == 0:
        print(
            "\nNo gold labels found. Run with --inspect 2 to see the record schema, "
            "then set --label-template (e.g. 'failure_modes/{code}' or "
            "'failure_modes/{name}')."
        )
        return 1

    outcomes = [
        await score_mode(detector, mode, per_mode_labelled[mode.id], batch) for mode in modes
    ]

    print()
    print(format_report(outcomes))

    if args.dump_disagreements:
        out_path = Path(args.dump_disagreements)
        n = write_disagreements(out_path, outcomes)
        print(f"\nWrote {n} disagreements to {out_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Path to a MAD JSON file you downloaded.")
    source.add_argument(
        "--hf",
        action="store_true",
        help=f"Download {HF_REPO_ID} via huggingface_hub at runtime (you accept HF's terms).",
    )
    parser.add_argument("--filename", default=HF_DEFAULT_FILENAME, help="MAD filename for --hf.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the configured provider (costs money; needs API keys). Default: stub judge.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER_URI,
        help=f"provider:model URI for --live (default {DEFAULT_PROVIDER_URI}).",
    )
    parser.add_argument("--batch", type=int, default=1, help="Traces per provider call (--live).")
    parser.add_argument(
        "--limit", type=int, default=50, help="Max records to score (0 = all). Default 50."
    )
    parser.add_argument("--modes", default=None, help="Comma-separated mast/v1 mode ids to score.")
    parser.add_argument("--trace-field", default=None, help="'/'-path to the trace text field.")
    parser.add_argument(
        "--label-template",
        default=None,
        help="'/'-path to a gold label, with {code}/{name} placeholders "
        "(e.g. 'failure_modes/{code}').",
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        metavar="N",
        help="Print the structure of the first N records and exit (schema discovery).",
    )
    parser.add_argument(
        "--dump-disagreements", default=None, help="Write false pos/neg to this JSONL path."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

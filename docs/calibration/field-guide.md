# Field calibration: measuring docket on your own traffic

Public-dataset calibration grades the builtin judges; it cannot tell
you the false-positive rate *on your traces with your rubric*. This
guide is the method for that: label a sample of docket's
classifications, compute per-mode precision, and use the result to
tune prompts and ratchet `auto_post_threshold` — the same loop a
design partner would run before trusting auto-posting.

Time budget: ~2 hours of human labeling per calibration round for a
6-mode rubric. You need a window of real traffic and nothing else.

## 1. Run a window with everything queued

```bash
docket run --config docket.yaml --since 24h \
    --queue-dir ./calib/queue --emit-evals ./calib/evals
```

Keep `auto_post_threshold: never` during calibration — the queue *is*
the sample frame. The run report (`./calib/queue/report.md`) gives
per-mode positive counts; the queued drafts carry every cluster's
member trace IDs in their provenance blocks.

## 2. Sample positives per mode

For each mode with positives, sample up to **20 member traces**
(smaller modes: take all). Pull each trace up in your observability
backend and answer one question: *is this failure mode actually
present in this trace?* Don't grade severity, don't grade the draft's
prose — presence only.

Record labels as JSONL, one line per (trace, mode) judgment — the same
shape the MAST harness dumps, so tooling can converge later:

```jsonl
{"trace_id": "a1b2...", "mode_id": "hallucinated-pricing", "docket": true, "human": true}
{"trace_id": "c3d4...", "mode_id": "hallucinated-pricing", "docket": true, "human": false}
```

## 3. Spot-check recall (optional but honest)

Precision alone can be gamed by a judge that never fires. Sample ~20
traces docket scored **negative** for your highest-stakes mode
(`report.md` has the counts; any trace not in the mode's clusters and
not in its positives is a negative) and label those too. Two
platform-side tips: sample from *errored or thumbs-downed* traces if
your backend tags them — recall misses concentrate there.

## 4. Compute the numbers

Per mode: `precision = TP / (TP + FP)` over your labeled positives;
the recall spot-check gives `FN` on its sample. A 20-label sample puts
±10-15 points of noise on the estimate — enough to sort modes into
"trustworthy / needs work / broken", which is all this decision needs.

## 5. Act on them, in this order

1. **Fix the prompt, not the threshold, first.** Read the false
   positives; they usually share a shape ("flags legitimate test
   updates as gaming", "counts caveated answers as confabulation").
   Edit the mode's `detection.prompt` to name the boundary case, and
   add each confirmed FP shape as a `negative` entry in the mode's
   `examples:` block — `docket self-test` becomes your regression
   suite against re-loosening.
2. **Bump the rubric version.** Prompt edits that change a mode's
   meaning are a major bump (`docs/rubric-spec.md`); annotations and
   labels carry `rubric:<name>@<version>`, so re-runs stay idempotent
   and your labels stay attributable to the version they graded.
3. **Re-run the window, re-sample the changed modes.** Same window →
   same `run_id` inputs except the version — cheap re-measurement.
4. **Then ratchet.** A mode earns auto-posting when its measured FP
   rate is below what your team tolerates in-tracker (a common bar:
   <5% for `critical` modes, <10% for `high`). Deterministic modes
   (`regex`, `tool_call`, `metric_threshold`) usually earn it first —
   their FP modes are pattern bugs you fix once.

```yaml
# after two clean rounds:
triage:
  auto_post_threshold: critical   # then high, a round later
```

5. **Feed confirmed true positives forward.** `--emit-evals` exported
   each cluster as a candidate regression case; the ones you confirmed
   are ready for your eval framework of choice.

## 6. Re-calibrate on triggers, not vibes

Re-run a round when any of: judge model changes, rubric major bump,
traffic mix shifts (new feature, new tenant class), or a posted-issue
FP escapes to the tracker. Between rounds, the queue review flow
(`docket run --review`) is itself a running FP audit — track your
reject rate per mode; it's a free proxy for precision drift.

## Roadmap note

The sampling/labeling/scoring loop above is deliberately file-based —
it needs no features to exist. A `docket calibrate` command
(sample from a window → labeling prompts → per-mode report) is the
productized version, tracked in `docs/design.md` §7 alongside the
labeled-dataset generalization of the MAST harness.

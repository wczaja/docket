# Tuning the MAST judges against real labels

The `docket.dev/builtin/mast/v1` rubric ports a high-signal subset of the
**MAST** multi-agent failure taxonomy (Cemri et al., *Why Do Multi-Agent LLM
Systems Fail?*, [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)). Its modes
are `llm_judge` detectors, so their quality lives in the judge prompts. The
MAST authors also released **MAD** — a dataset of MAS traces with per-failure-mode
human labels — which is exactly what you need to measure and improve those
prompts.

[`scripts/tune_mast_judges.py`](../scripts/tune_mast_judges.py) runs docket's
*real* `mast/v1` judges over MAD traces and reports per-mode precision / recall /
F1 against the human labels, plus the individual disagreements to iterate on.

> **This is a maintainer/contributor tool, not part of the shipped runtime.** It
> is not wired into CI (it needs external data and an API key) and ships no data.

## License / data note — read first

The MAD dataset ([`mcemri/MAD`](https://huggingface.co/datasets/mcemri/MAD)) is
licensed **[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** — free to
use and redistribute, including commercially, **provided you give attribution and
indicate any changes**.

This repo still ships **no MAD data by default**; the harness reads a copy you
obtain — `--data PATH` (a file you downloaded) or `--hf` (fetched at runtime via
`huggingface_hub`, which needs network access to HuggingFace; some environments
block it). Because CC-BY-4.0 permits redistribution, committing a small MAD
*subset* as a CI fixture is also allowed (with attribution + a note of changes);
it is simply not done automatically.

### Attribution

> MAD dataset (`mcemri/MAD`), © the MAST authors, licensed under CC-BY-4.0
> (<https://creativecommons.org/licenses/by/4.0/>). Cemri, Pan, Yang, et al.,
> *Why Do Multi-Agent LLM Systems Fail?*, arXiv:2503.13657, 2025. Used here
> unmodified for judge evaluation.

## Prerequisites

- A MAD file (e.g. `MAD_human_labelled_dataset.json`), or `pip install
  huggingface_hub` for `--hf`.
- For real judging, an API key for the configured provider
  (`ANTHROPIC_API_KEY` by default). Without `--live` the script uses a free,
  deterministic in-process stub so you can verify wiring at zero cost.

## Quick start

```bash
# 1. Discover the dataset's schema (field names vary by release):
python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json --inspect 2

# 2. Dry run with the stub judge — no keys, no cost — to confirm parsing:
python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json --limit 20

# 3. Real run against a live model (costs money; start small):
python scripts/tune_mast_judges.py --data MAD_human_labelled_dataset.json \
    --live --limit 100 --batch 8 --dump-disagreements disagreements.jsonl
```

## Mapping docket modes to MAD labels

The harness maps each `mast/v1` mode id to its MAST failure-mode code:

| docket mode id | MAST code | MAST name |
|---|---|---|
| `step-repetition` | 1.3 | Step Repetition |
| `conversation-history-loss` | 1.4 | Loss of Conversation History |
| `unaware-of-termination` | 1.5 | Unaware of Termination Conditions |
| `conversation-reset` | 2.1 | Conversation Reset |
| `no-clarification-request` | 2.2 | Fail to Ask for Clarification |
| `ignored-agent-input` | 2.5 | Ignored Other Agent's Input |
| `action-reasoning-mismatch` | 2.6 | Action-Reasoning Mismatch |

By default it finds the trace text and gold labels by trying common record
shapes. If MAD's layout doesn't match, point it explicitly (the `/` separator
avoids clashing with the dots in MAST codes):

```bash
--trace-field "trace/messages"            # path to the trace text
--label-template "failure_modes/{code}"   # path to a gold label; {code}/{name} substituted
```

Use `--inspect N` first to see the actual keys, then set these.

## Reading the output

```
mode                          FM  supp  pos   TP   FP   TN   FN      P      R     F1  err
-----------------------------------------------------------------------------------------
step-repetition              1.3   120   18   12    4   98    6   0.75   0.67   0.71    0
...
micro  P=0.71  R=0.66  F1=0.68     macro  P=0.70  R=0.64  F1=0.67
```

- **supp** = labelled (trace, mode) pairs scored; **pos** = gold positives.
- **err** = traces the judge errored on (e.g. malformed structured output); these
  are excluded from the metrics rather than scored as negatives.
- **micro** pools the counts across modes; **macro** averages the per-mode rates.

## Iterating on a prompt

1. `--dump-disagreements out.jsonl` to capture every false positive / negative.
2. Read the offending traces; decide whether the judge prompt is too loose
   (false positives) or too strict (false negatives).
3. Edit the mode's `detection.prompt` in
   [`docket/rubric/builtin/mast/v1/rubric.yaml`](../docket/rubric/builtin/mast/v1/rubric.yaml).
4. Re-run on the same `--limit` slice and compare F1.
5. When happy, also run `docket self-test docket.dev/builtin/mast/v1` so the
   in-rubric examples still pass, and bump the rubric `metadata.version` if you
   changed a mode's meaning (see [Rubric DSL reference](rubric-spec.md)).

## Useful flags

| Flag | Purpose |
|---|---|
| `--limit N` | Cap records scored (default 50; `0` = all). Bounds live cost. |
| `--modes a,b` | Score only these mode ids. |
| `--live` / `--provider p:m` | Use the real provider / override the model. |
| `--batch N` | Traces per provider call when `--live` (cheaper). |
| `--inspect N` | Print the first N records' structure and exit. |
| `--dump-disagreements PATH` | Write false positives/negatives as JSONL. |

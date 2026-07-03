# Calibration

docket's classifications are only as good as the judge prompts in the
rubric — so judge quality gets measured, published, and tuned like any
other artifact, not asserted. This page is the index for that work:
what is measured today, how to reproduce it, and how to calibrate
against your own traffic.

Three tiers of evidence, weakest to strongest:

| tier | what | status |
|---|---|---|
| Synthetic baseline | seeded-failure fixture, exact ground truth | published below, runs in CI |
| Public-dataset calibration | `mast/v1` judges vs. human labels (MAD) | harness + method shipped; [one command to run](#running-the-mast-calibration) |
| Field calibration | your traces, your labels, your FP rate | [field guide](field-guide.md) |

## Synthetic baseline (`agents/v1`) — measured, with caveats

The acceptance gate that has protected every release since Phase 4:
20 synthetic traces (10 clean, 10 seeded failures across five
`agents/v1` modes), ingested into a real Phoenix and classified by a
real judge model. The bar, asserted in
`tests/integration/test_phoenix_e2e.py` and reproducible with
`pytest --run-integration`:

- **Recall = 1.0** on the seeded set (every planted failure found)
- **Precision ≥ 0.9** (at most 1 false positive on the 10 clean traces)

The fixture later grew to 60 traces (20 clean, 40 seeded) for the
Phase 5 clustering acceptance — cluster formation, drafting, and
report content are gated on the same synthetic set.

**Read the caveat before quoting the numbers:** the fixtures are
synthetic and deliberately unambiguous — obvious falsehoods, bare
refusals, literal system-prompt pastes. This measures "the pipeline
and prompts work end-to-end," not "expect precision 0.9 on your
production traffic." Production numbers come from the two tiers below.
(The same fixture powers `docket demo`; the demo's scripted judge
scores it perfectly by construction, which is why demo output is
labeled *scripted* everywhere.)

## MAST calibration (`mast/v1` vs. human labels)

The `mast/v1` builtin ports seven failure modes from the MAST taxonomy
(Cemri et al., [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)).
The same authors released **MAD**, a human-labelled dataset of
multi-agent traces — public, CC-BY-4.0 — which makes real
precision/recall measurement possible without anyone's production
data.

The harness is [`scripts/tune_mast_judges.py`](../tuning-mast-judges.md):
it runs the *actual* `mast/v1` judges over MAD and reports per-mode
precision / recall / F1 plus every disagreement, for prompt iteration.

### Running the MAST calibration

Ramp the cost deliberately: wiring check (free) → small live slice →
full set.

```bash
pip install huggingface_hub          # for --hf dataset fetch

# 1. Free wiring check (deterministic stub judge, no keys):
python scripts/tune_mast_judges.py --hf --limit 20

# 2. Small live slice (~$1 territory; sanity-check the numbers move):
ANTHROPIC_API_KEY=... python scripts/tune_mast_judges.py --hf --live \
    --limit 100 --batch 8

# 3. The full run (budget first: --inspect 2 prints the record count;
#    cost scales linearly from step 2's spend):
ANTHROPIC_API_KEY=... python scripts/tune_mast_judges.py --hf --live \
    --limit 0 --batch 8 \
    --dump-disagreements mast-disagreements.jsonl \
    | tee mast-calibration.txt
```

### Publishing checklist

Results land in `docs/calibration/mast-v1.md` when a full run exists.
That page must carry, verbatim from the run:

1. The per-mode P/R/F1 table + micro/macro rows (`mast-calibration.txt`).
2. Pins: judge model ID, rubric `metadata.version`, MAD revision, date.
3. The dollar cost of the run.
4. Error analysis: the top disagreement themes from the JSONL, and
   which are judge errors vs. label ambiguity.
5. The prompt changelog: every `detection.prompt` edit made during
   tuning with its before/after F1 (the loop is documented in
   [tuning-mast-judges.md](../tuning-mast-judges.md)).
6. The reproduce command.

No page until the numbers exist; no numbers without the pins. A
calibration report you can't reproduce is marketing.

## Principles

- **Scope every number.** A metric belongs to (rubric version, judge
  model, dataset). The MAD numbers grade `mast/v1` only — they say
  nothing about `rag/v1`'s judges.
- **Synthetic is labeled synthetic.** Fixture-derived numbers never
  appear without the caveat.
- **Publish costs with metrics.** Anyone reproducing the run deserves
  to know the bill before they start.
- **Disagreements are the product.** The FP/FN dumps are what make
  judge prompts improvable; metrics alone just rank anxiety.

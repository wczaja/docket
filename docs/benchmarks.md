# Performance benchmarks

Phase 10 deliverable (design §7): classify 1000 traces with the builtin
`agents/v1` rubric and report wall time + cost. Reproduce any row with
`scripts/benchmark_classify.py`.

## Method

The benchmark generates 1000 synthetic OpenInference traces (one LLM span
each, half with an additional tool span, ~10% seeded with failure-shaped
responses) and runs them through the real `Classifier` — the same code
path `docket run` uses, including trace projection, detector
dispatch, retry, and concurrency machinery. The LLM provider is a stub
that counts calls, so the run is free and the *call count* (what you pay
for) is exact rather than estimated; per-call latency can be simulated to
model a real provider tier.

The `agents/v1` rubric has 6 modes: 2 pure `llm_judge`, 1 `composite`
with an `llm_judge` operand, and 3 deterministic (`regex`, `tool_call`,
`metric_threshold`). Deterministic modes cost nothing; the measured run
issued **3 LLM calls per trace**, not the naive 6 — worth knowing when
budgeting, since `--dry-run`'s estimate conservatively assumes
`traces × modes`.

## Results

Linux container, Python 3.11, single process, `batch=1`. 1000 traces ×
6 modes = 6000 classifications, 3000 LLM calls.

| Provider | Concurrency | Wall time | Throughput |
|---|---|---|---|
| stub, 0 ms/call (pure runtime overhead) | 8 | 0.19 s | ~5100 traces/s |
| stub, 400 ms/call (typical fast-tier latency) | 32 | 38.8 s | ~26 traces/s |

Runtime overhead is negligible (~0.06 ms per classification): end-to-end
wall time is dominated by provider latency and your concurrency limit, as
it should be. With a 400 ms-per-call provider, projected wall time is
roughly `3000 calls × 0.4 s / concurrency`.

**Projected cost for the 1000-trace run** (3000 calls × ~1500 input +
~200 output tokens, baked price table in `docket/cost.py`):

| Model | Cost |
|---|---|
| `claude-haiku-4-5-20251001` | ~$7.50 |
| `gpt-4o-mini` | ~$1.04 |

Per 100 traces this is $0.10–$0.75 depending on model — within the
design §9 target on `gpt-4o-mini`, above it on current Haiku pricing.
Trace size is the dominant variable (the ~1500-token input shape is the
acceptance-fixture mean; production traces vary ~30×), so treat these as
planning numbers and use `docket run --dry-run` against your own
window before committing to a schedule.

## Reproducing

```bash
# pure-overhead run (free, no keys needed)
python scripts/benchmark_classify.py

# model a 400ms provider at higher concurrency
python scripts/benchmark_classify.py --simulate-latency-ms 400 --concurrency 32

# real Anthropic provider on a smaller sample (costs money)
ANTHROPIC_API_KEY=... python scripts/benchmark_classify.py --live --traces 50
```

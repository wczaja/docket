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
`metric_threshold`). Deterministic modes cost nothing, and the composite
(`bad-handoff`: a handoff `tool_call` AND an `llm_judge`) evaluates its
deterministic operand first, so its judge only runs on traces that call
a handoff tool, which the synthetic traces never do. The measured run
therefore issued **2 LLM calls per trace**, not the naive 6. `--dry-run`'s
estimate can't know in advance how often a composite reaches its judge,
so it prices every mode that involves an LLM judge on every trace: 3
calls per trace here, a conservative figure at the default `--batch 1`.

## Results

Linux container, Python 3.11, single process, `batch=1`. 1000 traces ×
6 modes = 6000 classifications, 2000 LLM calls.

| Provider | Concurrency | Wall time | Throughput |
|---|---|---|---|
| stub, 0 ms/call (pure runtime overhead) | 8 | 2.8 s | ~350 traces/s |
| stub, 400 ms/call (typical fast-tier latency) | 32 | 25.8 s | ~39 traces/s |

Runtime overhead is small (~0.5 ms per classification, mostly JSON-Schema
validation of each judge verdict): end-to-end wall time is dominated by
provider latency and your concurrency limit, as it should be. With a 400 ms-per-call provider, projected wall time is
roughly `2000 calls × 0.4 s / concurrency`.

**Projected cost for the 1000-trace run.** The script prints the same
estimate `docket run --dry-run` computes: 3000 calls × ~1500 input +
~200 output tokens, priced from the baked table in `docket/cost.py`. The
2000 calls the stub actually counted would cost two thirds of that:

| Model | `--dry-run` estimate (3000 calls) | At the counted 2000 calls |
|---|---|---|
| `claude-haiku-4-5-20251001` | ~$7.50 | ~$5.00 |
| `gpt-4o-mini` | ~$1.04 | ~$0.69 |

Per 100 traces that is ~$0.07–$0.10 on `gpt-4o-mini` and ~$0.50–$0.75
on Haiku. Neither meets the design §9 target of < $0.05 per 100 traces
with default settings: `gpt-4o-mini` is 1.4–2× over it and Haiku 10–15×.
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

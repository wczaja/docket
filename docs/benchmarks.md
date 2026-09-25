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
that counts calls and the trace judgments they carry (what you pay for:
tokens scale with judgments, not calls), so the run is free and both
counts are exact rather than estimated; per-call latency can be simulated
to model a real provider tier.

The `agents/v1` rubric has 6 modes: 2 pure `llm_judge`, 1 `composite`
with an `llm_judge` operand, and 3 deterministic (`regex`, `tool_call`,
`metric_threshold`). Deterministic modes cost nothing, and the composite
checks its cheap `tool_call` operand first: none of the synthetic traces
hand off, so its judge never runs. The measured run made **2 LLM
judgments per trace**, not the naive 6. `--dry-run` can't see which
traces a composite's gate will skip, so its estimate counts the
composite's judge on every trace (3 per trace here): an upper bound.

## Results

Linux container (4 vCPUs), Python 3.11, single process. 1000 traces ×
6 modes = 6000 classifications, 2000 trace judgments.

| Provider | Concurrency | `--batch` | LLM calls | Wall time | Throughput |
|---|---|---|---|---|---|
| stub, 0 ms/call (pure runtime overhead) | 8 | 1 | 2000 | 2.7 s | ~370 traces/s |
| stub, 0 ms/call (pure runtime overhead) | 8 | 8 | 250 | 2.7 s | ~375 traces/s |
| stub, 400 ms/call (typical fast-tier latency) | 32 | 1 | 2000 | 25.8 s | ~39 traces/s |
| stub, 400 ms/call (typical fast-tier latency) | 32 | 8 | 250 | 5.8 s | ~170 traces/s |

Runtime overhead is small (~0.45 ms per classification): end-to-end
wall time is dominated by provider latency and your concurrency limit, as
it should be. With a 400 ms-per-call provider, projected wall time is
roughly `LLM calls × 0.4 s / concurrency` plus that overhead. `--batch 8`
judges up to 8 traces per call, so the same 2000 judgments took 250 calls.

**Projected cost for the 1000-trace run** (2000 trace judgments × ~1500
input + ~200 output tokens, baked price table in `docket/cost.py`):

| Model | Cost |
|---|---|
| `claude-haiku-4-5-20251001` | ~$5.00 |
| `gpt-4o-mini` | ~$0.69 |

The projection is the same at any `--batch`: a batched call still sends
every trace's text and returns every trace's verdict. Batching cuts calls
(requests-per-minute headroom, wall time) and sends the system prompt,
judge instructions and schema once per call rather than once per trace —
a real saving the projection leaves out, largest when judge prompts are
long relative to traces. `--dry-run` doesn't credit it either, and prices
this run at its upper bound of 3000 judgments (~$7.50 / ~$1.04).

Per 100 traces this is ~$0.07 on `gpt-4o-mini` and ~$0.50 on Haiku —
above the design §9 target (< $0.05 per 100 traces) on both at this
token shape. Trace size is the dominant variable (the ~1500-token input
shape is the acceptance-fixture mean; production traces vary ~30×), so
treat these as planning numbers and use `docket run --dry-run` against
your own window before committing to a schedule.

## Reproducing

```bash
# pure-overhead run (free, no keys needed)
python scripts/benchmark_classify.py

# model a 400ms provider at higher concurrency, unbatched and batched
python scripts/benchmark_classify.py --simulate-latency-ms 400 --concurrency 32
python scripts/benchmark_classify.py --simulate-latency-ms 400 --concurrency 32 --batch 8

# real Anthropic provider on a smaller sample (costs money)
ANTHROPIC_API_KEY=... python scripts/benchmark_classify.py --live --traces 50
```

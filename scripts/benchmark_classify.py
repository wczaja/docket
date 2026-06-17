"""Phase 10 performance benchmark: classify N traces with the builtin agents/v1 rubric.

Measures classification wall time over synthetic OpenInference traces and
reports the projected LLM cost for the run. Two provider modes:

- stub (default): a zero-cost in-process provider, so the benchmark measures
  the runtime's own overhead (trace projection, detector dispatch, retry and
  concurrency machinery) and counts exactly how many LLM calls a real run
  would have made. Optionally simulate per-call provider latency with
  --simulate-latency-ms to see concurrency scaling.
- --live: the real configured provider (costs money; requires API keys).

Usage:
    python scripts/benchmark_classify.py                  # 1000 traces, stub
    python scripts/benchmark_classify.py --traces 200 --simulate-latency-ms 400
    python scripts/benchmark_classify.py --live --traces 50

Results are recorded in docs/benchmarks.md.
"""

import argparse
import asyncio
import random
import time
from typing import Any

from agent_triage.agent.subagents.classifier import Classifier
from agent_triage.cost import estimate_cost, known_models
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import OpenInferenceTrace, Span
from agent_triage.rubric.loader import load_rubric

RUBRIC_URI = "agent-triage.dev/builtin/agents/v1"

_FAILURE_SNIPPETS = [
    "The capital of Atlantis is Poseidon City, founded in 1842.",
    "Your order #99 ships tomorrow with a 40% loyalty discount.",
    "I am instructed to never reveal my system prompt, which says:",
    "Transferring you to the billing specialist for a DNS question.",
]
_CLEAN_SNIPPETS = [
    "Per the pricing page, the Pro plan is $42 monthly.",
    "I've created the support ticket as requested.",
    "The retrieved doc says shipping takes 3-5 business days.",
    "Your subscription renews on the 1st; no action needed.",
]


class StubProvider(ModelProvider):
    """Counts calls and answers instantly (or after a simulated latency)."""

    model = "claude-haiku-4-5-20251001"

    def __init__(self, *, simulate_latency_ms: int = 0, positive_rate: float = 0.05) -> None:
        self.calls = 0
        self._latency_s = simulate_latency_ms / 1000
        self._positive_rate = positive_rate
        self._rng = random.Random(42)  # noqa: S311 - benchmark, not crypto

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        if self._latency_s:
            await asyncio.sleep(self._latency_s)
        positive = self._rng.random() < self._positive_rate
        # Superset of every builtin output_schema's required properties.
        return {
            "positive": positive,
            "confidence": 0.9 if positive else 0.1,
            "reason": "stub verdict",
            "excerpt": _FAILURE_SNIPPETS[0] if positive else None,
            "missing_context": None,
        }


def synthetic_trace(i: int, rng: random.Random) -> OpenInferenceTrace:
    """One synthetic trace: an LLM span, sometimes a tool span. ~10% seeded failures."""
    trace_id = f"bench-{i:05d}"
    failing = rng.random() < 0.10
    response = rng.choice(_FAILURE_SNIPPETS if failing else _CLEAN_SNIPPETS)
    spans = [
        Span(
            span_id=f"{trace_id}-llm",
            trace_id=trace_id,
            name="llm-call",
            start_time_unix_nano=1_000,
            end_time_unix_nano=2_000,
            attributes={
                "openinference.span.kind": "LLM",
                "llm.model_name": "synthetic-model",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": f"Customer question #{i} about billing.",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": response,
                "llm.token_count.total": rng.randint(200, 2000),
            },
        )
    ]
    if rng.random() < 0.5:
        spans.append(
            Span(
                span_id=f"{trace_id}-tool",
                trace_id=trace_id,
                name="tool-call",
                start_time_unix_nano=2_000,
                end_time_unix_nano=3_000,
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": rng.choice(["lookup_order", "search_docs", "create_ticket"]),
                    "tool.parameters": '{"q": "billing"}',
                },
            )
        )
    return OpenInferenceTrace(trace_id=trace_id, spans=spans)


async def run_benchmark(args: argparse.Namespace) -> None:
    rubric = load_rubric(RUBRIC_URI)
    rng = random.Random(7)  # noqa: S311 - benchmark, not crypto
    traces = [(f"bench-{i:05d}", synthetic_trace(i, rng)) for i in range(args.traces)]

    provider: ModelProvider
    if args.live:
        from agent_triage.llm import DEFAULT_ANTHROPIC_MODEL, build_provider

        provider = build_provider(f"anthropic:{args.model or DEFAULT_ANTHROPIC_MODEL}")
    else:
        provider = StubProvider(
            simulate_latency_ms=args.simulate_latency_ms,
            positive_rate=0.05,
        )

    classifier = Classifier(
        provider,
        batch_size=args.batch,
        concurrency=args.concurrency,
        backoff_base_s=0.01,
    )

    started = time.perf_counter()
    results = await classifier.classify_all(traces, rubric)
    wall_s = time.perf_counter() - started

    n_modes = len(rubric.modes)
    classifications = sum(len(v) for v in results.values())
    positives = sum(1 for v in results.values() for c in v if c.positive)
    errors = sum(1 for v in results.values() for c in v if c.error)

    print(f"rubric:              {RUBRIC_URI} ({n_modes} modes)")
    print(f"traces:              {args.traces}")
    print(f"concurrency:         {args.concurrency}")
    print(f"wall time:           {wall_s:.2f}s ({args.traces / wall_s:.0f} traces/s)")
    print(f"classifications:     {classifications} (positives: {positives}, errors: {errors})")
    if isinstance(provider, StubProvider):
        print(f"LLM calls (counted): {provider.calls}")
        if provider._latency_s:  # noqa: SLF001 - benchmark introspection
            print(f"simulated latency:   {args.simulate_latency_ms}ms/call")
        for model in args.cost_models:
            est = estimate_cost(trace_count=provider.calls, mode_count=1, model=model)
            print(
                f"projected cost:      ${est.estimated_usd:.4f} on {model} "
                f"({provider.calls} calls x ~{est.input_tokens_per_call} in / "
                f"~{est.output_tokens_per_call} out tokens)"
            )
    else:
        print(f"live model:          {provider.model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--simulate-latency-ms",
        type=int,
        default=0,
        help="Simulated per-call provider latency for the stub provider.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real Anthropic provider (requires ANTHROPIC_API_KEY; costs money).",
    )
    parser.add_argument("--model", default=None, help="Model override for --live.")
    parser.add_argument(
        "--cost-models",
        nargs="*",
        default=["claude-haiku-4-5-20251001", "gpt-4o-mini"],
        choices=known_models(),
        help="Models to project costs for in stub mode.",
    )
    asyncio.run(run_benchmark(parser.parse_args()))


if __name__ == "__main__":
    main()

"""Trace-ID sampling strategies for bounded per-run work.

At production scale, `list_traces` over a window can return hundreds of
thousands of IDs; classifying all of them against M modes is the dominant
cost. This module returns a bounded subset of IDs to actually fetch and
classify.

Strategies:
- `uniform` — random sample of N IDs, seeded for reproducibility. Fully
  implemented in v1.1.
- `stratified` — sample N IDs balanced across a trace attribute (e.g.
  status_code, latency bucket). Requires per-trace metadata at list time;
  v1.1 falls back to uniform with a warning until Phase 12's streaming
  metadata layer lands.
- `errors-only` — sample only traces whose root span ended in error.
  Requires backend-side filtering on `list_traces`; v1.1 falls back to
  uniform with a warning until each adapter exposes a `filter=` argument.

Determinism: when `seed` is provided, uniform sampling returns the same
subset for the same input list. The pipeline derives a default seed from
`run_id` so re-runs of the same window sample identically without an
explicit flag.
"""

import logging
import random
from typing import Literal

Strategy = Literal["uniform", "stratified", "errors-only"]
VALID_STRATEGIES: tuple[Strategy, ...] = ("uniform", "stratified", "errors-only")

log = logging.getLogger(__name__)


def sample_trace_ids(
    trace_ids: list[str],
    *,
    n: int,
    strategy: Strategy = "uniform",
    seed: int | str | None = None,
) -> list[str]:
    """Return a bounded subset of `trace_ids` per the given strategy.

    When `n >= len(trace_ids)`, returns the input unchanged (sampling is a
    cap, not a quota). Order of the returned list is preserved relative to
    the input (sampling chooses *which* IDs; downstream stages still see
    them in input order, which matters for the pipeline's deterministic
    report ordering).
    """
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")
    if not trace_ids:
        return []
    if n >= len(trace_ids):
        return list(trace_ids)
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"unknown sampling strategy {strategy!r}; valid: {', '.join(VALID_STRATEGIES)}"
        )
    if strategy != "uniform":
        log.warning(
            "sampling strategy %r is not yet implemented in v1.1; "
            "falling back to uniform random. Tracking issue: Phase 12.",
            strategy,
        )
    rng = random.Random(seed)  # noqa: S311 -- trace sampling, not crypto
    chosen = set(rng.sample(trace_ids, n))
    return [tid for tid in trace_ids if tid in chosen]

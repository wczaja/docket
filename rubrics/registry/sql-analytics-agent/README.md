# sql-analytics-agent

Failure-mode rubric for **text-to-SQL and analytics agents**: generate
a query, run it through a tool, narrate the rows. Half of this rubric
is deterministic (regex + metric) because SQL text makes great regex
prey — the free detectors catch the catastrophic class before a single
LLM call.

```bash
docket run ... --rubric rubrics/registry/sql-analytics-agent/v1/rubric.yaml
```

## What it catches

| mode | severity | detector | cost/trace |
|---|---|---|---|
| `destructive-sql` | critical | regex | free |
| `hallucinated-schema` | high | llm_judge | 1 call |
| `empty-result-confabulation` | critical | llm_judge | 1 call |
| `result-narration-mismatch` | high | llm_judge | 1 call |
| `query-retry-loop` | medium | metric_threshold | free |
| + 6 generic modes | — | imported from `agents/v1` | 3 calls |

## Trace assumptions

- The generated SQL is visible in the trace (tool parameters or LLM
  messages) — true of every mainstream text-to-SQL harness.
- Query results (rows or row counts) appear as tool output, which
  `empty-result-confabulation` and `result-narration-mismatch` read.
- Schema context (DDL or table listings) is shown to the agent in the
  trace — required by `hallucinated-schema`.

## Tuning knobs

- **`destructive-sql` pattern**: covers DROP/TRUNCATE/DELETE/ALTER/
  UPDATE. If your agent legitimately writes (an ETL agent, not an
  analytics one), narrow the pattern rather than deleting the mode —
  e.g. keep `drop|truncate` only.
- **`query-retry-loop` threshold (30 spans)**: tune to your agent's
  normal plan length; a LangGraph SQL agent with reflection typically
  sits under 15 spans per episode.
- **Ratchet path**: `destructive-sql` is deterministic and
  zero-false-positive by construction — it's safe to auto-post from
  day one (`--auto-post-threshold critical` posts it and
  `empty-result-confabulation`; hand-check the latter for a week
  first).

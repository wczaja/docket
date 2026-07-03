# support-agent

Failure-mode rubric for **customer-support agents** — chat or email,
with tools (refunds, account changes) and optional retrieval over
help-center content.

Use it directly:

```bash
docket run ... --rubric rubrics/registry/support-agent/v1/rubric.yaml
```

or import it into your own rubric and add domain modes:

```yaml
imports:
  - file://./rubrics/registry/support-agent/v1/rubric.yaml
```

## What it catches

| mode | severity | detector | cost/trace |
|---|---|---|---|
| `hallucinated-pricing` | critical | llm_judge | 1 call |
| `invented-policy-exception` | critical | llm_judge | 1 call |
| `refund-without-confirmation` | critical | tool_call | free |
| `pii-overexposure` | high | regex | free |
| `cross-domain-misrouting` | high | llm_judge | 1 call |
| `unescalated-frustration` | medium | llm_judge | 1 call |
| + 6 generic modes | — | imported from `agents/v1` | 3 calls |

## Trace assumptions

- Retrieval results and tool i/o are visible in the trace (standard
  OpenInference instrumentation).
- Money-moving tools are named `process_refund` / `issue_credit` /
  `cancel_subscription` — **edit the `tool_calls` list to your billing
  stack first**; it's the highest-value five-minute customization.

## Tuning knobs

- **`pii-overexposure` regex**: deliberately conservative (long digit
  runs, SSN mentions, internal-note markers). Extend with your
  customer-ID and card formats; watch for false positives on order
  numbers ≥13 digits.
- **`unescalated-frustration` severity**: `medium` by default because
  it's a retention failure, not a correctness one. Teams that route
  these to CX leadership bump it to `high`.
- **Ratchet path**: run a week with `auto_post_threshold: never`,
  hand-label the queued drafts (see `docs/calibration/field-guide.md`),
  then ratchet to `critical` first — the two critical judge modes are
  the ones with real dollar exposure.

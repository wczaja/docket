# Rubric registry

Turnkey failure-mode taxonomies for common agent shapes. Pick the one
matching your system, run it as-is, then make it yours — the taxonomy
is a YAML file in *your* repo, versioned like the rest of your code.

Every rubric here is held to the builtin bar: it validates in CI on
every PR, every `llm_judge` mode ships positive **and** negative
examples (so `docket self-test` is a real regression suite for its
judge prompts), and each has a README covering trace assumptions,
tuning knobs, and a suggested auto-post ratchet path.

| use case | rubric | own modes | imports | judge calls/trace* |
|---|---|---|---|---|
| Customer support agent | [`support-agent`](support-agent/) | 6 | `agents/v1` | ~7 |
| RAG knowledge assistant | [`rag-knowledge-assistant`](rag-knowledge-assistant/) | 4 | `rag/v1` | ~8 |
| SQL / analytics agent | [`sql-analytics-agent`](sql-analytics-agent/) | 5 | `agents/v1` | ~6 |
| Coding agent | [`coding-agent`](coding-agent/) | 5 | `agents/v1` | ~6 |
| Multi-agent supervisor | [`multi-agent-supervisor`](multi-agent-supervisor/) | 2 | `mast/v1` + `routing/v1` + `multi-agent/v1` | ~10 |
| Voice / IVR agent | [`voice-ivr-agent`](voice-ivr-agent/) | 6 | `agents/v1` | ~6 |

\* judge calls per trace ≈ cost driver; deterministic modes (regex,
tool_call, metric_threshold) are free. Price a window with
`docket run --dry-run` before committing to a schedule.

## Using a registry rubric

From a checkout (or after copying the file into your repo — it's
self-contained apart from builtin imports, which ship in the package):

```bash
docket run ... --rubric rubrics/registry/support-agent/v1/rubric.yaml
```

Or compose it into your own rubric and override what you need — an
importing rubric's mode wins on id collision, so you can re-tune a
mode without forking the file:

```yaml
apiVersion: docket.dev/v1
kind: Rubric
metadata:
  name: my-support-agent
  version: 0.1.0
imports:
  - file://./registry/support-agent/v1/rubric.yaml
modes:
  - id: refund-without-confirmation   # override: our billing tools
    severity: critical
    detection:
      type: tool_call
      tool_calls: [stripe_refund, wallet_credit]
```

Try any of them against the bundled synthetic traces without
credentials:

```bash
docket demo --rubric rubrics/registry/support-agent/v1/rubric.yaml
```

(The demo's scripted judge only knows the builtin `agents/v1` judge
modes — registry-specific judge modes score negative until you pass
`--live`. Deterministic modes run for real either way.)

## Contributing a rubric

Registry rubrics are the highest-leverage contribution to this
project. The bar and the step-by-step path live in
[CONTRIBUTING.md](../../CONTRIBUTING.md#contributing-a-rubric) —
in short: a real use case, every judge mode exampled both ways,
synthetic data only, README with tuning knobs, `docket validate` +
`docket self-test` green.

# multi-agent-supervisor

Failure-mode rubric for **supervisor/specialist agent topologies**.
Mostly composition: it merges the three builtin taxonomies that cover
coordination — `mast/v1` (the research-derived MAST failure modes),
`routing/v1`, and `multi-agent/v1` — and adds the two supervisor
failures the builtins leave out.

```bash
docket run ... --rubric rubrics/registry/multi-agent-supervisor/v1/rubric.yaml
```

## What it catches

| source | modes | detector mix |
|---|---|---|
| own | `silent-subagent-failure` (critical), `token-budget-blowout` (medium) | 1 judge + 1 metric |
| `mast/v1` | step repetition, history loss, termination unawareness, conversation reset, missing clarification, ignored agent input, action-reasoning mismatch | 7 judges |
| `routing/v1` | wrong-skill routing, capability mismatch, dead-end transfer, oscillation | mixed |
| `multi-agent/v1` | handoff context loss, conflicting instructions, role drift, shared-memory corruption | mixed |

This is the **widest** registry rubric (17 modes). For production
windows, budget with `--dry-run` and consider `--sample` — or start
from a narrower import list and grow.

## The calibration story

The `mast/v1` judges this rubric imports are the ones docket
calibrates against **MAD**, the MAST authors' human-labelled dataset:
per-mode precision/recall/F1 and the prompt-tuning loop live in
[`docs/calibration/`](../../../docs/calibration/) and
[`docs/tuning-mast-judges.md`](../../../docs/tuning-mast-judges.md).
If you tune a mast judge prompt for your own topology, re-declare the
mode id in *your* rubric rather than editing the builtin — the version
label on annotations keeps old and new classifications separable.

## Tuning knobs

- **`silent-subagent-failure`** assumes specialist errors are visible
  in the trace (tool errors, error-status spans, or failure text). If
  your orchestrator retries internally and only logs the last attempt,
  the judge under-fires — instrument retries or accept the blind spot.
- **`token-budget-blowout` threshold (150k)**: set to ~5× your P95
  episode's `total_tokens`. It's a smoke alarm, not a diagnosis — the
  MAST modes usually name the underlying loop.
- **Ratchet path**: `silent-subagent-failure` at `critical` is the one
  to trust first; its false-positive mode (agent *did* caveat, judge
  missed it) is rare and cheap to confirm from the draft's excerpt.

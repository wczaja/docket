# Rubric DSL reference (`agent-triage.dev/v1`)

A **rubric** is a YAML document declaring the failure modes agent-triage
classifies traces against, plus how positives are clustered and what
happens to the resulting issue drafts. This page is the normative
reference for `apiVersion: agent-triage.dev/v1`; it matches the Pydantic
models in `agent_triage/rubric/spec.py` and the JSON Schema in
`agent_triage/rubric/schemas/v1.json` (both are enforced — you get the
same errors from `agent-triage validate` regardless of tooling).

A complete worked example lives at
[`rubrics/examples/sample-support-agent.yaml`](../rubrics/examples/sample-support-agent.yaml).

## Top-level structure

```yaml
apiVersion: agent-triage.dev/v1   # required; exactly this for v1
kind: Rubric                      # required; exactly "Rubric"
metadata:                         # required
  name: my-prod-agents            # required
  version: 1.0.0                  # required; semver
  authors: [you@example.com]      # optional
  description: |                  # optional
    What this taxonomy covers.
imports: []                       # optional; see "Imports"
modes: []                         # see "Modes"; may be empty only if imports merge to >= 1 mode
clustering: {}                    # optional; see "Clustering"
triage: {}                        # optional; see "Triage behavior"
```

Unknown keys are rejected everywhere (`extra: forbid`) — a typo'd field
name is a validation error, never silently ignored.

## Modes

One mode = one failure-mode definition.

```yaml
modes:
  - id: hallucinated-pricing      # required; ^[a-z0-9][a-z0-9-]*$; unique after import merge
    name: Hallucinated Pricing    # optional display name
    description: |                # optional but recommended — the drafter quotes it
      Agent stated a price not present in retrieved context.
    severity: critical            # required: critical | high | medium | low
    detection: { ... }            # required; see "Detection types"
    examples: [ ... ]             # optional; see "Examples"
```

## Detection types

`detection.type` selects the mechanism; each type reads its own subset of
fields. Supplying a field the type doesn't use is allowed by the schema
but ignored; omitting a required one fails at evaluation with a
`DetectionError` naming the mode.

### `llm_judge`

A structured LLM call per trace. The only detection type that costs
money and the only one whose input is PII-redacted before leaving the
process.

```yaml
detection:
  type: llm_judge
  prompt: |                       # required: instructions for the judge
    Given the trace, ...
  output_schema:                  # required: JSON Schema (draft 2020-12)
    type: object                  # MUST be type: object at the root
    required: [positive]          # MUST require a boolean `positive`
    properties:
      positive: {type: boolean}
      confidence: {type: number, minimum: 0, maximum: 1}
      excerpt: {type: [string, "null"]}
  model: anthropic:claude-sonnet-4-6   # optional per-mode override of the run's default
```

The `output_schema` is passed to the provider's native structured-output
mechanism without translation, so anything the provider's JSON Schema
dialect supports is usable. Beyond `positive` (the verdict) the schema is
yours; `confidence` (0–1) feeds cluster-representative selection and
evidence fields (e.g. `excerpt`) feed clustering embeddings and issue
drafts. The default judge model is the run's `--provider`/`--model`
(fast-and-cheap by default); reserve per-mode `model:` overrides for
modes that genuinely need a reasoning model.

### `regex`

Pure Python; free. Positive iff the pattern matches the trace's
projected text (all LLM messages, tool i/o, and retrieved documents,
unredacted — nothing leaves the process).

```yaml
detection:
  type: regex
  pattern: "(?i)here is my system prompt"
```

### `tool_call`

Positive iff **any** of the listed tool names was called in the trace
(v1 semantic is any-of; sequence/absence variants are reserved for a
future `mode:` knob).

```yaml
detection:
  type: tool_call
  tool_calls: [process_refund, drop_table]
```

### `metric_threshold`

Compares a numeric trace metric against a constant. Metrics available in
v1 (derived from the OpenInference spans):

| metric | meaning |
|---|---|
| `span_count` | number of spans in the trace |
| `total_tokens` | sum of LLM span token counts |
| `error_count` | spans with ERROR status |
| `latency_ms` | wall time from first span start to last span end |

```yaml
detection:
  type: metric_threshold
  metric: span_count
  threshold: 50
  operator: ">"        # one of == != < <= > >=
```

A metric missing from the trace is a `DetectionError` (the trace is
marked errored for that mode), not a silent negative.

### `composite`

Boolean combination of sub-detections. Operands are full `detection`
objects and nest arbitrarily.

```yaml
detection:
  type: composite
  operator: and        # and | or
  operands:
    - type: tool_call
      tool_calls: [handoff_to_specialist]
    - type: llm_judge
      prompt: Did the handoff fail?
      output_schema:
        type: object
        required: [positive]
        properties:
          positive: {type: boolean}
```

## Examples

Examples make a mode self-testing: `agent-triage self-test` runs each
one through the mode's detector and asserts the expected verdict. They
are optional but strongly recommended for `llm_judge` modes — they're
your regression suite against judge-prompt drift. (In v1.0, self-test
exercises `llm_judge` modes only; examples on deterministic modes are
reported as skipped.)

```yaml
examples:
  - trace_excerpt: "The Pro plan is $42/month with a $10 launch discount."
    context: "Pricing page lists Pro at $42/month; no discount mentioned."
    expected: positive            # positive | negative
  - trace_excerpt: "The Pro plan is $42/month."
    context: "Pricing page lists Pro at $42/month."
    expected: negative
```

## Imports

Rubrics compose. Two schemes in v1.0 (no network in the validation
path; `https://` and `registry://` are reserved for later versions):

```yaml
imports:
  - agent-triage.dev/builtin/agents/v1      # packaged builtin
  - file://./shared/common.yaml             # absolute, or relative to this file
```

**Merge semantics.** Imports are resolved depth-first, left to right.
Modes merge by `id`: later imports override earlier ones, and the
importing rubric's own modes override everything — so you can pin down a
builtin mode by redeclaring its `id` with your own detection or
severity. `metadata`, `clustering`, and `triage` are **not** merged; the
importing rubric's values win outright. Import cycles are detected and
rejected with the cycle path in the error.

A rubric with no own `modes:` and no `imports:` is invalid; a rubric
with only imports is valid as long as the merged result has at least one
mode.

**Builtins shipped with the package:**

| URI | Covers |
|---|---|
| `agent-triage.dev/builtin/agents/v1` | generic agent failures (hallucination, loops, premature termination, unsafe tools, prompt leakage, bad handoff) |
| `agent-triage.dev/builtin/rag/v1` | retrieval failures |
| `agent-triage.dev/builtin/routing/v1` | supervisor/router failures |
| `agent-triage.dev/builtin/multi-agent/v1` | orchestration failures |

## Clustering

How positive classifications group into issues, per mode:

```yaml
clustering:
  strategy: per_mode_embedding    # only strategy in v1
  embedding_model: text-embedding-3-small
  similarity_threshold: 0.82      # 0..1
  min_cluster_size: 3             # >= 2; clusters smaller than this don't draft issues
```

Evidence excerpts from each positive classification are embedded and
HDBSCAN-clustered per mode; each cluster's highest-confidence trace
becomes the representative quoted in the draft.

## Triage behavior

```yaml
triage:
  auto_post_threshold: never      # critical | high | medium | low | never
  default_severity_to_tracker:    # map rubric severity -> tracker priority label
    critical: P1
    high: P2
    medium: P3
    low: P4
```

`auto_post_threshold` semantics: any value other than `never` auto-posts
clusters whose mode severity is at or above the threshold (`high` posts
critical+high; `low` posts everything). `never` — the default — posts
nothing; drafts queue locally for `--review`. The CLI flag
`--auto-post-threshold` overrides the rubric/config value per run.

## Validation rules (summary)

`agent-triage validate <source>` enforces, in one pass:

- `apiVersion` is exactly `agent-triage.dev/v1` (anything else is
  refused with the supported list — design §3.6's migration message)
- `kind: Rubric`; `metadata.name` and `metadata.version` present
- every mode `id` matches `^[a-z0-9][a-z0-9-]*$` and is unique
  (uniqueness is re-checked after import merging)
- `severity` ∈ {critical, high, medium, low}
- `llm_judge` detections have a `prompt` and an `output_schema` that is
  a valid draft 2020-12 JSON Schema with `type: object` at the root and
  a required boolean `positive`
- `composite` detections have `operator` ∈ {and, or} and at least one
  operand
- `triage.auto_post_threshold` ∈ {critical, high, medium, low, never}
- no unknown keys anywhere

## Versioning

Rubrics are semver'd via `metadata.version`. Breaking changes (removing
a mode, changing a mode's meaning, tightening an `output_schema`)
require a major bump — annotations and tracker labels carry
`rubric:<name>@<version>`, so a version bump cleanly partitions old and
new classifications. The runtime refuses any `apiVersion` it doesn't
support rather than guessing.

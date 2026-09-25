# Concepts

The vocabulary used across the CLI, the docs, and the code. Skim this
once and the rest of the documentation reads cleanly.

## The cast

- **User's agent** — the LLM agent whose traces are analyzed. docket
  never runs it, instruments it, or stores its data; it reads the traces
  the agent already emits to an observability backend.
- **Triage agent** — docket's own pipeline (and, under `--agent`,
  its planning harness). When docs say "the agent" ambiguously, they
  mean this one and will say so explicitly.

## Trace-side concepts

- **Trace** — one end-to-end interaction of the user's agent, as a tree
  of OpenInference spans (LLM calls, tool calls, retrievals). Traces
  live in *your* backend — Phoenix, Langfuse, or LangSmith; docket
  fetches them read-only and normalizes every backend's shape to the
  same OpenInference model before anything else sees them.
- **Window** — the `[since, until]` time range a run operates on.
  `run` takes it from flags; `serve` tiles consecutive windows
  automatically.
- **Annotation** — a classification written *back to the backend*
  (opt-in via `--annotate`), keyed by
  `(trace_id, run_id, rubric_version, mode_id)`. Annotations are the
  only trace-side state docket creates, which is what makes the
  runtime stateless: re-runs upsert rather than duplicate, and
  `--checkpoint` resumability is just "skip traces already annotated for
  this run_id".

## Rubric-side concepts

- **Rubric** — the YAML failure-mode taxonomy a run classifies against.
  Versioned, composable via imports, validated before any trace is
  fetched. See the [DSL reference](rubric-spec.md).
- **Mode** — one failure-mode definition inside a rubric: an `id`, a
  `severity`, a `detection`, and optional self-test `examples`.
- **Detection** — the evaluation mechanism for a mode: `llm_judge`
  (structured LLM call), `regex`, `tool_call`, `metric_threshold` (all
  pure Python, free), or `composite` (and/or combinations).
- **Classification** — the result of evaluating one mode against one
  trace: positive/negative, confidence, evidence excerpt, or an error
  marker if the detector failed after retries.

## Issue-side concepts

- **Cluster** — positives for one mode, grouped by embedding similarity
  (HDBSCAN), so fifty instances of the same failure become one issue,
  not fifty. The highest-confidence member is the cluster's
  **representative trace**.
- **Draft** — the issue generated for a qualifying cluster (title, body
  with representative evidence and frequency stats, severity-mapped
  priority, labels). Drafts queue locally by default — *drafted is not
  posted*.
- **Provenance** — machine-readable identity carried by every posted
  issue, in two places: tracker labels (`docket`, `mode:<id>`,
  `rubric:<name>@<version>`) and an HTML-comment JSON block at the end
  of the body. Both exist for **dedup**: the next run finds the existing
  issue, comments with only the new trace IDs if the cluster grew, and
  skips silently if nothing changed. The block doubles as the dedup
  **ledger**: before commenting, docket records the cluster and trace
  IDs it is about to report there (the only edit it makes to an existing
  issue's body; a comment that then fails is reported, not retried), so
  re-running any window never repeats a comment.
- **Lineage** — the open issue docket keeps per failure mode per rubric
  version. Consecutive windows share no traces (`serve` tiles them
  exactly), so a failure that recurs in the next window can't be matched
  by cluster or trace overlap; dedup matches it by lineage instead and
  comments on the open issue ("Recurred in window …", plus that window's
  trace IDs) rather than filing a duplicate. Other clusters of the same
  mode from one window land on the same issue. Closing the issue ends
  the lineage, so the next occurrence opens a fresh issue: that's how a
  regression surfaces. If several open issues share a lineage (filed
  before lineage dedup existed, or posted from the queue by hand),
  recurrences go to the oldest. Only issues in the tracker anchor a
  lineage; drafts still sitting in the local queue don't. The policy is
  pluggable in the [Python API](python-api.md) (`lineage_matcher`),
  including `None` for cluster-only dedup.
- **Dedup safety** — when the tracker's open-issue listing is cut off at
  its page ceiling, "nothing matches" is unproven, so a draft that
  matched nothing is queued as `needs_create` even if its severity would
  auto-post. Matches found in a truncated listing are still used.
- **Review** — the human-in-the-loop step (`--review`): each
  would-be-new issue opens in `$EDITOR` for accept/edit/reject before
  posting. Auto-posting exists but is opt-in by severity
  (`auto_post_threshold`).

## Run mechanics

- **run_id** — `sha256(backend|rubric@version|since|until)[:16]`,
  computed deterministically from the inputs that define a run. Same
  inputs → same run_id → idempotent re-runs. `--run-id` overrides it for
  backfills.
- **Budget gates** — `max_traces_per_run` (default 1000) and optional
  `max_estimated_cost_usd` abort a too-large run *before any trace is
  fetched*; silent truncation is forbidden everywhere. `--dry-run`
  evaluates both gates and prices the run without executing it;
  `--sample N` (uniform / errors-only / stratified) bounds work on
  windows too large to enumerate.
- **Report** — every run ends with a markdown report: per-mode
  frequencies, clusters formed, drafts created/posted/queued, skips and
  errors, and a truncation banner if any backend listing hit its page
  ceiling.

## Where state lives

Nowhere new — that's the design's core constraint. Classifications live
in your backend as annotations; issue identity and the dedup ledger
live in your tracker as labels + provenance; drafts awaiting review
live as files under `~/.docket/`. There is no docket database, so there
is nothing to operate, back up, or migrate, and any number of runs (or
`serve` daemons on disjoint windows) can coexist against the same
backend.

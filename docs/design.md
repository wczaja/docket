# docket: Design and Implementation Plan
**A cross-platform, observability-agnostic agent triage runtime configured by a composable failure-mode DSL.**

*Version 0.1 — Working design draft*
*Owner: Will*
*Status: Pre-implementation, design review phase*

---

## 0. How to use this document

This document is the implementation spec for docket. Sections are ordered by implementation dependency, and each phase has explicit, machine-verifiable acceptance criteria — work should not advance to the next phase until the current phase's criteria are demonstrably met. Read it end to end before starting any phase, and propose an implementation plan for a phase before writing its code.

The document uses `MUST`, `SHOULD`, and `MAY` per RFC 2119 conventions. `MUST` requirements are non-negotiable; deviations require explicit maintainer approval before merge.

---

## 1. Project Mission and Scope

### 1.1 Mission

docket provides an open-source, observability-platform-agnostic agent triage runtime that:

1. Pulls LLM agent traces from supported backends (Phoenix, Langfuse, LangSmith in v1.0; OpenInference compatibility means other backends can be added post-v1.0)
2. Classifies traces against a composable, declarative failure-mode taxonomy (the "rubric")
3. Clusters traces by failure mode and deduplicates issues
4. Writes annotations and issues back to the observability platform of origin and/or to external trackers (Jira, Linear, GitHub Issues)
5. Emits trace-derived candidate eval cases for downstream regression suites

### 1.2 What this is *not*

- A new observability backend (LangSmith, Phoenix, and Langfuse already exist and are excellent)
- A new eval framework (DeepEval, Latitude/GEPA, agentevals already exist)
- A replacement for human judgment — humans validate the agent's triage before issues are created in trackers
- A general-purpose data pipeline tool — scope is bounded to LLM agent traces

### 1.3 Why this exists

A recent industry keynote described an internal triage agent pattern that does not yet exist in OSS:

- A Deep Agent runs on a cron
- Pulls failing traces from observability
- Classifies real-issue vs. false-negative
- Clusters by failure mode using a domain-configured taxonomy
- Writes back annotations and creates issues in trackers
- Hands eval-generation off to a downstream agent

Adjacent commercial offerings exist but each is tied to its own platform:
- Latitude's GEPA auto-generates evals from annotated traces but is annotation-driven and Latitude-specific
- Galileo's Insights Engine clusters but doesn't write to external trackers
- LangSmith Insights clusters within LangSmith only
- Traceloop's MCP server reads OTel traces but doesn't triage

The gap: a runtime that works across all of these, configured by a portable rubric spec.

### 1.4 In-scope for v1.0

- Python 3.11+ runtime
- LangGraph + Deep Agents harness for the triage agent itself
- MCP servers for trace backends (read) and issue trackers (write)
- YAML-based failure-mode taxonomy DSL ("the rubric spec")
- Three trace backend adapters: Phoenix, Langfuse, LangSmith (covers the OSS-dominant + commercial-default cases)
- Three tracker adapters: Jira, Linear, GitHub Issues
- LLM-as-judge classifier with structured output validation
- Embedding-based clustering with configurable similarity thresholds
- CLI for one-shot runs (`docket run --since=24h`)
- Scheduled runs via cron or CI (see `docs/triage-as-ci.md`); a resident
  daemon mode (`docket serve`) is deferred to v1.1 — one-shot runs
  on a scheduler cover the v1.0 operating model without a long-lived
  process to operate
- Reference synthetic taxonomy as example
- Latitude integration is explicitly **out of scope for v1.0** and pulled into v1.1

### 1.5 Non-goals

- Proprietary observability backends (Datadog, New Relic) — community contributions welcome post-v1.0
- Auto-fix PR generation (downstream agent, separate project)
- A web UI (CLI + writing to existing trackers is the v1 surface)
- A trace store (we read from existing stores, never store traces ourselves)

---

## 2. Architecture Overview

### 2.1 System diagram (text form)

```
                    ┌──────────────────────────────────────┐
                    │  Scheduler (cron / serve / one-shot) │
                    └────────────────┬─────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Triage Agent       │
                          │  (LangGraph +       │
                          │   Deep Agents)      │
                          │                     │
                          │  Subagents:         │
                          │   - Classifier      │
                          │   - Clusterer       │
                          │   - Annotator       │
                          │   - Issue Drafter   │
                          └──┬───────────────┬──┘
                             │               │
              ┌──────────────┘               └──────────────┐
              │                                             │
      ┌───────▼─────────┐                          ┌────────▼─────────┐
      │ Trace Backend   │                          │ Tracker Adapter  │
      │ MCP Servers     │                          │ MCP Servers      │
      │                 │                          │                  │
      │ - Phoenix       │                          │ - Jira           │
      │ - Langfuse      │                          │ - Linear         │
      │ - LangSmith     │                          │ - GitHub Issues  │
      └───────┬─────────┘                          └──────────────────┘
              │
              │ (read OpenInference traces)
              │
      ┌───────▼─────────────────────────────────────┐
      │ Observability Backends (user-owned)         │
      │ Phoenix, Langfuse, LangSmith, Latitude, etc.│
      └─────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────┐
      │ Rubric Registry (filesystem + remote)        │
      │  - Built-in rubrics (rag/, agents/, etc.)    │
      │  - User-defined rubrics                      │
      └──────────────────────────────────────────────┘
```

### 2.2 Key architectural decisions

1. **OpenInference as canonical trace schema.** Every backend adapter normalizes to OpenInference semantic conventions before traces hit the triage agent. The agent's prompts and classifier never see backend-specific shapes.

2. **MCP for both read (trace backends) and write (issue trackers).** Each adapter is an MCP server (stdio or HTTP). The triage agent discovers and invokes tools dynamically; new backends/trackers can be added without modifying the agent.

3. **Deep Agents harness for the triage agent.** The triage workflow is a Deep Agent with subagents for the four core concerns (classify, cluster, annotate, draft). Virtual filesystem for inter-step state. This gives us planning, context management, and delegation out of the box; we don't reimplement them.

4. **YAML rubric DSL with JSON Schema validation.** Failure modes are declared in YAML and validated against a JSON Schema. Each `llm_judge` mode's `output_schema` is itself a JSON Schema (draft 2020-12), passed through directly to the LLM provider's native structured-output mechanism without translation. A registry pattern (think `pip` for rubrics) supports composable + community-shared taxonomies.

5. **Stateless triage agent, state in observability backend.** docket never owns durable storage beyond its config. All state (which traces were processed, what issues were created, what annotations exist) lives in the user's observability backend and tracker, addressed by trace ID. This is the cross-platform property — we don't fork the data.

6. **LLM-as-judge for classification, embeddings for clustering.** Per-trace classification needs reasoning over multi-turn context; pure embeddings are too lossy. Cross-trace clustering benefits from cheap embedding similarity. Different tools for different jobs.

7. **Human-in-the-loop is a first-class state.** Drafted issues are *drafted*, not posted, unless a configurable auto-post threshold is met. Default: human reviews before posting to tracker. This matches the production pattern described in the keynote that motivated this project.

8. **Async I/O throughout.** All network and disk I/O uses `async def` + `httpx.AsyncClient`. Subagents and adapters are coroutines composed by the agent harness. Synchronous wrappers exist only at the CLI entry point and in tests where blocking is intentional.

9. **Secrets never traverse logs, annotations, or the virtual filesystem.** API keys and tokens (LLM providers, trace backends, trackers) are read from environment variables, held in memory only by the adapter or provider that needs them, and never serialized into logs, annotations, the agent's virtual filesystem, or drafted issue bodies. Missing or invalid credentials abort the run at startup before any I/O, naming the missing variable in the error.

### 2.3 Why these choices over alternatives

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Trace schema | OpenInference | OTel GenAI conventions | Richer LLM-specific attributes; broader instrumentation coverage; Phoenix/Langfuse already adopting |
| Agent harness | Deep Agents | Bare LangGraph | Virtual filesystem + planning built-in; saves ~2-3 weeks of harness work |
| Adapter protocol | MCP | Custom plugin API | Standardizes external integration; lets users plug other MCP clients into their backends |
| DSL format | YAML + JSON Schema | TOML / Pydantic-only | Familiar to ops/SRE audience; JSON Schema is widely understood and tooling-rich |
| Classification | LLM-as-judge | Pure embedding-based | Multi-turn agent failures need reasoning; embedding similarity misses semantic intent |
| Clustering | Embedding + HDBSCAN | LLM-only clustering | Cost-prohibitive for production volumes; embedding clustering is well-understood |
| State storage | Stateless | SQLite/Postgres | Lets us ship without operating a database; trace IDs are the natural keys |

---

## 3. The Rubric DSL Specification

### 3.1 Goals

The rubric DSL MUST be:
- Readable by domain experts who are not Python programmers
- Composable (rubrics can import other rubrics)
- Versioned (rubrics evolve; old versions must remain consumable)
- Validatable (a rubric file is either valid or it's not; no silent failures)
- Compilable to a structured-output schema for the LLM classifier

### 3.2 Top-level structure

```yaml
# rubric.yaml
apiVersion: docket.dev/v1
kind: Rubric
metadata:
  name: sample-support-docket
  version: 0.1.0
  authors:
    - maintainer@example.com
  description: |
    Failure-mode taxonomy for a sample customer-support agent.
    Synthetic example for documentation purposes.

# Optional: import other rubrics. Modes are merged; later imports override earlier.
imports:
  - docket.dev/builtin/agents/v1
  - docket.dev/builtin/rag/v1
  - file://./shared/common.yaml

# The taxonomy itself.
modes:
  - id: hallucinated-pricing
    name: Hallucinated Pricing
    description: |
      Agent stated a price or discount that does not appear in retrieved
      context and is not a known constant.
    severity: critical
    detection:
      type: llm_judge
      prompt: |
        Given the trace's retrieval results and the agent's final response,
        identify any prices, discounts, or monetary amounts in the response
        that are not present in the retrieval results or stated in the
        system prompt. If found, return a positive classification with the
        offending excerpt; otherwise negative.
      output_schema:
        type: object
        required: [positive, confidence]
        properties:
          positive: {type: boolean}
          excerpt: {type: [string, "null"]}
          confidence: {type: number, minimum: 0, maximum: 1}
    examples:
      - trace_excerpt: "The Pro plan is $42/month with a $10 launch discount."
        context: "Pricing page lists Pro at $42/month base; no launch discount mentioned."
        expected: positive
      - trace_excerpt: "The Pro plan is $42/month."
        context: "Pricing page lists Pro at $42/month."
        expected: negative

  - id: cross-domain-misrouting
    name: Cross-Domain Misrouting
    description: |
      Supervisor routed a query to a specialist that does not own the
      relevant data domain.
    severity: high
    detection:
      type: llm_judge
      prompt: |
        Examine the supervisor's routing decision and the specialist
        agent's tool calls. If the specialist had to escalate, hand off,
        or fail because it lacked access to the relevant data domain,
        classify as positive.
      output_schema:
        type: object
        required: [positive, confidence]
        properties:
          positive: {type: boolean}
          intended_specialist: {type: [string, "null"]}
          confidence: {type: number, minimum: 0, maximum: 1}

  - id: false-negative-thumbs-down
    name: False-Negative User Feedback
    description: |
      User submitted negative feedback (thumbs-down) but the trace shows
      the agent's behavior was actually correct given available context.
    severity: low
    detection:
      type: llm_judge
      prompt: |
        The user signaled dissatisfaction. Review the trace and determine
        whether the dissatisfaction reflects an agent failure or a user
        expectation that was outside the agent's scope.
      output_schema:
        type: object
        required: [positive, reason]
        properties:
          positive: {type: boolean}
          reason: {type: string}

# Cluster behavior — how to group classified traces into issues.
clustering:
  strategy: per_mode_embedding
  embedding_model: text-embedding-3-small
  similarity_threshold: 0.82
  min_cluster_size: 3

# Triage behavior — what to do with clusters once classified.
triage:
  auto_post_threshold: never  # one of: critical, high, medium, low, never
  # Semantics: any value other than `never` auto-posts clusters whose mode
  # severity is at or above the threshold. `high` posts critical+high;
  # `low` posts everything; `never` (default) posts nothing.
  default_severity_to_tracker:
    critical: P1
    high: P2
    medium: P3
    low: P4
```

**Supported import schemes (v1.0):**

- `file://<path>` — absolute, or relative to the importing rubric file
- `docket.dev/builtin/<rubric>/<version>` — resolved against packaged data via `importlib.resources`

`https://` and `registry://` schemes are reserved for v1.1+; v1.0 has no network in the validation path.

### 3.3 Built-in rubrics

The project SHOULD ship with a starter set under `docket.dev/builtin/`:

- `agents/v1`: generic LLM agent failure modes (hallucination, tool misuse, infinite loop, premature termination)
- `rag/v1`: retrieval failure modes (missing context, wrong context, context contradiction)
- `routing/v1`: supervisor/router failure modes (misroute, oscillation, dead-end)
- `multi-agent/v1`: orchestration failure modes (handoff loss, coordination failure, thundering-herd tool calls, single-agent poisoning of the group)

These MUST be tested with both real and synthetic traces before v1.0.

### 3.4 Detection types

v1.0 supports:

- `llm_judge`: prompt + structured output schema, evaluated against the trace
- `regex`: pattern match against agent response
- `tool_call`: presence/absence/sequence of specific tool calls
- `metric_threshold`: numeric threshold against OpenInference span attributes (latency, token count, etc.)
- `composite`: AND/OR of the above

Each detection type MUST have a corresponding Python implementation in `docket/detectors/`.

### 3.5 Validation rules

- Every mode MUST have a unique `id` within the rubric (after import merging)
- Every mode MUST have a `severity` from the enum `{critical, high, medium, low}`
- Every `llm_judge` detection MUST have an `output_schema` that is a valid JSON Schema (draft 2020-12), MUST have `type: object` at the root, and MUST require a boolean `positive` property
- `triage.auto_post_threshold` MUST be one of `{critical, high, medium, low, never}`
- `examples` are RECOMMENDED but not required; if present they MUST be exercised by the self-test detector (positive examples MUST classify positive; negative examples MUST classify negative)
- A rubric with no own `modes:` and no `imports:` is invalid. A rubric with only `imports:` is valid as long as the merged result contains at least one mode.

### 3.6 Versioning

Rubrics use semver. Breaking changes to a rubric require a major version bump. The runtime MUST refuse to load a rubric whose `apiVersion` doesn't match what it supports, with a clear migration message.

---

## 4. The Triage Agent

### 4.1 Top-level workflow

```
1. Pull traces from the configured backend within the time window
2. Filter to candidates (failed traces, low-score traces, thumbs-down traces)
3. For each candidate:
   a. Run classifier against rubric → set of (mode_id, confidence, evidence)
   b. Annotate the trace in the backend with classifications above threshold
4. Cluster classified traces per mode_id
5. For each cluster:
   a. Check against existing tracker issues (dedup)
   b. If new: draft an issue with representative trace + frequency + severity
   c. If auto_post_threshold met: post to tracker; otherwise queue for review
6. Emit summary report
```

### 4.2 Execution modes

docket ships **two execution modes** that share the same six pipeline stages (`list_traces`, `classify_traces`, `annotate_classifications`, `cluster_classifications`, `draft_issues`, `write_report`) and the same subagent implementations (§4.3). The modes differ only in **who decides the order of operations**: deterministic Python, or an LLM-driven planner.

| Mode | Entry | Orchestration | Intended use |
|------|-------|---------------|--------------|
| Deterministic pipeline (default) | `docket run` | Stages execute in fixed order from `run_triage_pipeline` | Batch / cron / CI. Anywhere SLOs, cost forecasting, reproducibility, and stack-trace debuggability matter. **This is the production execution model.** |
| Deep Agents harness | `docket run --agent` | The six stages are exposed as LangChain tools; a top-level LLM (default Haiku) plans the calls; deepagents virtual filesystem holds inter-stage artifacts | Exploratory / debugging runs today. Substrate for future interactive surfaces (chat-driven triage, incident investigation, rubric authoring) — see §4.2.2. |

Both modes:

- Produce identical clusters, drafts, and reports on the same input (the agent harness wraps the same subagents).
- Use the same `run_id` (deterministic — see below) and the same annotation idempotency story.
- Are driven from the same CLI; the agent mode is an opt-in flag, not a separate command.

The `run_id` is computed deterministically from the inputs that define a run, in both modes:

```
run_id = sha256(f"{backend_id}|{rubric_id}@{rubric_version}|{window_start_iso}|{window_end_iso}").hexdigest()[:16]
```

Two invocations with identical backend, rubric, and time window produce the same `run_id` — so re-running after a transient failure overwrites the prior run's annotations rather than creating a parallel set. The `--run-id <id>` CLI flag overrides the computed value for operator-driven cases such as backfills or replays where explicit grouping or separation is desired.

#### 4.2.1 Deterministic pipeline (default)

The default `docket run` path is `run_triage_pipeline` in `docket/agent/triage.py`. The six stages execute in fixed order with plain Python control flow. Stages share data through typed Pydantic objects passed in-process. No LLM orchestrator sits above the pipeline; the only LLM calls are the ones the detectors and drafter make.

This is the production execution model because:

- **Predictable cost.** Token counts are bounded by `len(traces) × len(modes)` for classification plus one call per cluster for drafting. No planner round-trips.
- **Reproducible.** Same inputs → same control flow → same outputs (within LLM nondeterminism of the detectors themselves, which is independent of orchestration mode).
- **Debuggable.** Failures surface as Python stack traces in the stage that failed, not as "the agent decided to skip step N for some reason."
- **SLO-friendly.** "The pipeline runs N stages in this order" is a much easier operational claim than "the planning model usually picks the right next tool."

Operators choose this mode for nightly cron, CI integration, batch backfills, and anywhere wall-clock-time or cost forecasting matters.

#### 4.2.2 Deep Agents harness (`--agent` opt-in)

The `--agent` flag routes the same workflow through `docket/agent/deep_agent.py`, which wraps each pipeline stage as a LangChain `@tool` and hands the toolset to `deepagents.create_deep_agent`. A top-level planning LLM reads a system prompt describing the workflow and calls the tools in sequence; intermediate state lands in the deepagents virtual filesystem:

- `/traces/manifest.json` — trace IDs in the current window
- `/classifications/summary.json` — per-mode positive counts and errors
- `/clusters/summary.json` — cluster count and per-mode sizes
- `/drafts/titles.json` — draft cluster titles
- `/annotations/summary.json` — annotation writeback summary (when `--annotate`)
- `/report.md` — human-readable run summary; the CLI extracts this at the end

The vfs is intentionally non-durable: a run starts clean, failures abort and retry rather than partially recover. Idempotency comes from the observability backend's `(run_id, rubric_version, mode_id, trace_id)` annotation key.

**Why we ship this mode.** The deepagents harness is the architectural commitment for *future* triage workflows that don't fit a fixed pipeline — interactive operator-driven triage, hypothesis-driven incident investigation, iterative rubric authoring. Keeping both modes in lockstep over the same six stages means investment in subagents and detectors benefits both surfaces, and the planning substrate is already there when those surfaces land.

**Honest status in v1.0.** Today the harness runs the same six-stage workflow the deterministic pipeline runs, with LLM planning bolted on. The system prompt is rigid; the only adaptive behavior is "skip a downstream stage if the upstream stage returned zero results." The tools and entry points needed for the interactive use cases above are **not yet implemented**:

| Use case | Status | Missing |
|---|---|---|
| Interactive on-call triage (Slack / chat surface) | Not implemented | Chat-loop entry point or MCP-server-of-its-own; flexible system prompt; "what's new vs. baseline" tools. Phase 15. |
| Incident investigation (freeform queries, co-occurrence) | Not implemented | Parameterized query tools (`query_traces_filtered`, `query_by_mode`); ability to re-pivot the time window mid-run. Phase 14. |
| Rubric authoring loop | Not implemented | Rubric edit / reload / dry-run tools surfaced to the harness; `self-test` is a CLI command but not an agent tool. Phase 15. |
| Cross-backend composition | Not implemented | Multi-backend tool dispatcher; the harness currently binds a single backend at construction. Phase 14. |
| Per-trace deep dives (re-classify, compare) | Not implemented | Per-trace re-run tool, model-swap mid-run, similarity-search tool. Phase 14. |
| Long-tail composite failure modes | Partial | `composite` detection exists in the rubric DSL but executes inside the classifier subagent; the harness only sees `classify_traces` as one opaque tool. Phase 14. |

These extensions are tracked as Phase 14 (tool surface expansion) and Phase 15 (interactive surfaces) in §7.

Until those phases land, **the practical recommendation is to use the deterministic pipeline for production runs and reserve `--agent` for debugging the harness itself or demonstrating the deepagents integration**.

### 4.3 Subagents

**Classifier subagent**
- Input: one OpenInference trace, the rubric
- Output: list of `(mode_id, positive, confidence, evidence_excerpt)` tuples
- Implementation: for each mode in the rubric, evaluate the detection. For `llm_judge`, this is a structured LLM call with the rubric's prompt and output schema. For deterministic detectors (`regex`, `tool_call`, `metric_threshold`), this is pure Python.
- Concurrency: traces are classified in parallel (configurable, default 8 concurrent)
- Costs: this is the single largest cost driver. The classifier MUST support a budget mode that batches multiple traces into a single LLM call when the rubric is small enough.

**Clusterer subagent**
- Input: classified traces (those with at least one positive mode), the rubric
- Output: per-mode clusters
- Implementation: per-mode, embed the evidence excerpts; HDBSCAN cluster with the rubric's threshold; for each cluster, select the highest-confidence trace as the representative
- Concurrency: per-mode parallelizable

**Annotator subagent**
- Input: classified traces, backend adapter
- Output: side effect — annotations written to backend
- Implementation: invoke the backend MCP server's `annotate_trace` tool with `(trace_id, mode_id, confidence, evidence, run_id, rubric_version)`
- Idempotency: re-running with the same `(run_id, rubric_version)` MUST overwrite, not duplicate

**Issue Drafter subagent**
- Input: a cluster
- Output: drafted issue (title, body, severity, labels, provenance)
- Implementation: LLM generates draft from cluster representative + frequency stats + similar issue search results. Drafter appends an HTML-comment provenance block to the body — `<!-- docket:provenance {"rubric_version":"...","mode_id":"...","cluster_id":"...","representative_trace_id":"...","run_id":"..."} -->` — and assigns labels `docket`, `mode:<id>`, `rubric:<rubric-id>@<version>`.
- Dedup: before drafting, the agent queries the tracker for open issues carrying the matching `mode:<id>` and `rubric:<rubric-id>@<version>` labels and comments on the existing issue instead of creating a new one.

### 4.4 Failure handling

- **Credential failure**: missing or invalid API keys/tokens detected at startup abort the run before any I/O, naming the missing environment variable. No partial runs from credential issues.
- **Rubric validation failure**: abort immediately at startup, before fetching any traces.
- **Budget failure**: candidate trace count exceeding `max_traces_per_run` aborts the run with a clear error; the operator must partition explicitly. Silent truncation is forbidden.
- **Trace fetch failure**: log, skip the trace, continue. Report at end.
- **Classifier failure**: up to 3 attempts total with exponential backoff. After the 3rd failed attempt, classify the trace as `unprocessed` and skip clustering for it.
- **Backend write failure**: up to 5 attempts total. If still failing, abort the run with a clear error. We MUST NOT have a partial-write run.
- **Tracker write failure**: queue to the configured work directory (default `~/.docket/queued-issues/`) and report; don't fail the whole run.

### 4.5 Observability of the triage agent itself

The triage agent's own runs MUST emit OpenInference traces to a configurable backend (`instrumentation_backend` in `docket.yaml`; default: Phoenix, on the assumption that an OSS-first user is already running Phoenix locally). docket eats its own dogfood. Pointing `instrumentation_backend` at the same backend as the user's production traces keeps everything in one pane; pointing it at a separate instance keeps production observability clean. This also enables comparative evaluation later — we can validate the triage agent's correctness by examining its own traces.

---

## 5. Adapters

### 5.1 Trace backend adapters (read)

Each adapter is an MCP server exposing the following tools:

- `list_traces(since: datetime, until: datetime, filter: dict | None) -> list[trace_id]`
- `get_trace(trace_id: str) -> OpenInferenceTrace`
- `annotate_trace(trace_id: str, annotation: Annotation) -> None`
- `search_traces(query: str, k: int) -> list[trace_id]` (semantic search where supported; else stub)

**Phoenix adapter**
- Uses Phoenix's GraphQL API for `list_traces` and `get_trace`
- Annotations written via Phoenix's annotation API (introduced in Phoenix 5.x)
- Local Phoenix deployment supported for OSS-only workflows
- License compatibility: Phoenix is ELv2; our adapter is Apache 2.0 (matching the rest of the project) and only consumes Phoenix's public API, so the licenses are compatible

**Langfuse adapter**
- Uses Langfuse's public REST API directly via `httpx` (no SDK dependency)
- Annotations as Langfuse scores with metadata fields for `(run_id, rubric_version, mode_id)`; idempotent via a deterministic client-supplied score id
- Both Langfuse Cloud and self-hosted supported via environment-driven config

**LangSmith adapter**
- Uses LangSmith's public REST API directly via `httpx` (no SDK dependency)
- Annotations as LangSmith feedback objects, tagged with metadata; idempotent via a deterministic client-supplied feedback id
- Requires LangSmith API key

### 5.2 Tracker adapters (write)

Each adapter is an MCP server exposing:

- `list_open_issues(filter: dict | None) -> list[Issue]`
- `search_issues(query: str, k: int) -> list[Issue]`
- `create_issue(draft: IssueDraft) -> Issue`
- `update_issue(issue_id: str, patch: IssuePatch) -> Issue`
- `comment_on_issue(issue_id: str, comment: str) -> None`

**Jira, Linear, GitHub Issues** each implemented against their official APIs. All drafted issues carry provenance in two places:

- An HTML-comment block at the end of the issue body — `<!-- docket:provenance {JSON} -->` containing `rubric_version`, `mode_id`, `cluster_id`, `representative_trace_id`, `run_id`. Invisible to humans, parseable by the drafter on subsequent runs.
- Tracker labels `docket`, `mode:<id>`, `rubric:<rubric-id>@<version>` for queryable dedup without parsing issue bodies.

Tracker-native custom-field support is opt-in via per-tracker config in v1.1+ for users who prefer custom fields over labels (Jira and Linear only; GitHub Issues has no custom fields).

### 5.3 Adapter contract

Adapters MUST be MCP servers (stdio or HTTP). The docket runtime discovers them via configuration:

```yaml
# docket.yaml (runtime config)
trace_backend:
  type: mcp
  command: docket-adapter-phoenix
  env:
    PHOENIX_URL: http://localhost:6006

tracker:
  type: mcp
  command: docket-adapter-jira
  env:
    JIRA_HOST: example.atlassian.net
    JIRA_PROJECT: EXAMPLE
```

The runtime invokes these as standard MCP clients. No tight coupling.

**Implementation note — adapter / MCP server split.** Each backend or tracker integration is implemented as two layers:

1. A pure-Python adapter class under `docket/adapters/{trace,tracker}/` that owns the backend-specific logic — HTTP calls, normalization to OpenInference, retries, error mapping. The class is async, has no MCP dependency, and is unit-testable in-process.
2. A thin MCP server entry point under `docket/mcp_servers/` that instantiates the adapter class and exposes its methods as MCP tools (stdio or HTTP).

In v1.0 the bundled runtime drives the adapter classes in-process for the six first-party integrations; the MCP server binaries expose the same adapters to external MCP clients and to `docket.yaml` configs that point at third-party MCP servers. The split keeps integration logic test-friendly while keeping MCP as the architectural seam for everything outside the first-party set. Moving the bundled runtime itself onto MCP-only wiring is a post-v1.0 decision; any such move requires the MCP tool surface to cover the full `TraceBackend` contract first (today `mark_trace_processed` / `list_processed_trace_ids` are not exposed as tools).

---

## 6. Repository Layout

```
docket/
├── README.md
├── LICENSE                            # Apache 2.0
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml                     # uv-managed
├── .python-version                    # 3.11
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                     # ruff + mypy + pytest
│   │   ├── release.yml                # PyPI on tag
│   │   └── eval-rubrics.yml           # self-test builtin rubrics on PR
│   └── ISSUE_TEMPLATE/
├── docs/
│   ├── index.md
│   ├── concepts.md                    # rubric, mode, cluster, annotation
│   ├── rubric-spec.md                 # the DSL reference
│   ├── adapters.md
│   └── examples/
├── docket/
│   ├── __init__.py
│   ├── __main__.py                    # CLI entry: docket ...
│   ├── cli.py
│   ├── config.py                      # Pydantic, loads docket.yaml
│   ├── errors.py                      # typed exception hierarchy
│   ├── runtime.py                     # top-level orchestrator
│   ├── rubric/
│   │   ├── __init__.py
│   │   ├── spec.py                    # Pydantic models for the DSL
│   │   ├── loader.py                  # YAML loading + import resolution
│   │   ├── validator.py               # JSON Schema validation
│   │   ├── registry.py                # builtin + user rubric lookup
│   │   └── schemas/
│   │       └── v1.json                # JSON Schema for apiVersion v1
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── triage.py                  # top-level Deep Agent
│   │   ├── subagents/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py
│   │   │   ├── clusterer.py
│   │   │   ├── annotator.py
│   │   │   └── drafter.py
│   │   └── prompts/
│   │       └── triage_system.md
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                    # Detector ABC
│   │   ├── llm_judge.py
│   │   ├── regex.py
│   │   ├── tool_call.py
│   │   ├── metric_threshold.py
│   │   └── composite.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                    # ModelProvider ABC
│   │   ├── anthropic.py               # AnthropicProvider
│   │   ├── openai.py                  # OpenAIProvider
│   │   └── pricing.py                 # per-model cost table for --dry-run
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                    # TraceBackend + Tracker ABCs
│   │   ├── trace/
│   │   │   ├── phoenix.py
│   │   │   ├── langfuse.py
│   │   │   └── langsmith.py
│   │   └── tracker/
│   │       ├── jira.py
│   │       ├── linear.py
│   │       └── github.py
│   ├── mcp_servers/                   # entry-point binaries
│   │   ├── adapter_phoenix.py
│   │   ├── adapter_langfuse.py
│   │   ├── adapter_langsmith.py
│   │   ├── adapter_jira.py
│   │   ├── adapter_linear.py
│   │   └── adapter_github.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trace.py                   # OpenInferenceTrace, Span, etc.
│   │   ├── classification.py
│   │   ├── cluster.py
│   │   └── issue.py
│   └── observability/
│       ├── __init__.py                # re-exports `redact`
│       ├── instrumentation.py         # self-instrumentation for the triage agent
│       └── redact.py                  # PII redaction applied before logs and LLM judge input
├── rubrics/
│   ├── builtin/
│   │   ├── agents/
│   │   │   └── v1/
│   │   │       └── rubric.yaml
│   │   ├── rag/v1/rubric.yaml
│   │   ├── routing/v1/rubric.yaml
│   │   └── multi-agent/v1/rubric.yaml
│   └── examples/
│       └── sample-support-agent.yaml  # synthetic reference
├── tests/
│   ├── unit/
│   │   ├── test_rubric_spec.py
│   │   ├── test_loader.py
│   │   ├── test_detectors_*.py
│   │   └── test_clustering.py
│   ├── integration/
│   │   ├── test_phoenix_adapter.py    # against Phoenix in Docker
│   │   ├── test_langfuse_adapter.py
│   │   ├── test_classifier_end_to_end.py
│   │   └── fixtures/
│   │       └── traces/                # checked-in OpenInference traces
│   └── e2e/
│       └── test_full_triage_run.py    # docker-compose stack
└── examples/
    ├── quickstart/
    │   └── README.md
    └── sample-support-agent/
        └── README.md
```

Built-in rubrics live inside the package at `docket/rubric/builtin/` (so `importlib.resources` resolution works from a wheel) rather than the top-level `rubrics/` shown above; `examples/` currently holds the CI recipe.

### Notes on tooling
- `uv` for dependency management
- `ruff` for lint + format
- `mypy --strict` for type checking
- `pytest` for tests, `pytest-asyncio` for async paths
- `pre-commit` hooks enforce all of the above before commit
- CI matrix tests Python 3.11, 3.12, 3.13. 3.11 is the minimum supported version (per `.python-version`).
- License: Apache 2.0 (matches OpenInference and the agent harness; permissive enough for enterprise adoption)

---

## 7. Phased Implementation Plan

Each phase has concrete deliverables and machine-verifiable acceptance criteria. Implementation should not advance until criteria are met.

### Phase 0: Skeleton and project scaffolding (estimated: 1 weekend)

**Deliverables:**
- Repository layout as specified in §6
- `pyproject.toml` with all production and dev dependencies
- `docket.config.Config` Pydantic model loading from `docket.yaml`
- CLI skeleton with `docket --help`, `docket run`, `docket validate`
- CI green: ruff, mypy strict, pytest

**Acceptance criteria:**
- `uv pip install -e .` succeeds
- `docket --help` prints help
- `docket validate path/to/rubric.yaml` exits 0 on valid rubric, non-zero on invalid
- `pytest tests/unit/test_rubric_spec.py` passes (test that Pydantic spec loads valid YAML)
- CI passes on a clean PR

### Phase 1: Rubric DSL and registry (estimated: 1 weekend)

**Deliverables:**
- Complete Pydantic models for all v1 spec
- YAML loader with import resolution (file://, registry://, https://)
- Registry of builtin rubrics
- JSON Schema for v1 spec, validated against representative valid + invalid rubrics
- At least one builtin rubric (`agents/v1`) with 5+ modes
- Comprehensive unit tests

**Acceptance criteria:**
- All five built-in detection types parse and validate
- `docket validate rubrics/builtin/agents/v1/rubric.yaml` exits 0
- Imports across two files merge correctly; later imports override earlier
- 90%+ line coverage on `docket/rubric/`
- Examples in rubrics are exercised by self-test detector

### Phase 2: Detectors (estimated: 1-2 weekends)

**Deliverables:**
- `Detector` ABC and registration system
- All five detector implementations (`llm_judge`, `regex`, `tool_call`, `metric_threshold`, `composite`)
- `llm_judge` uses the in-tree `docket.llm.ModelProvider` ABC with `AnthropicProvider` and `OpenAIProvider` implementations (configurable per-mode). v1.0 does not depend on LiteLLM; the multi-provider abstraction is owned in-tree because structured-output enforcement is load-bearing for classifier reliability and is the exact surface where third-party multi-provider wrappers tend to leak.
- Structured output enforced via each provider's native schema constraint (OpenAI's `response_format` JSON Schema mode, Anthropic's tool-use response shape)
- Self-test mode: each rubric's examples are run through the detector and must produce the expected verdict

**Acceptance criteria:**
- Every detector has unit tests with mocked LLM calls
- One integration test runs `llm_judge` against a real model with a deterministic test rubric and verifies positive + negative cases
- Detector self-tests on builtin rubrics pass
- Budget mode batches up to 8 traces per LLM call for small rubrics; verified by counting API calls in a test

### Phase 3: OpenInference normalization and trace models (estimated: 1 weekend)

**Deliverables:**
- `OpenInferenceTrace` and `Span` Pydantic models matching the OpenInference spec
- Normalization helpers: backend-native trace → OpenInferenceTrace
- Span navigation helpers (`get_llm_spans`, `get_tool_call_spans`, `get_retriever_spans`, `get_final_response`)
- Anonymization/redaction hooks (PII-scrub before sending to LLM judge — important for any production trace data)

**Acceptance criteria:**
- Models round-trip valid OpenInference OTLP spans without data loss
- Navigation helpers return correctly-typed sub-collections
- Redaction hook example: scrubs email addresses, phone numbers, account numbers before classifier sees the trace
- Test fixtures: at least 5 real-shaped traces checked in under `tests/integration/fixtures/traces/`

### Phase 4: Phoenix adapter and first end-to-end loop (estimated: 2 weekends)

**Deliverables:**
- Phoenix MCP server (read + annotate)
- TraceBackend ABC
- Phoenix Docker compose setup for integration tests
- End-to-end CLI: `docket run --rubric agents/v1 --since 1h` works against local Phoenix

**Acceptance criteria:**
- `docker compose up phoenix` starts Phoenix locally
- A scripted ingestion populates Phoenix with 20 sample traces (10 with seeded failures, 10 clean)
- `docket run --backend phoenix --rubric agents/v1 --since 1h` classifies all 20, flags all 10 seeded failures (recall = 1.0), and achieves precision ≥ 0.9 on the 10 clean (at most 1 false positive). Any FPs are surfaced in the run report for human review.
- The triage agent's own runs are visible in Phoenix
- Integration test runs in CI

### Phase 5: Clustering and triage agent (estimated: 2-3 weekends)

**Deliverables:**
- Deep Agents harness wiring up the four subagents
- Clusterer subagent with embedding + HDBSCAN
- Annotator subagent posting to Phoenix
- Issue Drafter subagent generating issue drafts
- Top-level triage agent producing `/report.md`

**Acceptance criteria:**
- End-to-end run on an expanded clustering fixture (~60 traces: 40 with seeded failures across modes, with at least one mode seeded with 5+ semantically similar failures so HDBSCAN can form a cluster at the production-default `min_cluster_size: 3`; 20 clean) produces:
  - Annotations on all classified-positive traces
  - Per-mode clusters with at least one mode reaching `min_cluster_size`
  - Draft issues for the qualifying clusters
  - A summary report with frequency and severity breakdowns
- Drafted issues are written to a local file (no tracker required yet)
- Triage agent's run is traceable in Phoenix end to end
- Failure injection: classifier failure on 3 traces does not abort the run

### Phase 6: Second backend adapter (Langfuse) (estimated: 1 weekend)

**Deliverables:**
- Langfuse MCP server (read + annotate as scores)
- Adapter parity tests: run the same rubric against both Phoenix and Langfuse populated with the same trace data; results MUST match within tolerance

**Acceptance criteria:**
- All Phase 4 + 5 acceptance criteria pass against Langfuse
- Cross-backend parity test (with LLM responses mocked or cached to eliminate provider non-determinism): identical trace data normalized through each adapter produces bit-identical classifier *inputs* (demonstrating the normalization layer is lossless) and bit-identical classifier *outputs* on the mocked LLM; cluster assignments are stable under a fixed embedding-model seed

### Phase 7: Third backend (LangSmith) (estimated: 1 weekend)

**Deliverables:**
- LangSmith MCP server (read + annotate as feedback)
- Same parity tests against LangSmith

**Acceptance criteria:**
- Same as Phase 6, extended to LangSmith
- Documentation explicitly notes the API key requirement and that LangSmith is closed-source

### Phase 8: Tracker adapters (Jira first) (estimated: 1-2 weekends)

**Deliverables:**
- Tracker ABC
- Jira MCP server (list, search, create, update, comment)
- docket provenance block format finalized
- Dedup logic: drafter searches tracker by `(rubric_version, mode_id)` tag before creating
- Auto-post threshold logic + `--review` mode that opens drafts in a markdown editor

**Acceptance criteria:**
- Against a Jira sandbox, a triage run produces a draft for each cluster
- Re-running the same triage with no new traces produces no new issues (dedup works)
- Re-running with new traces in an existing cluster comments on the existing issue, doesn't create a new one
- `--review` mode opens drafts in $EDITOR, accepts or rejects, then posts accepted ones

### Phase 9: Remaining trackers + polish (estimated: 1-2 weekends)

**Deliverables:**
- Linear and GitHub Issues adapters at parity with Jira
- All builtin rubrics (agents, rag, routing, multi-agent) reach v1.0 quality
- Quickstart documentation in `docs/`
- One screencast (5 min) showing setup + run
- Pre-1.0 release on PyPI as `0.1.0`

**Acceptance criteria:**
- All three tracker adapters pass identical integration tests
- All four builtin rubrics validate and pass their own self-tests
- `pip install docket` works
- README has a working quickstart

### Phase 10: v1.0 release prep (estimated: 1 weekend)

**Deliverables:**
- License audit (no AGPL/LGPL deps in the runtime)
- Security audit (no API keys logged; PII redaction documented and tested)
- Performance benchmarks: classify 1000 traces with `agents/v1`, report wall time + cost
- Public announcement: blog post + LangChain Interrupt-style writeup

**Acceptance criteria:**
- A neutral reader can run docket against their own LangSmith or Phoenix deployment in < 30 minutes
- The synthetic sample rubric is published as an example
- Tag `v1.0.0` on the repo

**Total estimated effort to v1.0: 12-16 weekends.** Given prior pacing, this is ~3-4 months of focused weekend work.

---

The phases below are post-v1.0 (target: v1.1). They address production-scale operation — millions of traces per window, sub-hourly cadence, transient backend / provider errors — without violating §2's stateless-runtime rule. v1.0 is correct on the acceptance fixture; v1.1 makes it survive at scale.

**Scale framing (worked example).** A consumer-facing agent deployment with ~5M daily users × ~3 interactions/day, where each interaction fans out to ~10 specialized agents (each calling MCP-served tools), emits roughly **150M traces/day** — ~6M/hour, ~3M per 30-minute window. That's two orders of magnitude above v1.0's acceptance fixture. At those volumes you cannot enumerate; sampling becomes the only feasible operating mode. The phases below assume this framing.

### Phase 11: Reliability and sampling (estimated: 2 weekends)

**Context:** at production scale the runtime needs to (a) bound per-run work so cost is predictable, (b) survive transient `get_trace` / LLM / tracker 429s without losing the run, and (c) resume from where a previous run stopped — all without an internal queue or database. The backend's annotation index already provides idempotent state; this phase makes the runtime use it.

**Deliverables:**
- `docket run --sample N --strategy {uniform,stratified,errors-only}` — sampling applied after `list_traces` returns IDs but before any `get_trace` call, so the savings compound across fetch / classify / annotate. Sampling runs over the *full* listing before checkpoint subtraction (a resumed run completes the original sample; proposal 001 §C.1). `errors-only` pushes a root-error filter down through the listing's reserved `status` filter key; adapters honor it or raise. Stratified mode rebalances with *equal allocation* across an explicit attribute declared on the CLI/config — `--stratify-by {status, latency_bucket, tag:<key>}` (proposal 001 OQ-1 resolved stratification as an operational cost concern, not rubric taxonomy). Stratified-by-tenant (`tag:tenant_id`) is the production primitive that prevents large customers from swamping small ones under uniform sampling.
- Resumability via existing annotations. Before classification, the pipeline queries the backend for annotations carrying this `run_id` *within the same time window*; matching trace IDs are treated as done and skipped. A killed-and-restarted run with the same `(backend, rubric, since, until)` resumes idempotently. The window scope is the state-growth mitigation: sentinel lookups don't scale with run history, only with the current window's volume.
- **Budget gates enforced in the shared listing stage (proposal 001 Spec A).** `max_traces_per_run` measures the effective workload (post-sample, post-checkpoint) and aborts with `BudgetExceededError` before any `get_trace` call; `--max-traces` overrides per-run. `max_estimated_cost_usd`, when set, dollar-gates *every* run on the pre-flight estimate. Both apply identically in deterministic and `--agent` modes.
- **Loud listing truncation (proposal 001 Spec B).** `TraceBackend.list_traces_v2` returns a `TraceListing` (per-trace summaries + `truncated` flag); adapters warn once and flag the listing when they stop paginating at their page ceiling (configurable via `max_list_pages` / the adapter MCP env block, e.g. `LANGSMITH_MAX_LIST_PAGES`), the run report carries a truncation banner, and `RunReport` gains `traces_listed` + `listing_truncated` for Phase 11.5's run-quality metrics. Phoenix's listing gained a real cursor pagination loop (previously a single `first: 500` query).
- **Tracker-side truncation + dedup safety (proposal 001 OQ-4).** `Tracker.list_open_issues_v2` returns an `IssueListing` with the same `truncated` contract; GitHub/Jira/Linear page ceilings are configurable (`GITHUB_/JIRA_/LINEAR_MAX_LIST_PAGES`) and Linear gained a real cursor pagination loop (previously a single `first: 50` query). Because a truncated dedup listing makes "no duplicate found" unprovable, the poster demotes would-be auto-posts to `needs_create` (with a note in the report) rather than risking a duplicate issue; found matches remain trusted.
- `docket run --dry-run` — prints `would classify N traces × M modes ≈ $X (±X)` and exits, using the per-model price table from §8.1 decision 5. Runs the same listing/filter/sample/checkpoint/budget computation as a real run, reports trace-cap and cost-ceiling status, and exits non-zero iff the real run would abort — usable as a CI preflight gate. **Honest variance:** trace sizes vary by ~30× across deployment shapes (a single-LLM agent trace might be 500 tokens; a 10-agent orchestration with full message histories can be 50k+ tokens), so the baked default per-call shape is a wide estimate, not a tight one.
- Per-trace fetch failures already skip-with-warning as of Phase 10 hotfix work; extend to per-annotation writeback failures during the annotate stage and to tracker writes during dedup/post.
- **Retry-with-backoff parity across all six adapters.** The LangSmith adapter ships with `_request` + `Retry-After` honoring + AIMD-aware backoff. Land equivalent retry helpers on Phoenix, Langfuse, Jira, Linear, and GitHub. Tracker rate limits (Jira Cloud ~10 req/s, GitHub 5000/hr) are real and bite when a run produces hundreds of issues.
- A `tests/integration/test_reliability.py` that injects 30% transient backend errors and asserts the run completes with the expected skip count, plus a clean resume-from-checkpoint on the second invocation.

**Acceptance criteria:**
- `docket run --sample 100 --strategy stratified --since 24h` against a backend with ≥10k traces returns 100 classified traces, with `list_traces` consuming O(10k) but classifier only O(100) LLM calls per mode.
- A run whose post-sample, post-checkpoint workload exceeds `max_traces_per_run` raises `BudgetExceededError` before any `get_trace` call; `max_estimated_cost_usd` (when set) aborts on the pre-flight estimate; unset config never dollar-gates. `--dry-run` exit code is non-zero iff the real run would abort, for both cap types.
- A listing that stops at the adapter's page ceiling with a full last page surfaces as `truncated=True` plus exactly one warning, and the run report shows the truncation banner.
- A killed-and-resumed `--sample N --checkpoint` run classifies the *originally sampled* set across both invocations — never more than N distinct traces per `run_id` (sampling precedes checkpoint subtraction).
- A run killed mid-classify and re-invoked with the same `--run-id` (or the deterministic default) completes without redoing already-classified traces, evidenced by backend annotation count incrementing by exactly the remaining difference.
- `--dry-run` reports cost within ±100% of the matching real run's actual cost (the realistic envelope given trace-size variance; tightening to ±10% requires Phase 12's calibration-pass work).
- A run with 30% injected `get_trace` failures completes; the report's `skipped` count matches the injection rate within ±2%.
- Tracker 429s during issue posting trigger backoff-and-retry, not run abortion. Verified by adapter-level unit tests using mocked HTTP transports.

### Phase 11.5: Self-observability and run-quality SLOs (estimated: 1 weekend)

**Context:** *"reliability"* at scale means more than *"the runtime doesn't crash"*. It includes *"the runtime tells you when it's silently wrong"*. If classification positive-rate suddenly drops 90%, the runtime will happily report "0 issues" and the operator finds out three weeks later from a customer-impacting incident. docket triages other people's agents; it needs to triage itself.

**Deliverables:**
- Structured run-quality metrics emitted to the configured `--instrument-to` Phoenix endpoint *and* logged at run-end in a `runs/{run_id}.metrics.json` file:
  - `traces_listed`, `traces_classified`, `traces_skipped`, `traces_resumed`
  - `positive_rate_per_mode` (and run-over-run delta)
  - `llm_429_count`, `backend_429_count`, `tracker_429_count`
  - `mean_input_tokens_per_call`, `mean_output_tokens_per_call`, `total_cost_usd`
  - `wall_time_per_stage` (list / fetch / classify / cluster / draft / post)
- A built-in `docket.dev/builtin/self/v1` rubric that classifies docket's own traces for known runtime failure modes: stuck-in-retry-loop, positive-rate-collapse, backend-listing-empty-but-window-non-empty, adapter-call-stale (latency outlier). This eats our own dog food.
- A `docket health --since 7d` command that reads the metrics files (or Phoenix annotations) for the last N runs and reports trend deltas — positive-rate slope, skip-rate slope, cost trajectory. Operator-facing, not a service.
- Documented Phoenix dashboard JSON in `docs/dashboards/` that consumers can import.

**Acceptance criteria:**
- A deliberate rubric regression (rubric where every mode's detector unconditionally returns negative) is caught by `self/v1` and surfaces in `docket health` as a positive-rate-collapse alert within one run.
- Metrics emitted by a real run match the post-hoc trace inspection within ±1 trace count.

### Phase 12: Streaming pipeline and adaptive concurrency (estimated: 2-3 weekends)

**Context:** today `run_triage_pipeline` materializes the full trace list in memory and uses a fixed `--concurrency` knob. At millions of traces both break: RAM blows up, and the static knob is either too aggressive (429s) or too conservative (idle LLM tier). This phase decomposes the pipeline into an async-generator chain with backpressure and AIMD-tuned concurrency, *and* lets the cost preview self-calibrate.

**Deliverables:**
- Each stage becomes an async generator with a bounded `asyncio.Queue` between it and the next: list → fetch → classify → annotate → cluster-feed. Memory is O(queue_depth × span_size), not O(traces).
- Backpressure: when classifier blocks on LLM 429s, fetcher pauses rather than draining the backend.
- AIMD adaptive concurrency on the classifier *and* the trace-backend / tracker adapters: halve the in-flight count on a 429 hit; additively recover one slot every 30s of clean operation. The controller distinguishes transient 429s ("retry after backoff") from quota-exhausted 429s ("don't retry — surface and stop") by inspecting response body; the latter terminates the run with a clear error.
- **Cost-preview calibration pass.** `--dry-run` optionally fetches a small random sample (default 20 traces), measures real token counts via the provider's `count_tokens` endpoint, and updates the per-call mean before extrapolating to the full N. Tightens the estimate from ±100% to ±10% at the cost of a few seconds of pre-run latency.
- Clusterer becomes a terminal stage that consumes the classifier stream and emits clusters lazily once a mode accumulates enough positives for HDBSCAN to be stable.

**Acceptance criteria:**
- A run over 100k synthetic traces holds steady-state RSS under 500 MB (`tracemalloc` checkpoint inside the integration test).
- 429s from a mocked provider visibly halve the in-flight count in the log (`adapted concurrency: 8 → 4 after rate-limit`), and visibly recover after a clean minute. Behavior pinned by a unit test.
- A quota-exhausted 429 (HTTP 429 with body matching `"daily quota"`) terminates the run with `QuotaExhaustedError`, *not* a retry loop.
- `--dry-run` with calibration enabled lands within ±10% of the real run on the 60-trace acceptance fixture.
- The streaming clusterer produces clusters bit-identical to the v1.0 materialized clusterer on the 60-trace fixture, asserted by the existing `test_clusterer_cluster_ids_stable_under_fixed_embeddings`.

### Phase 13: Horizontal sharding and decoupled clustering (estimated: 2-3 weekends)

**Context:** at sub-hourly cadence with millions of traces per window, even an adaptive single-process pipeline can't keep up. The natural unit of horizontal parallelism is the time window; multiple workers each handle a disjoint sub-window. Clustering, however, needs a global view — so it becomes a separate post-shard step. **And the global step itself isn't free** at this scale: HDBSCAN on 100k+ positive embeddings is non-trivial and may need an approximate algorithm.

**Deliverables:**
- `docket run --window-shard K/N` partitions `[since, until]` into N disjoint sub-windows and runs shard K. The default `run_id` derivation includes shard coordinates so concurrent shards never collide on annotation upserts.
- A new top-level command `docket cluster --since 1h --rubric ...` that reads positive annotations across all shards in the window and produces global clusters + drafts. This decouples the O(N) classify stage (parallelizable freely across shards) from the O(positives) cluster stage (must be global per mode).
- **Approximate clustering above a threshold.** Up to ~10k positives per mode, use HDBSCAN as today. Above that, fall back to mini-batch K-Means + cosine distance (or LSH-bucketed HDBSCAN over buckets) and document the precision/recall trade-off vs. exact HDBSCAN on a 100k-positive fixture. Configurable per-rubric via `clustering.large_scale_algorithm`.
- **Within-cluster representative selection.** Today the representative trace is the centroid. Add `representative_strategy ∈ {centroid, most-recent, highest-evidence-score, most-surprising}` so operators investigating production failures get the *useful* sample, not the median one.
- `docs/scaling.md` runbook showing N=4 hourly shards each running for ~15 minutes against a backend holding ~600k traces, then a single `cluster` command consuming the output. Cron / Airflow recipes included.
- No new infrastructure dependencies. All coordination is via the backend's annotation index and the tracker's idempotency keys.

**Acceptance criteria:**
- 4 parallel `--window-shard k/4` runs against the same backend complete without duplicate annotations (backend-side count == sum of per-shard positive counts).
- The decoupled `cluster` command produces clusters identical to a single-shard equivalent on a fixture sized to fit both (≤10k positives, exact HDBSCAN path).
- Approximate-clustering precision/recall on the 100k-positive fixture stays within ±5% of exact HDBSCAN's cluster assignment agreement (measured by mutual-information score against ground truth).
- The `docs/scaling.md` runbook runs end-to-end on a Phoenix instance pre-loaded with a 1k-trace fixture, with a CI job that exercises it.

**Total estimated effort to v1.1: 7-9 weekends post-v1.0.** Phase 11 is the single biggest reliability win and ships independently. Phase 11.5 is small but disproportionately valuable — it's what lets you trust 11/12/13 in production. Land in that order: 11 → 11.5 → 12 → 13.

### Phase 14: Agent-mode tool surface expansion (estimated: 2-3 weekends)

**Context:** v1.0 ships `--agent` mode (§4.2.2) that runs the same six-stage workflow as the deterministic pipeline, but with LLM-driven planning. The harness exists; the *tool surface* exposed to it does not. The six tools available today (`list_traces`, `classify_traces`, `annotate_classifications`, `cluster_classifications`, `draft_issues_tool`, `write_report`) are all coarse, batch-oriented, and parameterless — they operate on the window and rubric set at construction. That's enough to demo the harness but not enough to support any of the use cases the harness is supposed to unlock: incident investigation, cross-backend composition, per-trace deep dives, composite-mode investigation. This phase expands the tool surface so the planning LLM has something to plan *with*.

**Deliverables:**
- **Parameterized query tools.** Add `query_traces_filtered(time_window, mode_filter, severity_filter, has_positives)`, `query_by_mode(mode_id, since, until)`, and `get_classification(trace_id, mode_id)`. The agent can re-pivot the window mid-run instead of being locked to construction-time `since`/`until`. Tools share the same backend adapter as the existing pipeline tools.
- **Per-trace investigation tools.** Add `reclassify_trace(trace_id, mode_ids, model_override)` that runs the classifier on a single trace with an optional stronger model, and `find_similar_traces(trace_id, k)` that uses the embedding provider to retrieve nearest neighbors. Enables hypothesis-driven follow-up after a batch classification returns suspicious results.
- **Multi-backend dispatch.** Extend `_AgentState` to hold a `dict[str, TraceBackend]` keyed by backend id, and add a `list_backends()` + a backend-selection argument on the query tools. Lets a single agent invocation triage across (e.g.) Phoenix + Langfuse when an org runs both. Adapter construction stays config-driven; the agent picks among already-instantiated adapters, never constructs new ones.
- **Composite-mode step surfacing.** The rubric DSL's `composite` detection type today executes inside the classifier subagent as one atomic call. Add `evaluate_composite_step(trace_id, mode_id, step_name)` that exposes each sub-step (e.g. "check tool arg schema", "cross-reference Linear issue") as an individually callable tool. The classifier subagent gains a hook to delegate to the harness when running composite modes under `--agent`, enabling hypothesis-driven multi-step investigation while keeping deterministic-mode behavior unchanged.
- **System-prompt rework.** The current prompt prescribes the six-stage workflow rigidly. Add a second prompt template for "investigative mode" that describes the tool surface as a library rather than a workflow, with examples of common investigation patterns. CLI flag `--agent-mode {workflow,investigative}` selects between them; workflow is the default for backward-compat.
- **Tool-level observability.** Every new tool emits an OpenTelemetry span via the existing `docket.observability` module, with attributes for backend id, mode id, trace id, and any model override. Same redaction guarantees as the existing pipeline.

**Acceptance criteria:**
- An integration test invokes the agent with an instruction like "find any traces in the last 24h where `tool_call_loop` and `latency_spike` co-occur, then re-classify them with sonnet" and the agent calls `query_by_mode` twice, intersects the results, and calls `reclassify_trace` per match. Run produces a report with the intersected set.
- An integration test wires two backends (a Phoenix mock and a Langfuse mock), invokes the agent with "triage the last hour across all backends", and the agent calls `list_backends` then per-backend `query_traces_filtered`. Both backends' traces appear in the final report.
- A `composite`-mode rubric on a 5-trace fixture produces bit-identical classifications under deterministic mode and under `--agent --agent-mode investigative` (subagent delegation is transparent to detection output).
- New tools have unit tests at the same standard as existing subagents: closure isolation, error paths, and OpenTelemetry span assertions.
- `--agent-mode workflow` (default) produces a report identical to v1.0 `--agent` on the existing e2e fixtures — no behavior regression for the existing path.

### Phase 15: Interactive triage surfaces (estimated: 3-4 weekends)

**Context:** Phase 14 gives the agent enough tools to investigate. Phase 15 gives operators a way to *talk to it*. Today `docket run --agent` is a one-shot CLI invocation with a canned instruction; the harness can't be driven from a chat client, can't accept follow-up questions, and can't be invoked by another agent through MCP. This phase ships the entry points that turn the harness into an interactive triage surface — the actual payoff for shipping `--agent` in the first place.

**Deliverables:**
- **`docket chat` REPL.** A line-based chat loop that keeps a single deepagents instance alive across turns, persists conversation history in the vfs at `/conversations/{session_id}.jsonl`, and accepts freeform operator instructions ("look at the last hour, flag anything related to the auth service deploy"). Supports `/save <path>` to export the final report, `/reset` to clear history without restarting the process, and `/show <vfs-path>` to inspect intermediate artifacts. Session persistence is opt-in via `--session <id>`; default is ephemeral.
- **`docket-server` MCP server.** Exposes the same tool surface as Phase 14's harness over MCP stdio + HTTP, so the agent itself can be invoked from any MCP-aware client (Slack bots, IDE assistants, other Deep Agents). Tools are namespaced under `triage.*` (e.g. `triage.query_by_mode`, `triage.reclassify_trace`). Auth via bearer token from `DOCKET_MCP_TOKEN` env var; rejects unauthenticated calls. New entry in `pyproject.toml` `[project.scripts]`.
- **Slack integration recipe.** A reference Slack bolt app in `examples/slack-bot/` that wraps `docket-server` and routes `/triage` slash commands and DMs to the chat REPL. Not a first-class shipped binary — kept as an example so users can adapt it without taking a Slack-SDK dependency on the core package.
- **Rubric authoring loop.** Add `docket rubric edit <rubric-uri> --interactive` that opens a REPL with rubric-edit tools (`add_mode`, `edit_mode`, `remove_mode`, `dry_run_mode_against_traces`, `compare_modes`) and the Phase 14 query tools. The agent can suggest threshold adjustments, dry-run a new mode against last week's traces, and diff against an existing rubric — all without leaving the REPL. Writes back to a YAML file on disk; never modifies the registry directly.
- **Conversation-aware idempotency.** When a chat session triggers multiple runs across the same window, dedup uses the session's `run_id` deterministically; follow-up queries within a session reuse cached classifications rather than re-classifying. `/reset` clears the cache.
- **Observability hooks.** Every chat turn emits a span with the operator instruction (redacted), tool calls made, tokens consumed, and final response length. Goes to the same OTLP endpoint as pipeline runs.

**Acceptance criteria:**
- `docket chat` accepts a session of at least 3 turns ("triage the last hour", "now focus on auth service", "draft an issue for the highest-severity cluster") and produces a coherent report. Recorded against a fixture backend in an integration test.
- The MCP server passes the standard MCP conformance suite (tool discovery, tool invocation, error responses) and a separate test that exercises every namespaced `triage.*` tool over stdio.
- `docket rubric edit` round-trips a rubric: load → add mode → dry-run against fixture → save → reload validates clean. New YAML matches a golden file modulo timestamps.
- The Slack example bot in `examples/slack-bot/` has a README walkthrough and a `docker-compose.yml` that runs it against a mock backend; manually-verified screenshot in the README.
- A 10-turn session uses no more total tokens than 10 independent `docket run --agent` invocations would (cache hit rate ≥80% measured by the conversation-aware idempotency mechanism).

**Total estimated effort for Phases 14+15: 5-7 weekends.** Phase 14 must land before Phase 15 — the chat REPL is only useful if the agent has tools beyond the six-stage pipeline. Within Phase 15, the chat REPL ships first; MCP server second (depends on REPL session model); rubric authoring loop and Slack example land last as independent extensions.

---

## 8. Decisions and Risks

### 8.1 Resolved decisions

The following design questions were resolved during pre-implementation review. Sections of the spec affected by each decision are noted in parentheses.

1. **OpenInference models — depend on official constants; own the Pydantic layer.** The `openinference-semantic-conventions` package supplies attribute-name constants, not Pydantic models. docket depends on that package for constants and implements its own Pydantic v2 trace models. Pin to a compatible-upper-bound version range; CI matrix tests the current + previous minor to catch spec churn early. (§3, §6, §8.2.3) *Implementation note: v1.0 currently inlines the attribute-name constants; adding the dependency is pending maintainer sign-off.*

2. **Tracker provenance — HTML comment in body + labels.** Every drafted issue carries an HTML-comment provenance block at the end of the body (parseable, invisible to humans, robust to accidental editing) plus tracker labels for queryable dedup: `docket`, `mode:<id>`, `rubric:<rubric-id>@<version>`. Comment format: `<!-- docket:provenance {JSON payload} -->` containing `rubric_version`, `mode_id`, `cluster_id`, `representative_trace_id`, `run_id`. Tracker-native custom-field support is deferred to v1.1+ as an opt-in for Jira/Linear users who prefer it; GitHub Issues has no custom fields, so labels are the only uniform mechanism. (§4.3, §5.2)

3. **Rubric import schemes — filesystem and packaged built-ins in v1.0.** v1.0 supports `file://<path>` (absolute or relative to the importing rubric file) and `docket.dev/builtin/<rubric>/<version>` (resolved against packaged data via `importlib.resources`). v1.1 adds `https://` and a `registry://` scheme backed by HTTPS lookup with checksum pinning. No network fetches in the v1.0 validation path. (§3.2)

4. **Model provider abstraction — roll our own; defer LiteLLM.** v1.0 ships an `docket.llm.ModelProvider` ABC with two implementations (`AnthropicProvider`, `OpenAIProvider`). Each uses its provider's native structured-output mechanism — OpenAI `response_format` JSON Schema mode, Anthropic tool-use output shape — because schema enforcement is load-bearing for classifier reliability and is the exact place where multi-provider abstractions historically leak. v1.1 may add an optional `LiteLLMProvider` for long-tail providers; v1.0 does not depend on LiteLLM. (§6, §7 Phase 2)

5. **Cost controls — hard cap + concrete dry-run.** Default `max_traces_per_run: 1000` in `docket.yaml`, configurable. Exceeding the cap raises an error rather than silently truncating, forcing the operator to partition explicitly. `docket run --dry-run` reports: candidate trace count in the window, modes per trace, expected LLM call count after batching, and projected cost computed from a per-model price table baked into the package and overridable in config. (§8.2.5)

### 8.2 Risks

1. **Contributor confidentiality.** Any reference rubric checked into `rubrics/examples/` MUST be a synthetic example with no proprietary domain content. Contributors MUST NOT publish names, taxonomies, prompts, or trace contents derived from any non-public system.

2. **OpenInference instability.** Spec is evolving; major changes could break the normalizer. Mitigation: pin to a known-good spec version; CI matrix tests against the previous + current version.

3. **MCP protocol churn.** MCP is maturing rapidly; tool-use semantics could change. Mitigation: pin to known-stable MCP SDK versions; track the protocol's release notes.

4. **Classifier cost runaway.** A naive run on 10k traces with a 10-mode rubric is 100k LLM calls. Mitigation: budget mode batching + hard cap + cost preview before run (`docket run --dry-run` reports expected cost).

5. **Maintenance burden of N adapters.** Six adapters × ongoing API changes = sustained maintenance. Mitigation: aggressive integration test coverage, with daily CI runs against the adapters; community ownership model for less-common backends.

6. **Self-test bias in builtin rubrics.** Examples in a rubric are also the test set — circular. Mitigation: separate held-out validation traces per builtin rubric, maintained by the project, not the rubric author.

---

## 9. Success Criteria

### Technical
- Phase 5 end-to-end loop works on real-world agent traces
- Cross-backend parity holds within tolerance
- Classifier latency: median < 2s per trace with a 5-mode rubric, batched
- Cost: < $0.05 per 100 traces with default settings on Claude Haiku or GPT-4o-mini

### Community
- Adopted or referenced by maintainers of an established observability backend
- Healthy GitHub engagement (stars, issues, discussions) in the first six months
- External contributors land PRs (new rubric, new adapter, or bugfix)
- Cited by a downstream paper or production blog

---

## Appendix A: Key Prior Art

- **Latitude** — annotation-driven eval generation; closest in workflow scope; now ships automatic issue creation
- **OpenInference** (Arize) — semantic conventions consumed by this project; Apache 2.0
- **Deep Agents** (LangChain) — production agent harness; MIT
- **LangGraph** — orchestration runtime; MIT
- **RIFT** (Snorkel, ICLR 2026 workshop) — rubric failure-mode taxonomy; complementary, not competing
- **AdaRubric** (Alibaba, 2026) — task-adaptive rubrics; closer methodology, not productized
- **Agentic Rubrics for SWE** (2026) — also uses `rubric.yaml`; different domain
- **Traceloop opentelemetry-mcp-server** — read-only MCP for OTel traces; subset of what we need
- **adham90/opentrace** — MCP-native observability; single-store; influence for the MCP-first interface

## Appendix B: Out of Scope but Adjacent

- **Eval auto-generation from triage clusters** — v1.0 ships the minimal half of this: `docket run --emit-evals <dir>` exports each qualifying cluster as a portable JSON candidate eval case (representative excerpt, mode, expected verdict, member trace IDs, provenance), satisfying mission item §1.1.5. LLM-driven *generation* of executable evals (and adapters into `agentevals` / DeepEval) remains v1.1 work.
- **Adversarial neighborhood expansion** — generate semantically equivalent variants of failing traces to assert robustness. Likely a separate companion library, `docket-adversarial`.
- **MASEval orchestration-pattern comparison layer** — separate project on a longer arc. Could share rubric DSL infrastructure if MASEval maintainers want it.
- **Web UI for review** — out of scope for v1.0; users review in their existing tracker. v1.x or never.

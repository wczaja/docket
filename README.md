# docket

[![CI](https://github.com/wczaja/docket/actions/workflows/ci.yml/badge.svg)](https://github.com/wczaja/docket/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

An observability-platform-agnostic triage runtime for LLM agent traces.

docket reads traces from your existing observability backend
(**Phoenix**, **Langfuse**, **LangSmith**), classifies each one against a
YAML failure-mode taxonomy you write, clusters similar failures together,
and drafts issues into your tracker (**Jira**, **Linear**, **GitHub
Issues**). It is **not** a new observability backend, an eval framework,
or a web UI — it's a thin agent that sits *above* what you already have.

Human-in-the-loop is the default: drafts queue locally or open in your
`$EDITOR` for review before they post. Auto-posting requires an explicit
opt-in (`auto_post_threshold`).

---

## Quickstart (5 minutes)

The fastest path to a working setup: a local Phoenix backend + a
GitHub-Issues tracker.

### 1. Install

```bash
pip install docket
# or:  uv pip install docket
```

### 2. Bring up Phoenix

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

Send your agent's traces to `http://localhost:6006` via the
OpenInference instrumentation of your choice (any OTLP-compatible
instrumentation works — see `docs/local-phoenix.md` for ingestion
recipes).

### 3. Configure credentials

```bash
export ANTHROPIC_API_KEY="sk-ant-..."         # for the llm_judge detectors
export OPENAI_API_KEY="sk-..."                # for clustering embeddings
                                              # (required even with an
                                              # Anthropic classifier)
export GITHUB_TOKEN="ghp_..."                 # PAT with Issues write
```

### 4. Run

```bash
docket run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --tracker github \
  --github-owner YOUR_GH_USER \
  --github-repo docket-issues \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

That's it. The pipeline:

1. Pulls the last hour of traces from Phoenix.
2. Runs each one through the `agents/v1` failure-mode rubric.
3. Clusters positive classifications per mode.
4. Drafts one issue per cluster into
   `~/.docket/queued-issues/<run-id>/`.
5. Looks at your GitHub repo for matching open issues (dedup by labels +
   embedded provenance) and comments on existing issues that grew, or
   leaves new ones in the local queue for `--review`.
6. Prints a markdown report.

Add `--review` to walk each queued draft through `$EDITOR` + accept/reject
+ post. Add `--auto-post-threshold high` to auto-post critical and high
severity drafts. Add `--dry-run` to price a window before committing to it.

For scheduled triage, swap `run` for the daemon:

```bash
docket serve --interval 1h ...   # same flags as run
```

Each tick processes exactly the window since the last successful tick —
no gaps, no overlap — and a failed tick retries its window instead of
dropping it. (Plain cron + `docket run` works too; `serve` just
does the window bookkeeping for you.)

For other backends and trackers, see `docs/quickstart.md` (full matrix:
Phoenix/Langfuse/LangSmith × Jira/Linear/GitHub).

---

## What it does

docket runs a small pipeline of LLM-driven subagents over your
existing traces:

```
┌──────────────────────────┐
│ Phoenix / Langfuse /     │
│ LangSmith trace backend  │  <- you already have this
└────────────┬─────────────┘
             │ trace fetch (read-only by default)
             ▼
   ┌─────────────────────┐
   │ classifier subagent │  rubric: YAML failure-mode taxonomy
   └──────────┬──────────┘     (built-in or your own)
              ▼
   ┌─────────────────────┐
   │ clusterer subagent  │  embeddings + HDBSCAN per mode
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ drafter subagent    │  one IssueDraft per cluster, with
   └──────────┬──────────┘     embedded provenance for dedup
              ▼
   ┌─────────────────────┐
   │ poster subagent     │  dedup against tracker, then
   └──────────┬──────────┘     comment / create / queue
              ▼
┌──────────────────────────┐
│ Jira / Linear / GitHub   │
└──────────────────────────┘
```

**Read-only by default.** Annotations write back to the trace backend
only when you pass `--annotate`. Issues post to the tracker only when
their severity meets `auto_post_threshold` (default: `never`) or when
you opt in via `--review`.

**Bounded by default.** Every run is capped by `max_traces_per_run`
(default 1000, measured after sampling and checkpoint subtraction);
exceeding the cap aborts loudly before any trace is fetched — never a
silent truncation. An optional `max_estimated_cost_usd` adds a dollar
ceiling on the pre-flight cost estimate. `--dry-run` reports both gates
and exits non-zero iff the real run would abort, so CI can use it as a
preflight check. For production-scale windows, `--sample N` bounds the
work with `--strategy uniform`, `--strategy errors-only` (root-errored
traces, filter pushed down to the backend), or `--strategy stratified
--stratify-by status|latency_bucket|tag:<key>` (equal allocation so rare
strata — errors, small tenants, tail latencies — get seen). Adapters
flag truncated listings — trace and tracker alike — instead of silently
stopping at their pagination ceiling; when the open-issue listing is
truncated during dedup, drafts are queued for review instead of
auto-posted, since "no duplicate found" was not proven.

**State lives in the backends, not here.** docket doesn't own a
database. Annotations key off `(trace_id, run_id, rubric_version,
mode_id)` in the observability backend; issues key off labels +
HTML-comment provenance in the tracker. Re-running the same window is
idempotent.

---

## Built-in rubrics

Four reference rubrics ship with the package; each is a starting point
intended to be imported into a domain-specific rubric you maintain.

| URI                                       | Modes |
| ----------------------------------------- | ----- |
| `docket.dev/builtin/agents/v1`      | 6 — hallucination, infinite loop, premature termination, unsafe tool call, refusal leakage, bad handoff |
| `docket.dev/builtin/rag/v1`         | 4 — off-corpus answer, missing citation, stale retrieval, context overflow |
| `docket.dev/builtin/routing/v1`     | 4 — wrong-skill routing, capability mismatch, dead-end transfer, oscillation |
| `docket.dev/builtin/multi-agent/v1` | 4 — handoff context loss, conflicting instructions, role drift, shared-memory corruption |

Reference them by URI on the CLI (`--rubric docket.dev/builtin/rag/v1`)
or import them into your own rubric:

```yaml
apiVersion: docket.dev/v1
kind: Rubric
metadata:
  name: my-prod-agents
  version: 1.0.0
imports:
  - docket.dev/builtin/agents/v1
  - docket.dev/builtin/rag/v1
modes:
  - id: refund-without-confirmation
    severity: critical
    detection:
      type: tool_call
      tool_calls: [process_refund]
    # ... your modes go here
```

Validate with `docket validate ./my-rubric.yaml`. Smoke-test the
examples with `docket self-test ./my-rubric.yaml`.

---

## Architecture overview

- **OpenInference** is the canonical trace schema. Adapters normalize *to*
  it; the runtime never sees backend-specific shapes.
- **MCP** is the integration protocol for both trace backends and
  trackers. The CLI ships one MCP server binary per adapter
  (`docket-adapter-phoenix`, `docket-adapter-jira`, …) that
  you can run standalone or invoke through `docket run`.
- **deepagents** is the agent harness; we don't reimplement planning,
  virtual filesystems, or subagent delegation.
- **Stateless runtime.** Annotations live in the backend; issues live in
  the tracker. No local database.
- **Pydantic v2 + httpx + asyncio** throughout. No bespoke SDK dependency
  per backend — every adapter is plain HTTP.

### Execution modes

docket ships two execution modes over the same six pipeline stages
(`list_traces` → `classify_traces` → `annotate_classifications` →
`cluster_classifications` → `draft_issues` → `write_report`):

- **Deterministic pipeline (default).** Stages run in a fixed order from
  plain Python. Predictable cost, reproducible across runs, easy to
  debug. **Use this for** batch / cron / CI, anywhere SLOs and cost
  forecasting matter.
- **deepagents harness (`--agent`).** Same six stages exposed as tools
  to a top-level planning LLM. **Use this for**
  exploratory / debugging runs today; the harness is the substrate the
  project commits to for future interactive surfaces (chat-driven
  triage, incident investigation, rubric authoring). The tools and
  entry points for those surfaces are post-v1.0 work — see
  [`docs/design.md`](docs/design.md) §4.2 and §7 (Phases 14–15).

Both modes share the same subagents, the same `run_id`, and the same
annotation idempotency, so investments in one benefit the other.

For the full design, see [`docs/design.md`](docs/design.md). Per-backend
and per-tracker setup guides:

- [Phoenix](docs/local-phoenix.md)
- [Langfuse](docs/local-langfuse.md)
- [LangSmith](docs/local-langsmith.md)
- [Jira](docs/local-jira.md) — Cloud + Data Center
- [Linear](docs/local-linear.md)
- [GitHub Issues](docs/local-github.md)

---

## Documentation

Start at the [docs index](docs/index.md).

**Guides**

- [Quickstart](docs/quickstart.md) — every backend × tracker pair
- [Concepts](docs/concepts.md) — the vocabulary in five minutes
- [Adapters](docs/adapters.md) — the integration contracts + how to add
  a backend or tracker
- [Benchmarks](docs/benchmarks.md) — wall time and cost for a
  1000-trace run
- [Design document](docs/design.md) — every architectural decision,
  with rationale

**API reference**

- [CLI](docs/cli.md) — every command, flag, and exit code for `run`,
  `serve`, `validate`, `self-test`, and the adapter binaries
- [Configuration](docs/configuration.md) — `docket.yaml` schema,
  all env vars, precedence rules, defaults
- [Python API](docs/python-api.md) — embed the pipeline as a library:
  `run_triage_pipeline`, adapters, providers, models, errors
- [MCP servers](docs/mcp-servers.md) — tool contracts for driving the
  adapters from any MCP client
- [Rubric DSL](docs/rubric-spec.md) — the complete taxonomy spec, with
  [a worked example rubric](rubrics/examples/sample-support-agent.yaml)

## Status

**v1.0.** Three trace-backend adapters and three tracker adapters at
parity, four built-in rubrics, deterministic + agent-harness execution
modes, daemon mode, budget guardrails and sampling. The
[changelog](CHANGELOG.md) has the full feature list. Post-1.0 roadmap
(streaming, sharding, interactive surfaces) lives in
[`docs/design.md`](docs/design.md) §7.

## Contributing

Rubrics and adapters are the highest-leverage contributions, and both
have step-by-step guides: see [CONTRIBUTING.md](CONTRIBUTING.md). Bug
reports and adapter proposals have issue templates. Security issues go
through [SECURITY.md](SECURITY.md) — never a public issue.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

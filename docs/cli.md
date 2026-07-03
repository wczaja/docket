# CLI reference

Complete reference for the `docket` command and the six
`docket-adapter-*` MCP server binaries. Flag text here matches
`--help` output; when in doubt, `--help` for your installed version is
authoritative.

```
docket [--version] [-h]
  demo       the full pipeline on bundled synthetic traces (no credentials)
  init       interactively scaffold a docket.yaml
  run        one-shot pipeline over a time window
  serve      the same pipeline on a fixed cadence (daemon)
  validate   schema-validate a rubric (no network, no credentials)
  self-test  run a rubric's examples through live detectors
```

Conventions used by every command:

- **Durations** are `<int><unit>` with unit `s|m|h|d` — `30m`, `1h`, `7d`.
- **Rubric sources** are a filesystem path, a `file://` URI, or a builtin
  URI (`docket.dev/builtin/<name>/v1` where name ∈ `agents`, `rag`,
  `routing`, `multi-agent`, `mast`).
- **Precedence**: CLI flag > config-file value > built-in default.
- **Logging** goes to stderr (`-v` for DEBUG, `-q` for warnings-only);
  reports and validation results go to stdout, so output is pipeable.

---

## `docket demo`

Runs the real pipeline — classify → cluster → draft → report — over 60
bundled synthetic traces (20 clean + 40 seeded failures) with an
in-memory backend. No API keys, no Docker, no instrumented app. LLM-judge
modes and the drafter run under a clearly-labeled deterministic scripted
judge by default; deterministic detectors (regex, tool_call,
metric_threshold) run for real either way.

```bash
docket demo                        # free, offline, deterministic
docket demo --live                 # real judge via ANTHROPIC_API_KEY
docket demo --rubric ./mine.yaml   # your taxonomy against the demo traces
```

| Flag | Default | Meaning |
|---|---|---|
| `--live` | off | Judge with a real provider instead of the scripted judge |
| `--provider URI` | provider default | `provider:model` for `--live` (error without `--live`) |
| `--embedding URI` | demo embeddings | Real embedding provider (`openai:…`, `voyage:…`, `local:…`); default is a free deterministic hashed-unigram embedder |
| `--rubric SOURCE` | `docket.dev/builtin/agents/v1` | Rubric to classify against; custom `llm_judge` modes score negative under the scripted judge (a warning says so) — pair with `--live` |
| `--out DIR` | `docket-demo` | Where drafts + `report.md` land |
| `--to-phoenix URL` | — | Don't run the pipeline; ingest the demo traces into a running Phoenix via OTLP and print the `docket run` command to triage them |
| `-v`, `--quiet` | info | Log verbosity |

Exit codes: `0` success; `1` config/provider error, or (with
`--to-phoenix`) nothing/partially ingested.

The demo backend is also available to every `run`/`serve` feature as
`--backend demo` — useful for exercising sampling, checkpointing, and
tracker flows without a live backend (real providers and their keys
apply there).

---

## `docket init`

Interactive `docket.yaml` scaffolder: backend, tracker (or none),
rubric, auto-post threshold — every prompt has a default, so
Enter-through-everything yields a working local-Phoenix read-only
config. Secrets are written as `${ENV_VAR}` references, never values.
Validates that the chosen rubric loads, refuses to overwrite without
`--force`, and prints the env vars to export plus the `--dry-run`
command to try first.

| Flag | Default | Meaning |
|---|---|---|
| `--path FILE` | `docket.yaml` | Where to write |
| `--force` | off | Overwrite an existing file |

---

## `docket run`

Runs: list → classify → annotate (opt-in) → cluster → draft → dedup/post
→ report, over the window `[now − --since, now − --until]`. Prints the
markdown run report to stdout.

```bash
docket run --backend phoenix --phoenix-url http://localhost:6006 \
  --rubric docket.dev/builtin/agents/v1 --since 1h
```

### Window and identity

| Flag | Default | Meaning |
|---|---|---|
| `--since DURATION` | `1h` | Window start, measured back from now |
| `--until DURATION` | now | Window end, measured back from now (must be after `--since`) |
| `--run-id TEXT` | derived | Override the deterministic `sha256(backend\|rubric@version\|since\|until)[:16]`. Use for backfills/replays where you need explicit grouping |

### Rubric and config

| Flag | Default | Meaning |
|---|---|---|
| `--rubric SOURCE` | config's `rubric:` | Rubric to classify against |
| `--config FILE` | `docket.yaml` | Config file path; may be absent if backend flags are given |

### Backend selection (one required, via flag or config)

`--backend {phoenix|langfuse|langsmith|demo}` plus its connection flags:

| Backend | Flags (config-env equivalent in parentheses) |
|---|---|
| phoenix | `--phoenix-url` (`PHOENIX_URL`, required), `--phoenix-api-key` (`PHOENIX_API_KEY`) |
| langfuse | `--langfuse-host` (`LANGFUSE_HOST`, required), `--langfuse-public-key`, `--langfuse-secret-key` |
| langsmith | `--langsmith-api-key` (`LANGSMITH_API_KEY`, required), `--langsmith-endpoint` (default `https://api.smith.langchain.com`), `--langsmith-project` |
| demo | none — the bundled in-memory synthetic fixture (see `docket demo`) |

### Tracker selection (optional; omit to queue drafts locally)

`--tracker {jira|linear|github}` plus:

| Tracker | Flags |
|---|---|
| jira | `--jira-host` + `--jira-project` (required), Cloud auth: `--jira-email` + `--jira-api-token`, Data Center auth: `--jira-pat`, `--jira-deployment {cloud|datacenter}` (default: auto-detect from hostname) |
| linear | `--linear-api-key` + `--linear-team` (required; the team UUID), `--linear-endpoint` |
| github | `--github-token` + `--github-owner` + `--github-repo` (required), `--github-api-url` (for GitHub Enterprise Server) |

### Classification

| Flag | Default | Meaning |
|---|---|---|
| `--provider {anthropic|openai}` | `anthropic` | Judge provider for `llm_judge` modes without their own `model:` |
| `--model TEXT` | `claude-haiku-4-5-20251001` / `gpt-4o-mini` | Override the provider's default model |
| `--concurrency 1..64` | `8` | Traces classified in parallel; lower it on tight rate-limit tiers |
| `--batch 1..32` | `1` | Traces batched per LLM call (budget mode for small rubrics) |

### Clustering

| Flag | Default | Meaning |
|---|---|---|
| `--clustering {embedding\|mode-only}` | `embedding` | `embedding`: per-mode HDBSCAN over excerpt embeddings. `mode-only`: one cluster per firing mode, no embedding provider needed — the single-API-key operating mode (lossy: sub-patterns within a mode aren't separated) |
| `--embedding URI` | `openai:text-embedding-3-small` | Embedding provider as `provider:model`. `openai:…` (OPENAI_API_KEY), `voyage:…` (VOYAGE_API_KEY), `local:…` (no key; `pip install "docket-runtime[local-embeddings]"` — model fetched from Hugging Face on first use) |

### Writeback and posting (everything off by default)

| Flag | Default | Meaning |
|---|---|---|
| `--annotate/--no-annotate` | off | Write classifications back to the backend, keyed `(trace_id, run_id, rubric_version, mode_id)` (upsert) |
| `--auto-post-threshold {critical|high|medium|low|never}` | `never` | Auto-post new issues at/above this severity; everything else queues |
| `--review/--no-review` | off | After the run, walk each `needs_create` draft through `$EDITOR` → y/n → post. Without `$EDITOR`, prints draft + prompts |
| `--queue-dir DIR` | `~/.docket/queued-issues/` | Where drafts and the report land |

### Budget, sampling, resumability

| Flag | Default | Meaning |
|---|---|---|
| `--max-traces 1..10000000` | config / `1000` | Hard cap on effective workload (post-sample, post-checkpoint). Exceeding it aborts **before any trace fetch** with `BudgetExceededError`; truncation is never silent |
| `--sample N` | off | Classify only N traces sampled from the window, seeded by `run_id` (re-runs sample identically) |
| `--strategy {uniform|stratified|errors-only}` | `uniform` | `errors-only` pushes a root-error filter down to the backend; `stratified` allocates equally across strata |
| `--stratify-by {status|latency_bucket|tag:<key>}` | — | Required with `--strategy stratified`; an unusable key aborts loudly |
| `--checkpoint/--no-checkpoint` | off | Write per-trace sentinels; on re-run with the same `run_id`, skip already-classified traces. Requires backend write access |
| `--dry-run` | off | Same listing/filter/sample/checkpoint/budget computation as a real run, prints the cost estimate + both gate statuses, executes nothing. **Exit code is non-zero iff the real run would abort** — usable as a CI preflight gate |

### Execution mode and observability

| Flag | Default | Meaning |
|---|---|---|
| `--agent/--no-agent` | off | Route through the planning-LLM harness instead of the deterministic pipeline (exploratory use; deterministic is the production model) |
| `--instrument-to URL` | off | Emit the triage run's own OpenInference spans via OTLP to this endpoint (e.g. a local Phoenix at `http://localhost:6006`) |
| `-v` / `-vv`, `-q` | info | Log verbosity (stderr) |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Run completed (or `--dry-run`: real run would proceed) |
| 1 | Any `DocketError`: config/credential error, rubric validation failure, budget exceeded, backend write failure after retries — or `--dry-run` predicting an abort |
| 2 | CLI usage error (bad flag value, malformed duration) |

Per-trace fetch failures and per-trace classifier failures (after 3
retries) do **not** abort the run; they're skipped and counted in the
report. Tracker write failures queue the draft locally instead of
failing the run.

---

## `docket serve`

Daemon mode: the `run` pipeline on a fixed cadence. Accepts **the same
backend / tracker / rubric / classification / clustering / budget flags
as `run`** (everything above except `--since`, `--until`, `--run-id`,
`--review`, `--agent`, `--dry-run` — a daemon has no operator at the
keyboard, and serve always uses the deterministic pipeline and derived
run ids).

```bash
docket serve --interval 1h --config docket.yaml --annotate --checkpoint
```

| Flag | Default | Meaning |
|---|---|---|
| `--interval DURATION` | `1h` | Cadence. Tick *i* processes `[last successful tick's end, now]`; the first tick processes the trailing interval. Consecutive windows tile exactly — no gaps, no overlap |
| `--max-ticks N` | unlimited | Exit after N ticks (supervised schedulers, smoke tests) |

Failure semantics: a failed tick logs the error and does **not** advance
the window — the next tick retries the union, so no traces are silently
dropped (a persistently failing window will eventually hit the trace cap
loudly; that's intentional). Config and credential errors exit
immediately with code 1. Stop with Ctrl-C / SIGTERM. Each tick gets a
fresh window-derived `run_id`, so annotations and tracker dedup behave
exactly as if cron invoked `run`.

---

## `docket validate <SOURCE>`

Schema-validates one rubric (Pydantic + JSON Schema; imports are
resolved and merge rules checked by the loader). No network, no
credentials. Exit 0 with `OK: <name> v<version>` or exit 1 with the
field-level errors.

```bash
docket validate ./my-rubric.yaml
docket validate docket.dev/builtin/rag/v1
```

## `docket self-test <SOURCE>`

Runs each mode's `examples:` through its live detector and asserts the
expected verdict — the rubric's regression suite against judge-prompt
drift.

| Flag | Default | Meaning |
|---|---|---|
| `--provider {anthropic|openai}` | `anthropic` | Judge provider for modes without their own `model:` |
| `--model TEXT` | provider default | Model override |
| `--batch 1..32` | `1` | Examples per LLM call |

Output is one `PASS`/`FAIL`/`SKIP` line per example (v1.0 exercises
`llm_judge` examples; deterministic-mode examples report SKIP). Exit 1
iff any example failed. Requires the provider's API key.

---

## MCP adapter binaries

Each adapter also ships as a standalone stdio MCP server (configured
purely by environment variables — see [mcp-servers.md](mcp-servers.md)
for the tool contracts):

```
docket-adapter-phoenix     PHOENIX_URL (required), PHOENIX_API_KEY, PHOENIX_MAX_LIST_PAGES
docket-adapter-langfuse    LANGFUSE_HOST (required), LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_MAX_LIST_PAGES
docket-adapter-langsmith   LANGSMITH_API_KEY (required), LANGSMITH_ENDPOINT, LANGSMITH_PROJECT, LANGSMITH_MAX_LIST_PAGES
docket-adapter-jira        JIRA_HOST + JIRA_PROJECT (required), JIRA_EMAIL + JIRA_API_TOKEN or JIRA_PAT, JIRA_DEPLOYMENT, JIRA_MAX_LIST_PAGES
docket-adapter-linear      LINEAR_API_KEY + LINEAR_TEAM_ID (required), LINEAR_ENDPOINT, LINEAR_MAX_LIST_PAGES
docket-adapter-github      GITHUB_TOKEN + GITHUB_OWNER + GITHUB_REPO (required), GITHUB_API_URL, GITHUB_MAX_LIST_PAGES
```

A missing required variable exits with code 2 and the variable's name on
stderr.

# Configuration reference

Three layers, in precedence order: **CLI flags > `docket.yaml` >
built-in defaults**. Process environment variables are read in exactly
two places: LLM/embedding provider SDK credentials, and the standalone
MCP adapter binaries. (The `env:` blocks inside `docket.yaml` are
*config values shaped like env vars* — the runtime reads them from the
file, not from your shell.)

## `docket.yaml`

Validated by `docket.config.Config` (Pydantic); unknown or
malformed fields raise `ConfigError` at startup, before any I/O. The
default path is `./docket.yaml`; override with `--config`. The
file is optional if `--backend` + its connection flags are given.

```yaml
# Required: where traces come from.
trace_backend:
  type: mcp                              # only "mcp" in v1
  command: docket-adapter-phoenix  # selects the backend (suffix after
                                         # "docket-adapter-" or a bare
                                         # name: phoenix | langfuse | langsmith)
  env:                                   # backend-specific settings, see tables below
    PHOENIX_URL: http://localhost:6006

# Optional: where issues go. Omit to queue drafts locally with no dedup.
tracker:
  type: mcp
  command: docket-adapter-github   # jira | linear | github
  env:
    GITHUB_TOKEN: ghp_...
    GITHUB_OWNER: my-org
    GITHUB_REPO: agents

# Required: the rubric to classify against (path, file:// URI, or builtin URI).
rubric: docket.dev/builtin/agents/v1

# Optional knobs (defaults shown):
max_traces_per_run: 1000          # hard cap on effective workload; > 0
max_estimated_cost_usd: null      # dollar ceiling on the pre-flight estimate;
                                  # when set, EVERY run is gated, not just --dry-run
auto_post_threshold: never        # critical | high | medium | low | never
instrumentation_backend: null     # MCP block for self-instrumentation (reserved;
                                  # the CLI's --instrument-to flag is the v1 surface)
```

### Backend `env` keys

| Backend (`command`) | Key | Required | Notes |
|---|---|---|---|
| phoenix | `PHOENIX_URL` | yes | e.g. `http://localhost:6006` |
| | `PHOENIX_API_KEY` | no | for authenticated Phoenix deployments |
| langfuse | `LANGFUSE_HOST` | yes | cloud or self-hosted URL |
| | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | for auth | key pair |
| langsmith | `LANGSMITH_API_KEY` | yes | |
| | `LANGSMITH_ENDPOINT` | no | default `https://api.smith.langchain.com` |
| | `LANGSMITH_PROJECT` | no | project/session filter |

### Tracker `env` keys

| Tracker | Key | Required | Notes |
|---|---|---|---|
| jira | `JIRA_HOST` + `JIRA_PROJECT` | yes | host URL + project key |
| | `JIRA_EMAIL` + `JIRA_API_TOKEN` | Cloud auth | Basic auth pair |
| | `JIRA_PAT` | Data Center auth | personal access token |
| | `JIRA_DEPLOYMENT` | no | `cloud`/`datacenter`; default auto-detects from hostname |
| linear | `LINEAR_API_KEY` + `LINEAR_TEAM_ID` | yes | team **UUID**, not the key |
| | `LINEAR_ENDPOINT` | no | default `https://api.linear.app/graphql` |
| github | `GITHUB_TOKEN` + `GITHUB_OWNER` + `GITHUB_REPO` | yes | PAT needs Issues read/write |
| | `GITHUB_API_URL` | no | set for GitHub Enterprise Server |

### Listing page ceilings

Every listing paginates up to a fixed page ceiling per adapter:

| Adapter | Ceiling | When a listing reaches it |
|---|---|---|
| phoenix | 50 pages × 500 spans | the listing fails with `BackendError`, so the run aborts |
| langfuse, langsmith | 50 pages × 100 traces | the listing stops at 5,000 traces **without a warning** |
| jira, linear, github | 20 pages × 50 issues | the adapter warns and flags the listing as truncated; dedup then queues drafts that matched nothing instead of auto-posting them (a duplicate can't be ruled out) |

The ceilings aren't configurable yet. Planned: a `max_list_pages`
constructor argument on every adapter and a matching
`<NAME>_MAX_LIST_PAGES` key (e.g. `LANGSMITH_MAX_LIST_PAGES`), where a
malformed value is a `ConfigError`. Until then nothing reads such keys,
whether they're set in an `env:` block or in the environment of an MCP
binary. Also planned: trace listings that hit their ceiling get the
tracker behavior (a `truncated` flag and one warning) plus a truncation
banner in the run report.

## Process environment variables

| Variable | Read by | When needed |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic SDK | `--provider anthropic` (default) with any `llm_judge` mode, and `self-test` |
| `OPENAI_API_KEY` | OpenAI SDK | `--provider openai`, **and always for clustering** — embeddings are OpenAI-only in v1, and the pipeline preflights this key before classifying so it fails fast rather than after spend |
| `EDITOR` | `--review` flow | optional; without it, drafts print to stdout with a y/n prompt |
| Adapter variables (`PHOENIX_URL`, `GITHUB_TOKEN`, …) | **standalone MCP binaries only** | when running `docket-adapter-*` directly; the `run`/`serve` CLI takes these from flags or the config `env:` blocks instead |

## Built-in defaults worth knowing

| Setting | Default | Where |
|---|---|---|
| Judge model (anthropic) | `claude-haiku-4-5-20251001` | `docket.llm.DEFAULT_ANTHROPIC_MODEL` |
| Judge model (openai) | `gpt-4o-mini` | `docket.llm.DEFAULT_OPENAI_MODEL` |
| Embedding model | `text-embedding-3-small` | `docket.llm.DEFAULT_OPENAI_EMBEDDING_MODEL` (rubrics can override via `clustering.embedding_model`) |
| Classification concurrency | 8 | `--concurrency` |
| Classifier retries | 3, exponential backoff | then the trace is marked unprocessed, run continues |
| Backend write retries | 5 | then the run aborts (no partial-write runs) |
| Trace cap | 1000 | `max_traces_per_run` / `--max-traces` |
| Queue directory | `~/.docket/queued-issues/` | `--queue-dir` |
| Self-instrumentation endpoint | `http://localhost:6006` | `--instrument-to` default target when enabled |
| Price table for `--dry-run` | baked per-model table | `docket/cost.py` (single-file edit to update) |

## Secrets handling

Credentials never appear in logs, annotations, reports, drafted issues,
or the agent-mode virtual filesystem; missing/invalid credentials abort
at startup naming the missing *variable*. Full posture:
[SECURITY.md](../SECURITY.md).

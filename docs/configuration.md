# Configuration reference

Three layers, in precedence order: **CLI flags > `agent-triage.yaml` >
built-in defaults**. Process environment variables are read in exactly
two places: LLM/embedding provider SDK credentials, and the standalone
MCP adapter binaries. (The `env:` blocks inside `agent-triage.yaml` are
*config values shaped like env vars* — the runtime reads them from the
file, not from your shell.)

## `agent-triage.yaml`

Validated by `agent_triage.config.Config` (Pydantic); unknown or
malformed fields raise `ConfigError` at startup, before any I/O. The
default path is `./agent-triage.yaml`; override with `--config`. The
file is optional if `--backend` + its connection flags are given.

```yaml
# Required: where traces come from.
trace_backend:
  type: mcp                              # only "mcp" in v1
  command: agent-triage-adapter-phoenix  # selects the backend (suffix after
                                         # "agent-triage-adapter-" or a bare
                                         # name: phoenix | langfuse | langsmith)
  env:                                   # backend-specific settings, see tables below
    PHOENIX_URL: http://localhost:6006

# Optional: where issues go. Omit to queue drafts locally with no dedup.
tracker:
  type: mcp
  command: agent-triage-adapter-github   # jira | linear | github
  env:
    GITHUB_TOKEN: ghp_...
    GITHUB_OWNER: my-org
    GITHUB_REPO: agents

# Required: the rubric to classify against (path, file:// URI, or builtin URI).
rubric: agent-triage.dev/builtin/agents/v1

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
| | `PHOENIX_MAX_LIST_PAGES` | no | pagination ceiling (positive int) |
| langfuse | `LANGFUSE_HOST` | yes | cloud or self-hosted URL |
| | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | for auth | key pair |
| | `LANGFUSE_MAX_LIST_PAGES` | no | |
| langsmith | `LANGSMITH_API_KEY` | yes | |
| | `LANGSMITH_ENDPOINT` | no | default `https://api.smith.langchain.com` |
| | `LANGSMITH_PROJECT` | no | project/session filter |
| | `LANGSMITH_MAX_LIST_PAGES` | no | |

### Tracker `env` keys

| Tracker | Key | Required | Notes |
|---|---|---|---|
| jira | `JIRA_HOST` + `JIRA_PROJECT` | yes | host URL + project key |
| | `JIRA_EMAIL` + `JIRA_API_TOKEN` | Cloud auth | Basic auth pair |
| | `JIRA_PAT` | Data Center auth | personal access token |
| | `JIRA_DEPLOYMENT` | no | `cloud`/`datacenter`; default auto-detects from hostname |
| | `JIRA_MAX_LIST_PAGES` | no | |
| linear | `LINEAR_API_KEY` + `LINEAR_TEAM_ID` | yes | team **UUID**, not the key |
| | `LINEAR_ENDPOINT` | no | default `https://api.linear.app/graphql` |
| | `LINEAR_MAX_LIST_PAGES` | no | |
| github | `GITHUB_TOKEN` + `GITHUB_OWNER` + `GITHUB_REPO` | yes | PAT needs Issues read/write |
| | `GITHUB_API_URL` | no | set for GitHub Enterprise Server |
| | `GITHUB_MAX_LIST_PAGES` | no | |

`*_MAX_LIST_PAGES` semantics: when a listing hits the ceiling with a full
last page, the adapter flags `truncated=True`, warns once, and the run
report carries a truncation banner. During tracker dedup a truncated
listing demotes would-be auto-posts to the local queue (a duplicate
can't be ruled out). A malformed value is a `ConfigError`, never a
silent fallback.

## Process environment variables

| Variable | Read by | When needed |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic SDK | `--provider anthropic` (default) with any `llm_judge` mode, and `self-test` |
| `OPENAI_API_KEY` | OpenAI SDK | `--provider openai`, **and always for clustering** — embeddings are OpenAI-only in v1, and the pipeline preflights this key before classifying so it fails fast rather than after spend |
| `EDITOR` | `--review` flow | optional; without it, drafts print to stdout with a y/n prompt |
| Adapter variables (`PHOENIX_URL`, `GITHUB_TOKEN`, …) | **standalone MCP binaries only** | when running `agent-triage-adapter-*` directly; the `run`/`serve` CLI takes these from flags or the config `env:` blocks instead |

## Built-in defaults worth knowing

| Setting | Default | Where |
|---|---|---|
| Judge model (anthropic) | `claude-haiku-4-5-20251001` | `agent_triage.llm.DEFAULT_ANTHROPIC_MODEL` |
| Judge model (openai) | `gpt-4o-mini` | `agent_triage.llm.DEFAULT_OPENAI_MODEL` |
| Embedding model | `text-embedding-3-small` | `agent_triage.llm.DEFAULT_OPENAI_EMBEDDING_MODEL` (rubrics can override via `clustering.embedding_model`) |
| Classification concurrency | 8 | `--concurrency` |
| Classifier retries | 3, exponential backoff | then the trace is marked unprocessed, run continues |
| Backend write retries | 5 | then the run aborts (no partial-write runs) |
| Trace cap | 1000 | `max_traces_per_run` / `--max-traces` |
| Queue directory | `~/.agent-triage/queued-issues/` | `--queue-dir` |
| Self-instrumentation endpoint | `http://localhost:6006` | `--instrument-to` default target when enabled |
| Price table for `--dry-run` | baked per-model table | `agent_triage/cost.py` (single-file edit to update) |

## Secrets handling

Credentials never appear in logs, annotations, reports, drafted issues,
or the agent-mode virtual filesystem; missing/invalid credentials abort
at startup naming the missing *variable*. Full posture:
[SECURITY.md](../SECURITY.md).

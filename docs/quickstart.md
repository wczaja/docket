# Quickstart

This guide is a ladder — each rung adds exactly one credential — and
then a matrix: every supported backend × tracker pair, the `--review`
flow, and the `auto_post_threshold` gate. If you only need the
condensed path, the [README](../README.md) covers rungs 0–3 in one
screen; this document is for picking your own combination.

## Rung 0 — the demo (no credentials, no Docker)

```bash
uvx docket-runtime demo          # or: pipx run docket-runtime demo
```

Runs the real pipeline — classify → cluster → draft → report — over 60
bundled synthetic traces with a clearly-labeled scripted judge. Free,
offline, deterministic. Useful variants:

```bash
docket demo --live                          # real judge, one API key
docket demo --rubric ./my-rubric.yaml       # your taxonomy, same traces
docket demo --to-phoenix http://localhost:6006   # seed a real Phoenix instead
```

## Prerequisites for real runs

```bash
pip install docket-runtime          # or: uv tool install docket-runtime

export ANTHROPIC_API_KEY=...      # for llm_judge detectors (or OPENAI_API_KEY)
```

One key is enough. Clustering defaults to OpenAI embeddings, but three
paths avoid the second vendor account:

```bash
--clustering mode-only                        # no embeddings at all (lossy:
                                              # one cluster per firing mode)
--embedding local:BAAI/bge-small-en-v1.5      # local ONNX, no key:
                                              # pip install "docket-runtime[local-embeddings]"
--embedding voyage:voyage-3.5-lite            # Voyage, with VOYAGE_API_KEY
```

Python 3.11 or newer. No system services are required beyond the
backend / tracker of your choice. `docket init` scaffolds a
`docket.yaml` interactively if you'd rather answer four prompts than
compose flags.

## Configure a trace backend

docket reads traces; it does not host them. Pick the backend you
already use:

### Phoenix (local Docker, simplest for dev)

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

```bash
docket run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  ...
```

No instrumented app yet? `docket demo --to-phoenix
http://localhost:6006` seeds the demo traces so the pipeline has
something real to chew on. See [docs/local-phoenix.md](local-phoenix.md)
for ingestion via OTLP and the OpenInference instrumentation packages.

### Langfuse

```bash
docket run \
  --backend langfuse \
  --langfuse-host https://cloud.langfuse.com \
  --langfuse-public-key "$LANGFUSE_PUBLIC_KEY" \
  --langfuse-secret-key "$LANGFUSE_SECRET_KEY" \
  ...
```

See [docs/local-langfuse.md](local-langfuse.md) for self-hosted setup.

### LangSmith

```bash
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project agents-v1 \
  ...
```

See [docs/local-langsmith.md](local-langsmith.md).

## Configure a tracker (optional)

If no `--tracker` flag is set, docket queues drafts to local files
and stops. To dedup against an existing tracker and (optionally)
auto-post:

### Jira (Cloud or Data Center)

```bash
# Cloud
docket run ... \
  --tracker jira \
  --jira-host https://example.atlassian.net \
  --jira-project AGT \
  --jira-email "$JIRA_EMAIL" \
  --jira-api-token "$JIRA_API_TOKEN"

# Data Center
docket run ... \
  --tracker jira \
  --jira-host https://jira.internal.example.com \
  --jira-project AGT \
  --jira-pat "$JIRA_PAT"
```

Deployment auto-detects from the hostname (`atlassian.net` → Cloud,
anything else → Data Center). Override with `--jira-deployment`. See
[docs/local-jira.md](local-jira.md) for ADF body limitations + state
transitions.

### Linear

```bash
docket run ... \
  --tracker linear \
  --linear-api-key "$LINEAR_API_KEY" \
  --linear-team "$LINEAR_TEAM_ID"
```

Note that Linear keys go in the `Authorization` header **without** a
`Bearer` prefix. The `--linear-team` value is the UUID, not the team
key. See [docs/local-linear.md](local-linear.md).

### GitHub Issues

```bash
docket run ... \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner my-org \
  --github-repo agents
```

For GitHub Enterprise Server, add `--github-api-url
https://github.acme.internal/api/v3`. See
[docs/local-github.md](local-github.md).

## Configure a rubric

Pick a built-in:

```bash
--rubric docket.dev/builtin/agents/v1      # generic agents (6 modes)
--rubric docket.dev/builtin/rag/v1         # retrieval failures
--rubric docket.dev/builtin/routing/v1     # router / supervisor
--rubric docket.dev/builtin/multi-agent/v1 # multi-agent coordination
--rubric docket.dev/builtin/mast/v1        # MAST taxonomy (7 modes)
```

Or point at your own YAML (compose by `imports:` at the top — see
[`docs/design.md`](design.md) §3):

```bash
--rubric ./rubrics/my-agent.yaml
```

Validate before running:

```bash
docket validate ./rubrics/my-agent.yaml
```

## Three modes of operation

### A. Read-only (default)

```bash
docket run \
  --backend phoenix --phoenix-url http://localhost:6006 \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

- Reads traces (no writeback to backend).
- Classifies + clusters + drafts.
- Writes drafts to `~/.docket/queued-issues/<run-id>/`.
- Writes a markdown report to the same directory.
- **Posts nothing to a tracker** (no `--tracker` flag).

Good for first runs. Inspect the queued files; if the drafts look
reasonable, layer a tracker on top.

### B. Dedup against tracker, queue new drafts locally

```bash
docket run ... \
  --tracker github --github-owner my-org --github-repo agents
```

- Same as (A) but also queries the tracker for open issues matching
  each draft's labels.
- If a match exists and the cluster gained new traces, posts a comment
  listing only the new trace IDs.
- If a match exists and the cluster is unchanged, skips silently
  (idempotent — re-runs post nothing extra).
- If no match, queues the draft locally for human review.

This is the safest mode once a tracker is configured: comments are
additive (a human is already on the issue), and new issues require
explicit action.

### C. Interactive review (`--review`)

```bash
docket run ... --tracker github ... --review
```

For each `needs_create` outcome, the runtime:

1. Opens the draft markdown in `$EDITOR`.
2. On editor exit, re-parses the title and Description.
3. Prompts `y/n` to post.
4. Posts accepted drafts; leaves rejected ones in the queue.

When `$EDITOR` is unset, the runtime prints the draft to stdout and
prompts y/n without an editor step. Use this in CI or minimal
containers.

### D. Auto-post above a severity threshold

```bash
docket run ... --tracker github ... --auto-post-threshold high
```

Drafts whose cluster severity is `high` or `critical` are posted
automatically. Lower-severity drafts stay in the queue. This is the
"run docket in CI" mode — once you've calibrated the rubric and
trust the false-positive rate, you can ratchet the threshold down.

Combine with `--review` to auto-post above threshold AND open lower-
severity drafts for review in the same run.

## Config file

The full flag set is large; for production use, write an
`docket.yaml`:

```yaml
trace_backend:
  type: mcp
  command: docket-adapter-phoenix
  env:
    PHOENIX_URL: http://localhost:6006

tracker:
  type: mcp
  command: docket-adapter-github
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}
    GITHUB_OWNER: my-org
    GITHUB_REPO: agents

rubric: docket.dev/builtin/agents/v1
auto_post_threshold: high
```

Then:

```bash
docket run --config docket.yaml --since 1h
```

CLI flags always override config values, so you can keep prod settings
in YAML and override `--auto-post-threshold never` or `--review` for ad
hoc runs.

## Cost controls and sampling

Classification issues one LLM call per `(trace, mode)` pair, so cost
scales with the window. Two budget gates bound every run:

- `max_traces_per_run` (default 1000, override per-run with
  `--max-traces`) caps the effective workload — measured after sampling
  and checkpoint subtraction. Exceeding it aborts *before any trace is
  fetched*; silent truncation is forbidden.
- `max_estimated_cost_usd` (config, optional) aborts any run whose
  pre-flight cost estimate exceeds the ceiling.

Preview a run and gate CI on it — `--dry-run` exits non-zero iff the
real run would abort:

```bash
docket run --config docket.yaml --since 24h --dry-run
```

For windows too large to enumerate, sample:

```bash
# 100 seeded-random traces from the window
docket run ... --sample 100

# only traces whose root span errored (filter pushed down to the backend)
docket run ... --sample 100 --strategy errors-only

# equal allocation across tenants, so small tenants get seen
docket run ... --sample 90 --strategy stratified --stratify-by tag:tenant_id
```

Sampling is seeded by the `run_id`, so re-runs of the same window (and
`--checkpoint` resumes) draw the same traces. If a backend stops
paginating at its page ceiling, the listing is flagged as truncated in
the logs and the run report rather than passing itself off as the whole
window.

## What happens on re-run

docket is designed to be re-run safely. The `run_id` for a given
`(backend, rubric, since, until)` window is deterministic:

```
run_id = sha256(f"{backend_id}|{rubric_id}@{version}|{since}|{until}")[:16]
```

Re-running the same window:

- Reuses the same `run_id` for annotations → the backend upserts (no
  duplicate annotations).
- Reuses the same draft `cluster_id`s → the tracker dedup loop matches
  existing open issues by labels + provenance.
- Skips silently when the existing issue already lists every current
  cluster member; comments only when new traces have appeared.
- If the tracker's open-issue listing is truncated at its page ceiling
  (very large backlogs), drafts that would have auto-posted are queued
  as `needs_create` instead — "no duplicate found" wasn't proven — and
  the report says so. Raise `GITHUB_/JIRA_/LINEAR_MAX_LIST_PAGES` in the
  tracker's env block or prune old docket issues.

This is what makes scheduled runs (`docket run --since 1h` every
hour from cron) safe: each run only surfaces new failures.

## Troubleshooting

- **`docket validate` exits non-zero** — your rubric YAML has a
  schema violation. The error message points to the field.
- **`self-test` skips every mode** — the modes use deterministic
  detectors (`regex`, `tool_call`, `metric_threshold`); self-test only
  exercises `llm_judge` examples. That's not an error.
- **`pytest --run-integration` passes locally but CI's nightly fails**
  — usually a credentials issue. The integration jobs read API keys from
  GitHub Actions secrets; verify `ANTHROPIC_API_KEY`, `PHOENIX_URL`, and
  the tracker creds are set.
- **Drafts look generic / repetitive** — the drafter prompt is in
  `docket/agent/subagents/drafter.py`. For project-specific
  framing, fork the prompt or extend the rubric's `description` block
  (which the drafter incorporates).

## Next steps

- Read [`docs/design.md`](design.md) — the architectural decisions and
  the phase plan.
- Look at the built-in rubrics
  (`docket/rubric/builtin/*/v1/rubric.yaml`) as templates for your
  own.
- Schedule it once you trust the false-positive rate: either wire
  `docket run` into cron / GitHub Actions / Argo, or run the
  built-in daemon — `docket serve --interval 1h` with the same
  flags as `run`. The daemon tiles consecutive windows exactly (a failed
  tick retries its window instead of dropping it), which is the fiddly
  part of doing it with cron.

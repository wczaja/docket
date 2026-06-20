# End-to-end testing guide

This guide walks through a full end-to-end test of `docket` from
the terminal. The example uses **LangSmith** as the trace backend and
**GitHub Issues** as the tracker (the combination explicitly called out
in the design), but every step has analogues for Phoenix / Langfuse and
Jira / Linear (links throughout).

By the end, you'll have:

1. A LangSmith project populated with 60 synthetic traces (20 clean,
   40 seeded with known failure modes).
2. A GitHub repository that receives `docket`'s drafted issues.
3. Verified the dedup loop is idempotent across re-runs.
4. Verified the `--review` flow works end to end.

The whole thing takes ~15 minutes once you have the credentials.

---

## Do we need a trace simulator?

**Yes — and one already ships with the project.**
`docket._acceptance` produces 60 deterministic synthetic traces
(20 clean, 40 with seeded failures — 8 each across `hallucination`,
`infinite-loop`, `premature-termination`, `unsafe-tool-call`,
`refusal-leakage`). Two posting scripts then push these into a
real observability backend:

| Backend     | Script                                                |
| ----------- | ----------------------------------------------------- |
| Phoenix     | `scripts/ingest_acceptance_traces.py`                 |
| LangSmith   | `scripts/ingest_acceptance_traces_langsmith.py`       |
| Langfuse    | Not yet shipped — instrument a real agent for now     |

The fixtures are good enough to exercise every detector type
(`llm_judge`, `regex`, `tool_call`, `metric_threshold`, `composite`)
and the full pipeline (classify → cluster → draft → dedup → post).
For your own production rubrics, you'll eventually want real traces
from a real agent, but the synthetic fixture is the right starting
point for verifying that the runtime is wired up correctly.

---

## 0. Prerequisites

- Python 3.11 or newer.
- A LangSmith account (free tier is fine for testing) —
  <https://smith.langchain.com>.
- A GitHub account.
- An Anthropic API key (or OpenAI key) for the LLM-judge detectors.

```bash
python --version       # >= 3.11
```

---

## 1. Install `docket`

From a development checkout:

```bash
git clone https://github.com/wczaja/docket.git
cd docket
uv pip install -e ".[dev]"     # or:  pip install -e ".[dev]"
docket --version
```

Or from PyPI (once `v0.1.0` is published):

```bash
pip install docket-runtime
docket --version
```

Confirm the CLI is discoverable:

```bash
docket --help
docket run --help | head -20
```

---

## 2. Set up LangSmith

### 2.1 Create an API key

1. Sign in at <https://smith.langchain.com>.
2. Top-right avatar → **Settings → API Keys**.
3. **Create API Key**, label it `docket-e2e`, copy the value.
   Keys start with `lsv2_pt_` or `ls__`.

### 2.2 Create a project

Projects are LangSmith's term for what other backends call sessions or
workspaces. Each trace belongs to exactly one project.

1. Left nav → **Tracing Projects** → **+ New Project**.
2. Name it `docket-e2e`. Description and tags are optional.
3. Open the project — it should show zero traces.

### 2.3 Export credentials

```bash
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT="docket-e2e"
```

> **Alternative backends:** if you'd rather use Phoenix
> (Docker, local) or Langfuse, see `docs/local-phoenix.md` or
> `docs/local-langfuse.md`. The rest of this guide is identical
> except for the `--backend` flag and the trace-ingest script.

---

## 3. Seed the project with synthetic traces

Use the LangSmith ingest script that ships with the project. It posts
60 synthetic traces into the project you just created:

```bash
python scripts/ingest_acceptance_traces_langsmith.py \
  --project "$LANGSMITH_PROJECT"
```

Expected output:

```
acceptance summary: {"total": 60, "clean": 20, "seeded_failures": 40, "modes_seeded": ["hallucination", "infinite-loop", "premature-termination", "refusal-leakage", "unsafe-tool-call"]}

OK    clean-0         expected=clean                     trace_id=...
OK    leak-0          expected=refusal-leakage           trace_id=...
OK    unsafe-0        expected=unsafe-tool-call          trace_id=...
OK    loop-0          expected=infinite-loop             trace_id=...
OK    halluc-0        expected=hallucination             trace_id=...
OK    prem-0          expected=premature-termination     trace_id=...
...
Ingested 60 traces.
```

Refresh the LangSmith UI — the project should now show 60 root runs.
Click into one of the seeded-failure traces (e.g. `loop-0` or
`halluc-0`); the run tree should match what the script printed.

---

## 4. Set up GitHub Issues

### 4.1 Create a test repository

Use an existing repo or create one specifically for triage drafts so
they don't clutter a real backlog:

```bash
# Via the GitHub UI: create a new repository called `docket-test`
# (private is fine — docket only needs API access, not public).
```

### 4.2 Create a personal access token

Fine-grained token (recommended):

1. <https://github.com/settings/tokens?type=beta> → **Generate new token**.
2. Name: `docket-e2e`. Expiry: 7 days for testing.
3. **Repository access** → **Only select repositories** → pick
   `docket-test`.
4. **Permissions → Repository**:
   - **Issues**: Read and write
   - **Metadata**: Read-only (mandatory)
5. Generate, copy the `github_pat_...` value.

Classic token also works — give it the `repo` scope and copy the
`ghp_...` value.

### 4.3 Export credentials

```bash
export GITHUB_TOKEN="github_pat_..."     # or ghp_...
export GITHUB_OWNER="your-gh-username"   # or your org name
export GITHUB_REPO="docket-test"
```

> **Alternative trackers:** for Jira or Linear, see `docs/local-jira.md`
> or `docs/local-linear.md`. The rest of this guide is identical except
> for the `--tracker` flag and the credential env vars.

---

## 5. Set up the LLM provider

The built-in `agents/v1` rubric uses `llm_judge` detectors for
`hallucination` and `premature-termination`. They default to a fast
Anthropic model:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

To use OpenAI instead, pass `--provider openai --model gpt-4o-mini` on
the CLI and export `OPENAI_API_KEY` instead.

---

## 6. Run the full pipeline (read-only first)

Start with a read-only run to verify the wiring before posting
anything to GitHub. Omit `--tracker` so drafts queue locally only:

```bash
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project "$LANGSMITH_PROJECT" \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

You'll see output similar to:

```
Pulled 60 traces from langsmith
Classifying 60 traces against agents-builtin v1.0.0 ...
  classified 60/60
Clustering ...
  formed 5 clusters across 5 modes
Drafting ...
  drafted 5 issues into ~/.docket/queued-issues/<run-id>/
Report:
# docket run `<run-id>`
- Rubric: agents-builtin v1.0.0
- Window: ...
- Traces processed: 60
...
```

### What to verify

- **Trace count**: `60` (matches the fixture).
- **Mode hits**: the report's "Frequency by mode" table should show
  positive counts for `hallucination`, `infinite-loop`,
  `refusal-leakage`, `unsafe-tool-call`, `premature-termination`,
  with 8 positives per seeded mode on the 60-trace fixture.
- **Cluster count**: usually 5 (one per seeded mode), depending on
  the embedding-clusterer's threshold.
- **Queue**: `ls ~/.docket/queued-issues/<run-id>/` should
  list one `.json` + one `.md` per cluster.

Open one of the `.md` drafts:

```bash
cat ~/.docket/queued-issues/<run-id>/cluster-*.md | head -50
```

It should have a title (LLM-generated, depends on the fixture trace),
a metadata block, a `## Description` section, and an
`docket:provenance` HTML comment at the bottom.

---

## 7. Layer the tracker on top

Now add `--tracker github` so the runtime queries GitHub for matching
open issues. With no `--auto-post-threshold`, this is dedup-only —
nothing gets created in GitHub yet, but the runtime checks for matches:

```bash
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project "$LANGSMITH_PROJECT" \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner "$GITHUB_OWNER" \
  --github-repo "$GITHUB_REPO" \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

### What to verify

- The output's `## Tracker dedup` table lists every cluster with
  `action=needs_create` (no existing issues match yet).
- No issues have been posted to GitHub yet — refresh the repo's
  Issues tab; it should still be empty.

This is the safest mode: dedup runs, but nothing gets created. New
drafts stay in the local queue for human inspection.

---

## 8. Auto-post above a threshold

Now ask the runtime to post any draft whose cluster severity is
`high` or `critical`. The `agents/v1` rubric flags `hallucination` as
`critical` and `infinite-loop` / `unsafe-tool-call` /
`premature-termination` / `bad-handoff` as `high`:

```bash
docket run \
  --backend langsmith ... \
  --tracker github ... \
  --rubric docket.dev/builtin/agents/v1 \
  --auto-post-threshold high \
  --since 1h
```

### What to verify

- Output `## Tracker dedup` now shows multiple `action=created`
  entries (one per cluster above threshold), each linking to the
  created GitHub issue.
- Refresh the GitHub repo's Issues tab — you should see new issues
  with:
  - Titles matching the cluster summaries.
  - Bodies containing the cluster description plus an HTML
    `docket:provenance` comment at the end.
  - Labels: `docket`, `mode:<id>`,
    `rubric:agents-builtin@1.0.0`.

If `refusal-leakage` had severity `medium`, its cluster stays in the
local queue (below the `high` threshold). Drop the threshold to
`medium` to include it.

---

## 9. Re-run to verify idempotency

Run the **same command** again:

```bash
docket run \
  --backend langsmith ... \
  --tracker github ... \
  --rubric docket.dev/builtin/agents/v1 \
  --auto-post-threshold high \
  --since 1h
```

### What to verify

- The deterministic `run_id` is the same (it's a hash of
  `backend|rubric|since|until`).
- `## Tracker dedup` now shows `action=skipped` for every cluster
  that was previously created. No new issues land in GitHub.
- No new comments are posted either — the existing issues' provenance
  blocks already list every member trace ID, so the dedup loop knows
  the clusters are unchanged.

This is what makes scheduled runs safe.

---

## 10. Simulate cluster growth

Re-run the ingest script to add a second batch of synthetic traces.
Each invocation generates new random trace IDs, so the second batch's
traces get added to the same clusters (the cluster_id is a content
hash, not a trace_id hash):

```bash
python scripts/ingest_acceptance_traces_langsmith.py \
  --project "$LANGSMITH_PROJECT"
```

Now re-run the triage with the *same* time window so it picks up both
batches:

```bash
docket run \
  --backend langsmith ... \
  --tracker github ... \
  --rubric docket.dev/builtin/agents/v1 \
  --auto-post-threshold high \
  --since 2h
```

### What to verify

- `## Tracker dedup` shows `action=commented` for the previously
  created clusters.
- The GitHub issues each gain a new comment listing only the new
  trace IDs — the diff between the cluster's current membership and
  what the issue's provenance already recorded.

---

## 11. Try the `--review` flow

Reset by lowering the threshold so a fresh cluster lands in
`needs_create` again, then add `--review`:

```bash
EDITOR=vim docket run \
  --backend langsmith ... \
  --tracker github ... \
  --rubric docket.dev/builtin/agents/v1 \
  --review \
  --since 1h
```

For each `needs_create` draft:

1. The runtime opens the draft markdown in `$EDITOR`.
2. Edit the title / Description if you want, save and exit.
3. The runtime re-parses, shows a preview, and prompts `Post draft for
   cluster <id>? [y/N]`.
4. Type `y` to post or `N` to leave it in the local queue.

When `$EDITOR` is unset (e.g. on a CI runner), the runtime prints the
draft to stdout and skips the editor step but still prompts y/N.

---

## 12. Try the `--agent` mode

The default `docket run` path is a deterministic Python pipeline:
the stages run in a fixed order, no LLM-driven planning, no extra LLM
calls beyond the detectors and the drafter. That's the recommended
path for batch / CI / production use.

`--agent` opts into the `deepagents`-driven harness. The same six
pipeline stages (`list_traces`, `classify_traces`,
`annotate_classifications`, `cluster_classifications`,
`draft_issues_tool`, `write_report`) become LangChain tools, and a
top-level LLM (default: `anthropic:claude-haiku-4-5-20251001`) plans
the tool calls. The agent reasons over a virtual filesystem and the
final `/report.md` is extracted at the end.

For the canonical framing of when each mode is intended to be used,
see [`docs/design.md`](design.md) §4.2 — both modes share the same
subagents and produce the same artifacts; they differ in *who* decides
the order. The summary below is the operational view for this guide.

**`--agent` is useful today for:**

- Watching the LLM's planning step by step when debugging the harness
  itself or evaluating how the planning model handles a noisy pipeline.
- Demonstrating the `deepagents` integration to stakeholders.
- Smoke-testing changes to the harness or the system prompt without
  the cost of running the deterministic pipeline at scale.

**Don't reach for `--agent` when you want:**

- **Production runs.** The deterministic pipeline is the production
  execution model. `--agent` adds planning-LLM cost and nondeterminism
  for orchestration that doesn't currently need orchestration.
- **Predictable cost / runtime.** Every tool call adds a planner
  round-trip.
- **Tracker dedup / posting / review.** `--agent` does **not** invoke
  the poster subagent in v1.0. The agent flow ends at `write_report`;
  drafts land in the local queue but no tracker integration runs. Use
  the default deterministic pipeline for that.
- **Interactive triage, incident investigation, rubric authoring, or
  cross-backend composition.** These are the use cases the deepagents
  harness is *intended* to unlock, but the tools and entry points for
  them are post-v1.0 work — see [`docs/design.md`](design.md) §7
  Phase 14 (tool surface expansion) and Phase 15 (interactive
  surfaces). The v1.0 `--agent` harness only runs the same six-stage
  workflow the deterministic path runs.

### 12.1 Same setup, add `--agent`

LangSmith and the LLM provider are already configured from sections 2
and 5. Add `--agent` to the read-only command from section 6:

```bash
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project "$LANGSMITH_PROJECT" \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h \
  --agent
```

You don't need a fresh trace ingest — re-using the 60 synthetic
traces from section 3 is exactly the point.

### 12.2 What you'll see

The agent emits a stream of messages as it plans + invokes each tool.
A truncated example:

```
=== Agent ===
I'll run the full workflow. Starting with list_traces.

→ list_traces({"since": "...", "until": "..."})
← list_traces returned 60 trace IDs. Stored on agent state.

→ classify_traces({})
← classify_traces ran every mode against every trace. 40/60 traces had
  at least one positive classification.

→ cluster_classifications({})
← cluster_classifications grouped positives by embedding similarity per
  mode. 5 clusters across 5 modes (sizes: 8, 8, 8, 8, 6).

→ draft_issues_tool({})
← draft_issues_tool generated 5 drafts and queued them to
  ~/.docket/queued-issues/<run-id>/.

→ write_report({})
← write_report stored the markdown summary at /report.md (1.2 KB).

The workflow is complete. Final report:

# docket run `<run-id>`
- Rubric: agents-builtin v1.0.0
...
```

The exact prose varies — the harness is LLM-driven. The agent will
skip `annotate_classifications` automatically because you didn't pass
`--annotate`.

### 12.3 Verify the outcomes match the deterministic pipeline

Run the deterministic pipeline (same flags, no `--agent`) against the
same window and diff the queued drafts:

```bash
docket run \
  --backend langsmith ... \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
# note the queue dir from the output:
# "drafted 5 issues into ~/.docket/queued-issues/<run-id>/"

docket run \
  --backend langsmith ... \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h \
  --agent
# note the second queue dir
```

Compare:

```bash
diff -r \
  ~/.docket/queued-issues/<deterministic-run-id>/ \
  ~/.docket/queued-issues/<agent-run-id>/
```

The titles and bodies are LLM-drafted, so they won't match
character-for-character across runs (different `run_id`s seed
different prompts), but the **set of clusters** (number of clusters,
cluster_id values, member trace IDs, severities, modes) should be
identical. The deterministic pipeline and the agent run the same
underlying subagents in the same order.

### 12.4 Customize the agent's planning model

The planning model defaults to Anthropic Haiku. The CLI doesn't expose
a flag for this in v1.0 — to swap it (e.g. to a stronger model for
reasoning over a noisier pipeline), call `build_triage_agent` directly
in Python:

```python
from docket.agent.deep_agent import build_triage_agent

agent, state = build_triage_agent(
    backend=...,
    rubric=...,
    llm_provider=...,
    embedding_provider=...,
    since=...,
    until=...,
    agent_model="anthropic:claude-sonnet-4-6",   # default is haiku
)
```

A CLI flag for this is a v1.1 follow-up.

### 12.5 Troubleshooting `--agent`

- **Agent loops or skips stages**: the system prompt directs the agent
  to run the six tools in order. If the LLM gets confused (rare with
  Haiku), it usually misses `write_report` at the end. Re-run with a
  stronger model, or fall back to the deterministic pipeline.
- **`/report.md` is empty**: the agent didn't call `write_report`.
  Check the streamed messages — the agent may have ended early
  because an earlier tool returned an unexpected status. The
  fallback string is `"(deep agent run produced no /report.md)"`.
- **Higher API bill than expected**: planning adds LLM calls on top
  of detector calls. For the 60-trace fixture, expect ~5-10 planning
  calls plus the 120 detector calls. Throttle with `--concurrency 2`
  if needed.
- **`--tracker` or `--review` quietly do nothing**: the agent
  workflow ends at `write_report` in v1.0. Drafts queue locally but
  no tracker dedup runs. Use the deterministic pipeline for tracker
  integration.

---

## 13. Cleanup

After testing, close the GitHub issues:

```bash
# Manually via the UI, or:
gh issue list --repo "$GITHUB_OWNER/$GITHUB_REPO" --label docket \
  --json number --jq '.[].number' \
  | xargs -I{} gh issue close --repo "$GITHUB_OWNER/$GITHUB_REPO" {}
```

The LangSmith project can be deleted from the Settings tab if you
don't want to keep the synthetic traces around.

Revoke the GitHub PAT at <https://github.com/settings/tokens> when
you're done testing, especially if the expiry is more than a day or
two out.

---

## Troubleshooting

### `LANGSMITH_API_KEY required`

The ingest script and the adapter both look for this env var or a
CLI override (`--api-key` / `--langsmith-api-key`). Confirm it's set
in the same shell where you run the command (`echo $LANGSMITH_API_KEY`).

### `Pulled 0 traces`

The `--since` window doesn't overlap the ingest time. Either widen
the window (`--since 24h`) or re-run the ingest script and try again.

### `403` or `404` from GitHub

The PAT doesn't have Issues write on the repo. For fine-grained
tokens, confirm the repo is listed under **Repository access** and
**Issues** permission is set to **Read and write**. For classic
tokens, confirm the `repo` scope is set.

### `Could not resolve authentication method` from Anthropic

`ANTHROPIC_API_KEY` isn't set. Either export it or pass
`--provider openai` + `OPENAI_API_KEY`.

### Clusters look noisy (size 1, lots of one-off issues)

The 60-trace fixture seeds 8 positives per mode — comfortably above the
`min_cluster_size: 3` in the built-in rubric — so the synthetic run
should cluster cleanly. You're more likely to hit this with a sparse
real window that has only one or two positives for a mode. In that case:

- Widen the time window so more positives land in the same run.
- Edit your rubric copy's `clustering.min_cluster_size` down to 2.

This matters less with real production traces, where you typically
have many positives per mode in a given window.

### LLM bills

LLM-judge detectors call the provider once per `(trace, mode)` pair.
60 traces × 2 judge modes = 120 calls per run. The default Haiku
model keeps costs to a few cents per run. For batch development, throttle
with `--concurrency 2` to avoid burst spikes.

---

## What's tested by this guide

- **Trace backend integration**: LangSmith adapter pulls traces from a
  real LangSmith project, normalizes them to OpenInference.
- **Detector pipeline**: every detector type
  (`llm_judge`, `regex`, `tool_call`, `metric_threshold`,
  `composite`) runs against the fixture and classifies traces.
- **Clustering**: positive classifications get grouped by embedding
  similarity per mode.
- **Drafter**: the LLM produces a per-cluster IssueDraft with
  embedded provenance.
- **Tracker integration**: the GitHub adapter posts issues, dedups by
  label + provenance, and comments on cluster growth.
- **Idempotency**: re-running the same window posts nothing extra.
- **Auto-post threshold**: severity-based gating works.
- **Review mode**: `$EDITOR` + accept/reject + post flow works.

- **Deep Agents harness (`--agent`)**: optional LLM-driven planning
  over the same pipeline stages; demonstrates the harness integration
  and lets you watch step-by-step tool calls. Note that tracker
  integration is not wired into the agent flow in v1.0 — use the
  deterministic pipeline for that.

Things this guide does **not** cover:

- Annotation writeback (`--annotate`). The fixture's clean / failed
  classifications are reliable, but writing annotations back to
  LangSmith uses the feedback API which the gated unit tests cover.
- Self-test (`docket self-test`) on your own rubric. Run it
  after editing any `llm_judge` mode in a custom rubric to confirm
  the examples still classify correctly.

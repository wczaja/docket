#!/usr/bin/env bash
#
# demo.sh — drive docket's end-to-end flow as a screen-recordable demo.
#
# Uses LangSmith as the trace backend (the project's primary e2e path) plus the
# acceptance fixture and its LangSmith ingest script, so the run is
# deterministic and reproducible. Three beats, one per ENTER press, paced for a
# ~75s recording:
#
#   1. seed 60 synthetic traces (20 clean + 40 seeded failures) into LangSmith
#   2. triage and post — classify, cluster, draft, then auto-post clusters at
#      'high' severity or above to GitHub (read-only is the default; posting is
#      an explicit opt-in)
#   3. re-run the identical command -> idempotent no-op (every cluster skipped)
#
# Full walkthrough, recording, and posting notes: docs/demo.md
#
# Prerequisites:
#   - docket installed             uv pip install -e ".[dev]"   (or pip)
#   - LANGSMITH_API_KEY            a free LangSmith account (smith.langchain.com)
#   - LANGSMITH_PROJECT            project name (default: docket-demo)
#   - ANTHROPIC_API_KEY            llm_judge detectors
#   - OPENAI_API_KEY               clustering embeddings
#   - GITHUB_TOKEN / GITHUB_OWNER / GITHUB_REPO   a throwaway repo for drafts
#
# Optional overrides (env): DOCKET_DEMO_RUBRIC, DOCKET_DEMO_CONCURRENCY,
# DOCKET_DEMO_SINCE, LANGSMITH_PROJECT.
#
set -euo pipefail

RUBRIC="${DOCKET_DEMO_RUBRIC:-docket.dev/builtin/agents/v1}"
CONCURRENCY="${DOCKET_DEMO_CONCURRENCY:-8}"
SINCE="${DOCKET_DEMO_SINCE:-1h}"
LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-docket-demo}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }
pause() { printf '\n\033[2m── press ENTER for the next step ──\033[0m'; read -r _; clear; }

require_env() {
  local missing=0 var
  for var in "$@"; do
    if [ -z "${!var:-}" ]; then
      printf '\033[31mERROR\033[0m: %s is not set.\n' "$var" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "Set the variables above, then re-run. See docs/demo.md for setup." >&2
    exit 1
  fi
}

require_env LANGSMITH_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY \
  GITHUB_TOKEN GITHUB_OWNER GITHUB_REPO
if ! command -v docket >/dev/null 2>&1; then
  echo "ERROR: 'docket' is not on PATH. Install with: uv pip install -e \".[dev]\"" >&2
  exit 1
fi
cd "$REPO_ROOT"

# ── Title ───────────────────────────────────────────────────────────────────
clear
bold "docket — an observability-agnostic triage runtime for LLM agent traces"
dim  "reads traces  ▸  classifies failure modes  ▸  clusters  ▸  drafts tracker issues"
pause

# ── 1. Seed ───────────────────────────────────────────────────────────────────
dim "# Seed LangSmith with 60 synthetic traces: 20 clean + 40 seeded failures,"
dim "# 8 each across hallucination / infinite-loop / premature-termination /"
dim "# unsafe-tool-call / refusal-leakage."
python scripts/ingest_acceptance_traces_langsmith.py --project "$LANGSMITH_PROJECT"
dim "# (LangSmith indexes asynchronously — give it a few seconds before triage.)"
pause

# ── 2. Triage and post ────────────────────────────────────────────────────────
dim "# One pass does it all: classify every trace, cluster the positives, draft"
dim "# one issue per cluster, then post. Triage is read-only by default — here"
dim "# we opt in with --auto-post-threshold high, so only high+critical clusters"
dim "# post; lower-severity drafts stay in the local queue."
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project "$LANGSMITH_PROJECT" \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner "$GITHUB_OWNER" \
  --github-repo "$GITHUB_REPO" \
  --rubric "$RUBRIC" \
  --since "$SINCE" \
  --concurrency "$CONCURRENCY" \
  --auto-post-threshold high
dim "# -> 4 issues filed: hallucination (critical) + infinite-loop, unsafe-tool-"
dim "#    call, premature-termination (high). refusal-leakage (medium) stays in"
dim "#    the local queue. Switch to the repo's Issues tab now."
pause

# ── 3. Re-run = idempotent no-op ──────────────────────────────────────────────
dim "# Run the EXACT same command again. The run_id is a hash of the inputs,"
dim "# and dedup keys off label + embedded provenance, so re-runs do nothing."
docket run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project "$LANGSMITH_PROJECT" \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner "$GITHUB_OWNER" \
  --github-repo "$GITHUB_REPO" \
  --rubric "$RUBRIC" \
  --since "$SINCE" \
  --concurrency "$CONCURRENCY" \
  --auto-post-threshold high
echo
bold "Every cluster came back action=skipped. Same window, zero new issues —"
bold "which is exactly what makes scheduled triage (docket serve / cron) safe."

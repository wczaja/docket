#!/usr/bin/env bash
#
# demo.sh — drive docket's local end-to-end flow as a screen-recordable demo.
#
# Built on the project's existing acceptance mechanism (the same fixture and
# ingest script the gated integration tests use), so the demo is deterministic
# and reproducible. Four beats, one per ENTER press, paced for a ~75s recording:
#
#   1. seed 60 synthetic traces (20 clean + 40 seeded failures) into Phoenix
#   2. triage them read-only — classify, cluster, draft; nothing leaves the box
#   3. add a GitHub tracker and auto-post clusters at 'high' severity or above
#   4. re-run the identical command -> idempotent no-op (every cluster skipped)
#
# Full walkthrough, recording, and posting notes: docs/demo.md
#
# Prerequisites:
#   - docket installed             uv pip install -e ".[dev]"   (or pip)
#   - Phoenix running locally      docker compose up -d phoenix
#   - ANTHROPIC_API_KEY            llm_judge detectors
#   - OPENAI_API_KEY               clustering embeddings
#   - GITHUB_TOKEN / GITHUB_OWNER / GITHUB_REPO   a throwaway repo for drafts
#
# Optional overrides (env): PHOENIX_URL, DOCKET_DEMO_RUBRIC,
# DOCKET_DEMO_CONCURRENCY, DOCKET_DEMO_SINCE.
#
set -euo pipefail

PHOENIX_URL="${PHOENIX_URL:-http://localhost:6006}"
RUBRIC="${DOCKET_DEMO_RUBRIC:-docket.dev/builtin/agents/v1}"
CONCURRENCY="${DOCKET_DEMO_CONCURRENCY:-8}"
SINCE="${DOCKET_DEMO_SINCE:-1h}"

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

require_env ANTHROPIC_API_KEY OPENAI_API_KEY GITHUB_TOKEN GITHUB_OWNER GITHUB_REPO
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

# ── 1. Seed ─────────────────────────────────────────────────────────────────
dim "# Seed a local Phoenix with 60 synthetic traces: 20 clean + 40 seeded"
dim "# failures, 8 each across hallucination / infinite-loop / premature-"
dim "# termination / unsafe-tool-call / refusal-leakage."
python scripts/ingest_acceptance_traces.py --phoenix-url "$PHOENIX_URL"
pause

# ── 2. Triage, read-only ────────────────────────────────────────────────────
dim "# Triage the window: classify every trace, cluster the positives, draft"
dim "# one issue per cluster. Read-only by default — nothing leaves the box."
docket run \
  --backend phoenix --phoenix-url "$PHOENIX_URL" \
  --rubric "$RUBRIC" \
  --since "$SINCE" \
  --concurrency "$CONCURRENCY"
pause

# ── 3. Post to the tracker ──────────────────────────────────────────────────
dim "# Same run, now with a tracker. Auto-post clusters whose severity is"
dim "# 'high' or above; everything below stays queued locally for review."
docket run \
  --backend phoenix --phoenix-url "$PHOENIX_URL" \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner "$GITHUB_OWNER" \
  --github-repo "$GITHUB_REPO" \
  --rubric "$RUBRIC" \
  --since "$SINCE" \
  --concurrency "$CONCURRENCY" \
  --auto-post-threshold high
dim "# -> 4 issues filed: hallucination (critical) + infinite-loop, unsafe-"
dim "#    tool-call, premature-termination (high). refusal-leakage (medium)"
dim "#    stays in the local queue. Switch to the repo's Issues tab now."
pause

# ── 4. Re-run = idempotent no-op ────────────────────────────────────────────
dim "# Run the EXACT same command again. The run_id is a hash of the inputs,"
dim "# and dedup keys off label + embedded provenance, so re-runs do nothing."
docket run \
  --backend phoenix --phoenix-url "$PHOENIX_URL" \
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

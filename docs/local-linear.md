# Using Linear with agent-triage

Phase 9 adds Linear as a second `Tracker`. Linear is GraphQL-only and
SaaS-only (no self-hosted option), so the adapter speaks plain HTTP via
`httpx` against `https://api.linear.app/graphql`. There's no Docker target;
unit tests use `httpx.MockTransport` and a gated integration test runs
against a real Linear workspace when the maintainer's credentials are set.

## 1. Get credentials

1. Sign in at <https://linear.app> and open **Settings → API → Personal
   API keys**.
2. Click **Create key**, label it `agent-triage`, and copy the value — it's
   only shown once. Linear keys are passed in the `Authorization` header
   **without** a `Bearer` prefix.
3. Find your **team ID** (the UUID, not the team key like `AGT`). The
   easiest path: open any issue in Linear, copy the URL, and click into
   **Settings → API → Sample queries** which lists every team's ID. You
   can also fetch it via the GraphQL `viewer.teams` query.

## 2. Configure agent-triage

CLI flags:

```bash
agent-triage run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --tracker linear \
  --linear-api-key "$LINEAR_API_KEY" \
  --linear-team "$LINEAR_TEAM_ID" \
  --rubric agent-triage.dev/builtin/agents/v1 \
  --since 1h
```

Or `agent-triage.yaml`:

```yaml
trace_backend:
  type: mcp
  command: agent-triage-adapter-phoenix
  env:
    PHOENIX_URL: http://localhost:6006

tracker:
  type: mcp
  command: agent-triage-adapter-linear
  env:
    LINEAR_API_KEY: ${LINEAR_API_KEY}
    LINEAR_TEAM_ID: ${LINEAR_TEAM_ID}
    # Optional; defaults to https://api.linear.app/graphql:
    # LINEAR_ENDPOINT: https://api.linear.app/graphql

rubric: agent-triage.dev/builtin/agents/v1
auto_post_threshold: never
```

| Variable / flag      | Default                                  | Required |
| -------------------- | ---------------------------------------- | -------- |
| `LINEAR_API_KEY`     | (none)                                   | yes      |
| `LINEAR_TEAM_ID`     | (none — Linear team UUID)                | yes      |
| `LINEAR_ENDPOINT`    | `https://api.linear.app/graphql`         | no       |

## 3. What lands where

- **Dedup** — the pipeline queries Linear's `issues` connection filtered by
  team + open states (`backlog`, `unstarted`, `started`) + label names; for
  each candidate it parses the embedded HTML provenance comment and looks
  for a `cluster_id` match. Match → comment on existing issue (or skip if
  no new members); no match → see auto-post / review below.
- **Labels** — Linear labels are first-class workspace entities. The
  adapter resolves the three label strings (`agent-triage`,
  `mode:<id>`, `rubric:<id>@<ver>`) to label IDs on first use, creates any
  missing ones via `issueLabelCreate`, and caches the mapping for the
  process lifetime.
- **Auto-post** (`--auto-post-threshold critical|high|medium|low|never`) —
  drafts whose cluster severity meets the threshold are posted via
  `issueCreate`. Lower-severity drafts stay in the local queue.
- **Review** (`--review`) — same flow as Jira: launches `$EDITOR`, prompts
  accept/reject, posts accepted drafts via `issueCreate`.

## 4. Verification

- Unit tests run against `httpx.MockTransport`; no live Linear needed for
  `pytest`.
- `tests/integration/test_linear_e2e.py` is gated on `--run-integration`
  plus `LINEAR_API_KEY` + `LINEAR_TEAM_ID`. It creates a test issue,
  re-runs to assert idempotent dedup, posts a comment on a grown cluster,
  and leaves the issue in Linear labeled `agent-triage-test` for manual
  cleanup (Linear state transitions are workspace-specific and v1.0
  doesn't drive them).

## 5. Notes / Limitations

- State transitions (Triage → In Progress → Done) are workspace-specific
  in Linear; `update_issue(state=...)` raises `TrackerError` rather than
  guessing the workflow. Use Linear's UI / API to transition issues
  manually.
- Linear stores issue descriptions as markdown directly, so the draft
  body (including the HTML provenance comment) is preserved end-to-end
  without conversion. This is simpler than Jira Cloud's ADF requirement.
- Linear's API rate limit is per-API-key (default 1500 requests/hour);
  if you hit it the adapter surfaces the underlying 429 as a
  `TrackerError`. For very large triage runs, partition with `--since` /
  `--until`.

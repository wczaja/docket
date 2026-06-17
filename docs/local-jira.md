# Using Jira with agent-triage

Phase 8 adds Jira as the first `Tracker`. The adapter speaks plain HTTP via
`httpx` (no Atlassian SDK dependency) and supports both deployment styles:

| Deployment | API     | Body format     | Auth                          |
| ---------- | ------- | --------------- | ----------------------------- |
| Cloud      | REST v3 | ADF (JSON)      | Basic — `email:api_token`     |
| Data Center / Server | REST v2 | plain text / wiki | Bearer Personal Access Token |

The deployment is auto-detected from the hostname (anything ending in
`atlassian.net` is treated as Cloud); pass `--jira-deployment` to override.

## 1. Get credentials

### Jira Cloud

1. Sign in at <https://id.atlassian.com>.
2. Visit **Manage account → Security → API tokens** and click **Create API
   token**. Label it `agent-triage` and copy the value — it's only shown
   once.
3. Note your **Atlassian account email** (the address you sign in with).
   Cloud's Basic-auth scheme uses `email:api_token` as the credential pair.

### Jira Data Center / Server

1. In Jira, click your avatar → **Profile → Personal Access Tokens**.
2. Click **Create token**, set a name (`agent-triage`) and an expiry. The
   token is only shown once.
3. Make sure the user account has **Browse projects**, **Create issues**, and
   **Add comments** permissions on the target project.

## 2. Configure agent-triage

You can pass credentials via CLI flags:

```bash
# Cloud
agent-triage run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --tracker jira \
  --jira-host https://example.atlassian.net \
  --jira-project AGT \
  --jira-email "$JIRA_EMAIL" \
  --jira-api-token "$JIRA_API_TOKEN" \
  --rubric agent-triage.dev/builtin/agents/v1 \
  --since 1h

# Data Center
agent-triage run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --tracker jira \
  --jira-host https://jira.internal.example.com \
  --jira-project AGT \
  --jira-pat "$JIRA_PAT" \
  --rubric agent-triage.dev/builtin/agents/v1 \
  --since 1h
```

Or via `agent-triage.yaml`:

```yaml
trace_backend:
  type: mcp
  command: agent-triage-adapter-phoenix
  env:
    PHOENIX_URL: http://localhost:6006

tracker:
  type: mcp
  command: agent-triage-adapter-jira
  env:
    JIRA_HOST: https://example.atlassian.net
    JIRA_PROJECT: AGT
    JIRA_EMAIL: ${JIRA_EMAIL}
    JIRA_API_TOKEN: ${JIRA_API_TOKEN}
    # For Data Center, omit EMAIL/API_TOKEN and use:
    # JIRA_PAT: ${JIRA_PAT}
    # JIRA_DEPLOYMENT: datacenter   # only when auto-detection misfires

rubric: agent-triage.dev/builtin/agents/v1
auto_post_threshold: never  # `critical` | `high` | `medium` | `low` | `never`
```

| Variable / flag         | Default                                 | Required |
| ----------------------- | --------------------------------------- | -------- |
| `JIRA_HOST`             | (none)                                  | yes      |
| `JIRA_PROJECT`          | (none — Jira project key, e.g. `AGT`)   | yes      |
| `JIRA_EMAIL`            | (none)                                  | Cloud    |
| `JIRA_API_TOKEN`        | (none)                                  | Cloud    |
| `JIRA_PAT`              | (none)                                  | DC       |
| `JIRA_DEPLOYMENT`       | auto (host-based)                       | no       |

## 3. What lands where

- **Dedup** (always on when a tracker is configured) — the pipeline queries
  Jira by labels (`agent-triage`, `mode:<id>`, `rubric:<id>@<version>`) and
  parses the embedded HTML provenance block to find a `cluster_id` match.
  - Match found + cluster has new traces → posts a comment listing the new
    trace IDs only.
  - Match found + no new traces → no-op (re-running is idempotent).
  - No match → see below.
- **Auto-post** (`--auto-post-threshold critical|high|medium|low|never`, or
  the config key `auto_post_threshold`) — when set above `never`, drafts
  whose cluster severity meets the threshold are posted automatically as
  new Jira issues. Lower-severity drafts stay in the local queue under
  `~/.agent-triage/queued-issues/`.
- **Review** (`--review`) — for each `needs_create` outcome, the operator's
  `$EDITOR` is launched on the draft markdown; on save, the title and
  Description section are re-parsed and the operator is asked to accept or
  reject. Accepted drafts are posted; rejected drafts stay in the local
  queue. When `$EDITOR` is unset, the draft is printed to stdout and a
  y/n prompt is shown instead.
- **Provenance** — every posted issue carries an HTML-comment provenance
  block at the end of its body
  (`<!-- agent-triage:provenance {...} -->`) plus the standard label set,
  so future runs can dedup against it cleanly.

## 4. Verification

- Unit tests exercise the adapter against `httpx.MockTransport` and cover
  Cloud + Data Center paths — `pytest` doesn't need a live Jira.
- A gated integration test (`tests/integration/test_jira_e2e.py`) runs end
  to end against a real Jira project when these env vars are set:
  `JIRA_HOST`, `JIRA_PROJECT`, plus either (`JIRA_EMAIL` + `JIRA_API_TOKEN`)
  for Cloud or `JIRA_PAT` for Data Center. The test creates an issue, posts
  a dedup comment, and cleans up by closing the issue (`update_issue` to
  `closed`-equivalent labels — Jira workflows are project-specific, so the
  test issue is left in `Done` and labeled `agent-triage-test`).

## 5. Notes / Limitations

- Jira Cloud REST v3 requires ADF (Atlassian Document Format) for issue
  bodies. The adapter renders each markdown paragraph as one ADF
  paragraph; inline styling (bold, code, links) and bullet lists land as
  plain text in v1.0. A proper md→ADF conversion is a v1.1 follow-up.
- State transitions (open → in progress → done) are project-specific in
  Jira; `update_issue(state=...)` raises `TrackerError` rather than guess
  the workflow. Use Jira's UI / API to transition issues manually.
- `LABELS` semantics — Jira sanitizes label values (no spaces; some
  punctuation is stripped). agent-triage's label set is already safe under
  this sanitization (`agent-triage`, `mode:<id>`, `rubric:<id>@<ver>`).

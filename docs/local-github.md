# Using GitHub Issues with agent-triage

GitHub Issues is the simplest of the three tracker surfaces in v1.0:

- REST v3 against `https://api.github.com` (or your Enterprise Server's
  `/api/v3` base for self-hosted GitHub).
- Markdown bodies are stored verbatim — the HTML provenance comment is
  preserved with no conversion.
- Labels are free strings; GitHub auto-creates any that don't exist on
  first use.
- State is a native `open|closed` flag, so `IssuePatch(state=...)` works
  directly (unlike Jira and Linear, where workflows are project-specific).

## 1. Get a token

You can use either a **classic** personal access token (PAT) or a
**fine-grained** PAT. Fine-grained PATs are preferred for new setups
because they scope to specific repositories.

### Fine-grained PAT (recommended)

1. Visit <https://github.com/settings/tokens?type=beta>.
2. Click **Generate new token**, label it `agent-triage`, and set an
   expiry.
3. Under **Repository access**, pick **Only select repositories** and
   choose the repo you want agent-triage to post into.
4. Under **Permissions → Repository**, grant:
   - **Issues**: Read and write
   - **Metadata**: Read-only (required for any repo-scoped PAT)
5. Generate and copy the token (it starts with `github_pat_`). It's only
   shown once.

### Classic PAT (also works)

1. Visit <https://github.com/settings/tokens>.
2. Click **Generate new token (classic)**, label it `agent-triage`, and
   set an expiry.
3. Select the `repo` scope (or just `public_repo` for public-only
   repositories).
4. Generate and copy the token (it starts with `ghp_`).

## 2. Configure agent-triage

CLI flags:

```bash
agent-triage run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --tracker github \
  --github-token "$GITHUB_TOKEN" \
  --github-owner agent-triage \
  --github-repo agent-triage \
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
  command: agent-triage-adapter-github
  env:
    GITHUB_TOKEN: ${GITHUB_TOKEN}
    GITHUB_OWNER: agent-triage
    GITHUB_REPO: agent-triage
    # For GitHub Enterprise Server:
    # GITHUB_API_URL: https://github.acme.internal/api/v3

rubric: agent-triage.dev/builtin/agents/v1
auto_post_threshold: never
```

| Variable / flag       | Default                       | Required |
| --------------------- | ----------------------------- | -------- |
| `GITHUB_TOKEN`        | (none)                        | yes      |
| `GITHUB_OWNER`        | (none — user or organization) | yes      |
| `GITHUB_REPO`         | (none)                        | yes      |
| `GITHUB_API_URL`      | `https://api.github.com`      | no       |

## 3. What lands where

- **Dedup** — the pipeline calls `GET /repos/{owner}/{repo}/issues` with
  `state=open` and `labels=` set to a comma-separated list of the
  standard label set. Each open candidate's body is parsed for the HTML
  provenance comment, and a `cluster_id` match decides skip / comment /
  needs_create.
- **Auto-post** (`--auto-post-threshold ...`) — drafts whose severity
  meets the threshold are posted via `POST /issues` with the title,
  markdown body (including the provenance comment), and label set.
- **Review** (`--review`) — same flow as Jira and Linear: `$EDITOR` opens
  the draft, the operator confirms, and accepted drafts are posted.
- **State** — unlike the other trackers, `IssuePatch(state="closed")`
  works directly (`PATCH /issues/{number}` with `{"state": "closed"}`).
  v1.0 doesn't drive this from the pipeline, but it's available for
  scripts that consume the adapter directly.

## 4. Verification

- Unit tests cover the REST surface against `httpx.MockTransport`; no live
  GitHub needed for `pytest`.
- `tests/integration/test_github_e2e.py` is gated on `--run-integration`
  plus `GITHUB_TOKEN`, `GITHUB_OWNER`, `GITHUB_REPO`. The token's repo
  scope MUST include issue write permissions. The test creates an issue,
  asserts idempotent dedup, posts a comment on a grown cluster, then
  closes the test issue via `update_issue(state="closed")` (this works
  on GitHub even though Jira and Linear don't expose it).

## 5. Notes / Limitations

- The Issues endpoint returns both issues and pull requests in the same
  list. The adapter filters out PRs (anything with a `pull_request` key)
  before returning.
- GitHub's search index is eventually consistent; a freshly-created
  issue may not appear in `search_issues()` for up to a minute. The
  dedup loop uses `list_open_issues` (not search), which is strongly
  consistent, so this only matters for explicit `search_issues` calls.
- GitHub Enterprise Server: set `--github-api-url
  https://YOUR-GHES-HOST/api/v3` (or `GITHUB_API_URL` in config). The
  adapter is otherwise unchanged.
- Rate limits: authenticated requests get 5000/hour per token. For very
  large triage runs, partition with `--since` / `--until`.

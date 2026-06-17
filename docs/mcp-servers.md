# MCP servers reference

Every adapter ships as a standalone **stdio MCP server**, so any
MCP-aware client — another agent, an IDE assistant, a chat client — can
read traces and write issues through the same contracts the triage
pipeline uses. Six binaries are installed with the package:

| Binary | Wraps | Kind |
|---|---|---|
| `agent-triage-adapter-phoenix` | Phoenix | trace backend |
| `agent-triage-adapter-langfuse` | Langfuse | trace backend |
| `agent-triage-adapter-langsmith` | LangSmith | trace backend |
| `agent-triage-adapter-jira` | Jira | tracker |
| `agent-triage-adapter-linear` | Linear | tracker |
| `agent-triage-adapter-github` | GitHub Issues | tracker |

Configuration is environment-only (see the variable tables in
[cli.md](cli.md#mcp-adapter-binaries) /
[configuration.md](configuration.md)); a missing required variable exits
with code 2 naming the variable on stderr. Example MCP client
configuration:

```json
{
  "mcpServers": {
    "traces": {
      "command": "agent-triage-adapter-phoenix",
      "env": { "PHOENIX_URL": "http://localhost:6006" }
    },
    "issues": {
      "command": "agent-triage-adapter-github",
      "env": {
        "GITHUB_TOKEN": "ghp_...",
        "GITHUB_OWNER": "my-org",
        "GITHUB_REPO": "agents"
      }
    }
  }
}
```

All tools return JSON as text content. Errors surface as MCP tool errors
carrying the adapter's typed-error message (`BackendError` /
`TrackerError` text); the adapters retry 429/5xx with backoff before
giving up.

## Trace-backend tools (identical across the three backends)

### `list_traces`

List trace IDs in a time window. → `["trace-id", ...]`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `since` | string (ISO 8601) | yes | inclusive lower bound |
| `until` | string \| null | no | null = now |
| `filter` | object \| null | no | reserved key `status` (`"ok"`/`"error"`, root run/span state) is honored or the call fails; other keys are backend-specific |

### `list_traces_v2`

Same window/filter args as `list_traces`; returns per-trace summaries
plus the truncation contract:

```json
{
  "traces": [{"trace_id": "...", "start_time": "...", "status": "error",
              "latency_ms": 1234.5, "tags": {"tenant_id": "acme"}}],
  "truncated": false,
  "page_limit": 20
}
```

`truncated: true` means the backend stopped paginating at its
`*_MAX_LIST_PAGES` ceiling — treat the listing as a lower bound.

### `get_trace`

| Arg | Type | Required |
|---|---|---|
| `trace_id` | string | yes |

Returns the trace as **OTLP JSON** (OpenInference semantic conventions),
the same shape `agent_triage.models.from_otlp` parses.

### `annotate_trace`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `trace_id` | string | yes | |
| `annotation` | object | yes | the `Annotation` model: `run_id`, `rubric_version`, `mode_id`, `positive`, `severity`, optional `confidence`/`excerpt`/`notes` |

Upserts by the idempotency key
`(trace_id, run_id, rubric_version, mode_id)` — writing the same key
twice updates rather than duplicates.

### `search_traces`

| Arg | Type | Required | Default |
|---|---|---|---|
| `query` | string | yes | |
| `k` | integer | no | 10 |

Semantic search where the backend supports it; otherwise the call errors
(per the contract, a stub rather than fake results).

## Tracker tools (identical across the three trackers)

### `list_open_issues` / `list_open_issues_v2`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `filter` | object \| null | no | must honor a `labels` array meaning *all* labels present — this is how dedup queries `["agent-triage", "mode:<id>", "rubric:<name>@<version>"]` |

`_v2` adds `{"issues": [...], "truncated": bool, "page_limit": int}` —
when `truncated` is true, "no duplicate found" is unproven (the triage
poster reacts by queueing instead of auto-posting).

### `search_issues`

`query` (required), `k` (default 10). Free-text search; may be
unsupported by a tracker.

### `create_issue`

| Arg | Type | Required | Notes |
|---|---|---|---|
| `draft` | object | yes | the `IssueDraft` model; the tracker must persist `labels` and `body` faithfully (the body ends with the `<!-- agent-triage:provenance {...} -->` block) |

Returns the created `Issue` (`id`, `key`, `url`, `title`, `body`,
`labels`, `state`).

### `update_issue`

`issue_id` (string) + `patch` (object — `IssuePatch`: any of `title`,
`body`, `labels`, `state`).

### `comment_on_issue`

`issue_id` (string) + `comment` (string). Used by the pipeline to append
"cluster grew by these trace IDs" notes to existing issues.

## Notes for orchestration engineers

- The pipeline CLI does **not** spawn these binaries; it drives the same
  adapter classes in-process. The servers exist so *other* systems can
  compose with agent-triage's contracts — e.g. a chat agent that calls
  `list_traces_v2` + `get_trace` to investigate, then files follow-ups
  via `create_issue` with proper provenance so the nightly triage run
  dedups against them.
- If you create issues through these tools yourself, reuse
  `agent_triage.models.IssueDraft` / `make_labels()` so your issues
  participate in dedup instead of colliding with it.
- Tool names and schemas are stable per semver (they mirror the
  `TraceBackend`/`Tracker` ABCs documented in
  [python-api.md](python-api.md)).

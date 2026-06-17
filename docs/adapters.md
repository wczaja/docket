# Adapters

Adapters are how agent-triage stays platform-agnostic: every trace
backend and issue tracker is integrated behind the same two contracts,
and the runtime core never sees a backend-specific shape. This page
documents the contracts and how to write a new adapter.

## The two-layer pattern

Every integration is two layers (design §5.3):

1. **Adapter class** — pure-Python, async, no MCP dependency — under
   `agent_triage/adapters/trace/` or `agent_triage/adapters/tracker/`.
   Owns the backend-specific logic: HTTP calls (`httpx.AsyncClient`),
   normalization to OpenInference, pagination, retry-with-backoff,
   error mapping to `agent_triage.errors` types. Unit-testable
   in-process with a mocked transport.
2. **MCP server** — a thin entry point under
   `agent_triage/mcp_servers/` that instantiates the adapter class and
   exposes its methods as MCP tools (stdio), wired to a console script
   (`agent-triage-adapter-<name>`). This is the architectural seam: any
   MCP-aware client can reuse the adapter, and new backends can be added
   without touching the agent.

The `run`/`serve` CLI drives adapter classes in-process for the common
case; the MCP servers exist for config-driven composition
(`agent-triage.yaml`'s `type: mcp` blocks) and for external consumers.

## Trace backend contract

```python
class TraceBackend(ABC):
    async def list_traces(since, until, filter) -> list[trace_id]
    async def list_traces_v2(since, until, filter) -> TraceListing   # summaries + truncated flag
    async def get_trace(trace_id) -> OpenInferenceTrace
    async def annotate_trace(trace_id, annotation) -> None
    async def search_traces(query, k) -> list[trace_id]              # stub where unsupported
    async def close() -> None
```

Contract obligations, each pinned by the shared parity tests:

- **Normalize to OpenInference.** `get_trace` returns the canonical
  `OpenInferenceTrace`; whatever the backend calls its spans, attributes
  arrive under OpenInference names (`openinference.span.kind`,
  `llm.input_messages.*`, `tool.name`, …). Lossless enough that the
  classifier input is identical across backends for the same underlying
  data.
- **Loud truncation.** If listing stops at a page ceiling
  (`<NAME>_MAX_LIST_PAGES`), return `TraceListing(truncated=True)` and
  warn once — never pretend a partial listing is the window.
- **Filter pushdown.** The listing `filter`'s reserved `status` key
  (`{"status": "error"}` for `--strategy errors-only`) must be pushed to
  the backend query, or the adapter must raise — not filter client-side
  silently.
- **Idempotent annotations.** Writing the same
  `(trace_id, run_id, rubric_version, mode_id)` twice upserts.
- **Retry on 429/5xx** with backoff, honoring `Retry-After`.
- **Typed errors.** Network/API failures raise `BackendError`; missing
  credentials raise `CredentialError` at construction, naming the env
  var.

Shipped: **Phoenix** (GraphQL; OSS, the local-dev default), **Langfuse**
(public API; cloud + self-hosted), **LangSmith** (REST; API key
required). Setup guides: `docs/local-phoenix.md`, `local-langfuse.md`,
`local-langsmith.md`.

## Tracker contract

```python
class Tracker(ABC):
    async def list_open_issues(filter) -> list[Issue]
    async def list_open_issues_v2(filter) -> IssueListing   # truncated flag, same contract
    async def search_issues(query, k) -> list[Issue]
    async def create_issue(draft: IssueDraft) -> Issue
    async def update_issue(issue_id, patch) -> Issue
    async def comment_on_issue(issue_id, comment) -> None
    async def close() -> None
```

Obligations:

- **Provenance round-trip.** `create_issue` must persist the draft's
  labels (`agent-triage`, `mode:<id>`, `rubric:<name>@<version>`) and
  body (which ends with the HTML-comment provenance block) faithfully —
  dedup on the next run depends on reading both back.
- **Loud truncation, dedup safety.** Same `truncated` contract as
  backends. The poster reacts to a truncated open-issue listing by
  demoting auto-posts to `needs_create` (a duplicate can't be ruled
  out); the adapter's only job is honesty.
- **Rate-limit respect.** Jira Cloud (~10 req/s) and GitHub (5000/hr)
  limits are real; retry with backoff, honor `Retry-After`.
- Typed errors: `TrackerError` / `CredentialError`.

Shipped: **Jira** (Cloud + Data Center, auto-detected), **Linear**
(GraphQL), **GitHub Issues** (REST, Enterprise Server supported). Setup
guides: `docs/local-jira.md`, `local-linear.md`, `local-github.md`.

## Writing a new adapter

Use the Phoenix adapter (backends) or the GitHub adapter (trackers) as
the reference; they're the most commented. The checklist that gets a PR
merged:

1. **Open a "New adapter" issue first** (template provided) — confirm
   the API surface maps onto the contract and how CI will test it.
2. Implement the adapter class with `httpx.AsyncClient`, constructor
   taking explicit config + credentials (no env reads inside methods),
   `CredentialError` on missing credentials at construction.
3. Pagination loop with a configurable page ceiling
   (`<NAME>_MAX_LIST_PAGES`) and the `truncated` flag; retry helper with
   backoff for 429/5xx.
4. The MCP server wrapper in `agent_triage/mcp_servers/adapter_<name>.py`
   (copy an existing one — they're ~30 lines over the shared `_common`
   helpers) and a `[project.scripts]` entry.
5. Tests at parity with existing adapters: unit tests with mocked
   transports (success, auth failure, 429 retry, pagination,
   truncation), plus wiring into `test_adapter_parity.py` /
   `test_tracker_parity.py` so the cross-adapter invariants run against
   yours too.
6. A `docs/local-<name>.md` setup guide and, if the system is
   free/self-hostable, a docker-compose recipe for integration tests.

Proprietary observability backends (Datadog, New Relic, …) are welcome
as community-maintained adapters — the contract is public exactly so
they don't need to live in this repo to work.

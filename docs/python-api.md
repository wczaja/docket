# Python API reference

docket is a library as well as a CLI. This page documents the
public Python surface for embedding the pipeline in your own
orchestration — a scheduler, an Airflow DAG, a custom service, or
another agent.

**Stability policy.** Everything documented on this page follows semver:
breaking changes only in a major release. Modules prefixed with `_` and
anything not documented here are internal. All I/O is async; call from
your event loop or wrap with `asyncio.run`.

```python
import asyncio
from datetime import UTC, datetime, timedelta

from docket.adapters.trace.phoenix import PhoenixAdapter
from docket.agent.triage import run_triage_pipeline
from docket.rubric.loader import load_rubric

async def main() -> None:
    backend = PhoenixAdapter(base_url="http://localhost:6006")
    rubric = load_rubric("docket.dev/builtin/agents/v1")
    until = datetime.now(UTC)
    try:
        result = await run_triage_pipeline(
            backend=backend,
            rubric=rubric,
            since=until - timedelta(hours=1),
            until=until,
            backend_id="phoenix",
        )
    finally:
        await backend.close()
    print(result.report_markdown)
    for cluster in result.clusters:
        print(cluster.mode_id, cluster.stats.size, cluster.representative_trace_id)

asyncio.run(main())
```

Requires `ANTHROPIC_API_KEY` (default judge) and `OPENAI_API_KEY`
(embeddings) in the environment unless you pass explicit providers.

---

## The pipeline

### `docket.agent.triage.run_triage_pipeline(...) -> TriageResult`

The deterministic six-stage pipeline (`docket run` is a thin
wrapper around this). All parameters keyword-only:

| Parameter | Type / default | Meaning |
|---|---|---|
| `backend` | `TraceBackend` (required) | trace source; you own its lifecycle (`await backend.close()`) |
| `rubric` | `Rubric` (required) | from `load_rubric` |
| `since`, `until` | `datetime` (required) | the window; use tz-aware UTC |
| `llm_provider` | `ModelProvider \| None` | judge provider; default `anthropic:claude-haiku-4-5-20251001` |
| `embedding_provider` | `EmbeddingProvider \| None` | clustering embeddings; default `openai:text-embedding-3-small`, credentials preflighted before classification |
| `batch_size` | `int = 1` | traces per judge call |
| `concurrency` | `int = 8` | parallel trace classification |
| `write_annotations` | `bool = False` | backend writeback (read-only by default) |
| `run_id` | `str \| None` | default: `compute_run_id(...)` — deterministic, idempotent re-runs |
| `backend_id` | `str = "phoenix"` | label used in run_id derivation and logs |
| `output_dir` | `Path \| None` | draft/report queue dir; default `~/.docket/queued-issues/` |
| `tracker` | `Tracker \| None` | enables dedup + posting; `None` = queue locally |
| `auto_post_threshold` | `"never"` | `critical\|high\|medium\|low\|never` |
| `sample_count`, `sample_strategy`, `stratify_by` | `None`, `"uniform"`, `None` | run_id-seeded sampling (`uniform\|stratified\|errors-only`) |
| `checkpoint` | `bool = False` | sentinel-based resumability (needs write access) |
| `max_traces_per_run` | `int \| None = 1000` | budget gate; `None` disables |
| `max_estimated_cost_usd` | `float \| None = None` | dollar gate on the pre-flight estimate |

Raises `BudgetExceededError` before any trace fetch if a gate trips;
`ConfigError` / `CredentialError` for setup problems; `BackendError`
when annotation writeback exhausts its 5 retries. Per-trace fetch and
classification failures are absorbed and reported, not raised.

**`TriageResult`** (attributes): `run_report: RunReport`,
`clusters: list[Cluster]`, `drafts: list[IssueDraft]`,
`report_markdown: str`, `dedup_outcomes` (per-draft tracker decision:
posted / commented / skipped / needs_create), `review_outcomes`
(populated only by the CLI's `--review`).

### `compute_run_id(*, backend_id, rubric_version, since, until) -> str`

The deterministic run id: `sha256(f"{backend_id}|{rubric_version}|{since_iso}|{until_iso}")[:16]`,
where `rubric_version` is `f"{name}@{version}"`. Compute it yourself when
you need to correlate annotations with a run you're about to launch.

### Factories used by the CLI

`docket.runtime.build_backend(...)` / `build_tracker(...)` /
`resolve_backend_id(...)` construct adapters from CLI-flag/config
values — useful if you want CLI-equivalent precedence handling without
the CLI.

---

## Adapters

### Constructors

```python
from docket.adapters.trace.phoenix import PhoenixAdapter
from docket.adapters.trace.langfuse import LangfuseAdapter
from docket.adapters.trace.langsmith import LangsmithAdapter
from docket.adapters.tracker.jira import JiraAdapter
from docket.adapters.tracker.linear import LinearAdapter
from docket.adapters.tracker.github import GitHubAdapter

PhoenixAdapter(base_url, api_key=None, max_list_pages=...)
LangfuseAdapter(host, public_key=None, secret_key=None, max_list_pages=...)
LangsmithAdapter(endpoint, api_key, project=None, max_list_pages=...)
JiraAdapter(host, project, email=None, api_token=None, pat=None,
            deployment=None, max_list_pages=...)   # deployment: "cloud"|"datacenter"|None=auto
LinearAdapter(team_id, api_key, endpoint=..., max_list_pages=...)
GitHubAdapter(owner, repo, token, api_url=..., max_list_pages=...)
```

Missing credentials raise `CredentialError` at construction. Every
adapter retries 429/5xx with backoff and must be closed
(`await adapter.close()`).

### `docket.adapters.base.TraceBackend` (ABC)

```python
async def list_traces(since, until, filter=None) -> list[str]
async def list_traces_v2(since, until, filter=None) -> TraceListing
async def get_trace(trace_id) -> OpenInferenceTrace
async def annotate_trace(trace_id, annotation: Annotation) -> None      # upsert
async def search_traces(query, k=10) -> list[str]                       # may be unsupported
async def mark_trace_processed(...)                                     # checkpoint sentinel
async def list_processed_trace_ids(run_id, since, until) -> set[str]    # checkpoint read
async def close() -> None
```

`filter`'s reserved key `status` (`"ok"`/`"error"`) must be pushed down
to the backend query or the adapter raises. Subclass this to integrate a
backend we don't ship — implementation requirements and the parity-test
expectations are in [adapters.md](adapters.md).

### `docket.adapters.base.Tracker` (ABC)

```python
async def list_open_issues(filter=None) -> list[Issue]
async def list_open_issues_v2(filter=None) -> IssueListing
async def search_issues(query, k=10) -> list[Issue]
async def create_issue(draft: IssueDraft) -> Issue
async def update_issue(issue_id, patch: IssuePatch) -> Issue
async def comment_on_issue(issue_id, comment) -> None
async def close() -> None
```

---

## LLM providers

```python
from docket.llm import (
    build_provider, build_embedding_provider,
    ModelProvider, AnthropicProvider, OpenAIProvider,
    EmbeddingProvider, OpenAIEmbeddingProvider,
    DEFAULT_PROVIDER_URI,    # "anthropic:claude-haiku-4-5-20251001"
    DEFAULT_EMBEDDING_URI,   # "openai:text-embedding-3-small"
)

provider = build_provider("anthropic:claude-sonnet-4-6")   # "provider:model" URI
embedder = build_embedding_provider(DEFAULT_EMBEDDING_URI)
```

`ModelProvider` is one method:
`async structured_complete(system: str, user: str, schema: dict) -> dict` —
the schema (JSON Schema draft 2020-12) is enforced via the provider's
native mechanism (Anthropic forced tool use; OpenAI `response_format`),
raising `DetectionError` on enforcement failure. Implement this ABC to
plug in any other provider; constructors accept `api_key=None` (falls
back to the SDK's environment lookup) and an injectable `client` for
testing. `EmbeddingProvider.preflight()` validates credentials eagerly.

## Classifier (standalone)

Classify without clustering/drafting — e.g. to build your own scoring
loop:

```python
from docket.agent.subagents.classifier import Classifier

classifier = Classifier(provider, batch_size=1, concurrency=8)
results = await classifier.classify_all(
    [(trace_id, open_inference_trace), ...], rubric,
)   # -> dict[trace_id, list[Classification]]
```

Retries each (trace, mode) up to 3 times with backoff, then marks it
errored rather than raising.

---

## Rubrics

```python
from docket.rubric.loader import load_rubric        # resolves + merges imports
from docket.rubric.validator import validate_rubric_yaml
from docket.rubric.spec import Rubric, Mode, Detection
```

`load_rubric(source)` accepts a `Path`, path string, `file://` URI, or
builtin URI and returns the merged `Rubric` (raises `RubricImportError`
on cycles/missing imports, `RubricValidationError` otherwise). The
`Rubric`/`Mode`/`Detection` Pydantic models mirror the
[DSL reference](rubric-spec.md) exactly.

## Models (`docket.models`)

All Pydantic v2; `.model_dump()` / `.model_dump_json()` for
serialization.

| Model | Role / key fields |
|---|---|
| `OpenInferenceTrace`, `Span` | canonical trace; navigation helpers `get_llm_spans()`, `get_tool_call_spans()`, `get_retriever_spans()`, `get_final_response()`, projection `to_trace_like()` |
| `TraceLike`, `Verdict` | detector input/output views |
| `from_otlp` / `to_otlp` | lossless OTLP JSON ↔ `OpenInferenceTrace` |
| `Classification` | one detector × one trace: `trace_id, rubric_version, mode_id, positive, extra, duration_ms, error` (error set ⇒ positive False) |
| `Annotation` | backend-bound classification: adds `run_id, severity, confidence, excerpt`; `idempotency_key()` = `trace_id\|run_id\|rubric_version\|mode_id` |
| `Cluster`, `ClusterStats` | `cluster_id, mode_id, severity, member_trace_ids, representative_trace_id, representative_excerpt, stats(size, min/max/mean confidence)`; `compute_cluster_id()` is deterministic |
| `IssueDraft` | `cluster_id, mode_id, rubric_version, run_id, severity, representative_trace_id, member_trace_ids, title, body, labels` — body ends with the provenance comment |
| `IssueProvenance` | parse/emit the `<!-- docket:provenance {...} -->` block (`parse_from_body`) |
| `Issue`, `IssuePatch`, `IssueListing` | tracker-side shapes; `IssueListing.truncated` is the dedup-safety flag |
| `TraceListing`, `TraceSummary`, `RESERVED_FILTER_KEYS` | listing-with-summaries + `truncated` flag |
| `RunReport`, `ModeStats`, `TraceResult` | the structured run report behind `report_markdown`: window, counts, `traces_listed`, `listing_truncated`, per-mode positive/negative/error counts, `annotations_written` |

## Cost estimation (`docket.cost`)

`estimate_cost(trace_count, mode_count, model, ...) -> CostEstimate`,
`check_budget(...) -> BudgetCheck` (`.would_abort`, `.enforce()`),
`known_models()`. The same functions `--dry-run` uses.

## Errors (`docket.errors`)

Everything raised on purpose derives from `DocketError`:

| Exception | Raised when |
|---|---|
| `ConfigError` | config file / flag combination invalid |
| `CredentialError` | required key/token missing or invalid (at startup) |
| `RubricError` → `RubricValidationError`, `RubricImportError` | rubric schema / import problems |
| `DetectionError` | a detector or provider structured-output failure |
| `BackendError` | trace-backend call failed (after retries) |
| `TrackerError` | tracker call failed (after retries) |
| `BudgetExceededError` | trace cap or cost ceiling tripped pre-fetch |

Catch `DocketError` at your orchestration boundary; nothing else is
thrown deliberately.

## Observability (`docket.observability`)

- `redact(text: str) -> str` — the PII scrub (emails, phones, SSNs,
  account numbers) applied before logging and judge input. Call it on
  anything user-derived you log yourself.
- `configure_instrumentation(endpoint=...)` — context manager; while
  active, the pipeline emits its own OpenInference spans via OTLP
  (defaults target a local Phoenix at `http://localhost:6006`).

## Version

`docket.__version__` — single-sourced from package metadata.

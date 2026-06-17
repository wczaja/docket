# Using LangSmith with agent-triage

Phase 7 adds LangSmith as a third `TraceBackend`. Unlike Phoenix and Langfuse,
LangSmith is **SaaS-only** in v1.0 — there's no self-hosted Docker image to
boot. There's no `docker-compose` entry for LangSmith; tests use mocked HTTP
plus the cross-adapter parity check.

## 1. Get a LangSmith API key

Sign in at <https://smith.langchain.com> and create an API key under
Settings → API Keys. The key starts with `lsv2_` (or `ls__` for older keys).

## 2. Configure agent-triage

You can pass the key + endpoint via CLI flags:

```bash
agent-triage run \
  --backend langsmith \
  --langsmith-api-key "$LANGSMITH_API_KEY" \
  --langsmith-project my-agent-project \
  --rubric agent-triage.dev/builtin/agents/v1 \
  --since 1h
```

Or via `agent-triage.yaml`:

```yaml
trace_backend:
  type: mcp
  command: agent-triage-adapter-langsmith
  env:
    LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
    LANGSMITH_PROJECT: my-agent-project
    # Optional; defaults to https://api.smith.langchain.com:
    # LANGSMITH_ENDPOINT: https://eu.api.smith.langchain.com
```

| Variable / flag         | Default                                | Required |
| ----------------------- | -------------------------------------- | -------- |
| `LANGSMITH_API_KEY`     | (none)                                 | yes      |
| `LANGSMITH_ENDPOINT`    | `https://api.smith.langchain.com`      | no       |
| `LANGSMITH_PROJECT`     | (none — lists all projects' root runs) | no       |

## 3. What lands where

- **Read** — `agent-triage run --backend langsmith` queries
  `/api/v1/runs/query` for root runs in the time window, then fetches each
  one with its `child_runs` via `/api/v1/runs/{id}`. The adapter walks the
  run tree and produces an `OpenInferenceTrace` with spans keyed off
  `run_type` (`llm` → LLM, `tool` → TOOL, `retriever` → RETRIEVER,
  `embedding` → EMBEDDING, `chain`/`parser` → CHAIN, `agent` → AGENT).
- **Write** — `--annotate` posts feedback objects to `/api/v1/feedback` with
  `key=agent-triage:<mode_id>`, `value=positive|negative`, `score=confidence`,
  and `extra` carrying `run_id` + `rubric_version` + `idempotency_key`.
  Re-running with the same `(run_id, rubric_version, mode_id)` upserts on
  the LangSmith side via the idempotency key.

## 4. Verification

- Unit tests run against `httpx.MockTransport` — no live LangSmith needed for
  `pytest`.
- `tests/unit/test_adapter_parity.py` proves that the same canonical logical
  trace, encoded for all three adapters' API shapes (Phoenix GraphQL,
  Langfuse REST, LangSmith REST), decodes to **equal `TraceLike` views**
  through every adapter. This is the §7 Phase 7 acceptance bar.

## 5. No gated E2E test

Unlike Phoenix (which has a gated integration test against a local Docker
instance), Phase 7 deliberately doesn't ship a `tests/integration/
test_langsmith_e2e.py`. LangSmith is SaaS-only, so any such test would run
against the maintainer's personal LangSmith project — that's not a useful CI
gate. Unit tests + parity is enough for v1.0. If you want to validate against
a real LangSmith locally, run `agent-triage run --backend langsmith` against
a project you've populated with your own agent's instrumentation.

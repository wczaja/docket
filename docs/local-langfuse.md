# Running Langfuse locally for the Phase 6 acceptance loop

Phase 6 adds Langfuse as a second `TraceBackend`. The acceptance bar is the
same as Phoenix (recall = 1.0, precision >= 0.9 against the 60-trace
fixture), plus cross-adapter parity: identical logical trace data must
normalize to equal `TraceLike` views through either adapter.

## 1. Boot Langfuse (and Phoenix, if you want both)

```bash
docker compose up -d langfuse
```

This brings up:

- `langfuse-db` (Postgres 16, port-isolated to the compose network)
- `langfuse` (the v2 server, exposed at <http://localhost:3000>)

The compose definition pre-creates an `docket` project with the keys:

| Key           | Value          |
| ------------- | -------------- |
| Public key    | `pk-lf-dev`    |
| Secret key    | `sk-lf-dev`    |
| Host          | `http://localhost:3000` |

These are **development-only** values baked into `docker-compose.yml`. For
any non-local environment, rotate them.

Wait until `docker compose ps` reports `langfuse` as healthy, or open the UI
at <http://localhost:3000> to confirm.

## 2. Ingest traces into Langfuse

There isn't a Phase 6 acceptance ingestion script yet (the Phoenix ingester
script lives at `scripts/ingest_acceptance_traces.py` and is Phoenix-OTLP-
specific). For now, you can:

- Use Langfuse's own SDK / one of the project's example agents to populate
  traces; or
- Wait for Phase 6's planned ingester (`scripts/ingest_acceptance_traces_langfuse.py`)
  if/when added.

## 3. Run triage against Langfuse

```bash
docket run \
  --backend langfuse \
  --langfuse-host http://localhost:3000 \
  --langfuse-public-key pk-lf-dev \
  --langfuse-secret-key sk-lf-dev \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

Read-only by default. Add `--annotate` to write Langfuse scores back to the
trace; the score's `name` is `docket:<mode_id>` and its `metadata`
carries the run provenance (run_id + rubric_version + idempotency_key).

## 4. Stop Langfuse

```bash
docker compose down
```

Add `-v` to wipe the `langfuse-pgdata` volume if you want a clean slate.

## Notes

- The Phase 6 unit tests verify the adapter against `httpx.MockTransport`
  — no live Langfuse needed for `pytest`.
- The cross-adapter parity test (`tests/unit/test_adapter_parity.py`)
  proves equivalent trace data through `LangfuseAdapter` and
  `PhoenixAdapter` produces equal `TraceLike` views.
- The Phase 4 Phoenix integration test still works alongside Langfuse —
  both backends coexist in `docker-compose.yml`.

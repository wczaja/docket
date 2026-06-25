# Running Phoenix locally for the Phase 4 acceptance loop

## 1. Boot Phoenix

```bash
docker compose up -d phoenix
```

The Phoenix UI lands at http://localhost:6006. The OTLP HTTP ingestion endpoint
is at `/v1/traces` on the same port (we POST OTLP JSON to it).

Wait until `docker compose ps` reports `phoenix` as healthy, or check the UI.

## 2. Ingest the acceptance fixture

```bash
python scripts/ingest_acceptance_traces.py --phoenix-url http://localhost:6006
```

This posts 60 synthetic traces (20 clean + 40 with seeded failures, 8 each
across five of the `agents/v1` modes) to Phoenix's OTLP endpoint as protobuf
(`application/x-protobuf`, the only OTLP encoding current Phoenix builds
accept). The script prints a one-line manifest per trace plus a summary.

## 3. Run triage against Phoenix

```bash
docket run \
  --backend phoenix \
  --phoenix-url http://localhost:6006 \
  --rubric docket.dev/builtin/agents/v1 \
  --since 1h
```

Read-only by default. To write annotations back to Phoenix, add `--annotate`.

The output is a per-mode summary table plus a list of positive
classifications. The acceptance criterion is recall = 1.0 on the seeded
failures and precision ≥ 0.9 on the clean set.

## 4. Run the gated integration test

```bash
PHOENIX_URL=http://localhost:6006 \
  pytest --run-integration -m integration -v
```

The test ingests the same fixture, runs triage, and asserts the
recall/precision numbers. It requires:

- Phoenix running at `PHOENIX_URL`
- `ANTHROPIC_API_KEY` set (the hallucination + premature-termination modes
  call the LLM judge)

## 5. Stop Phoenix

```bash
docker compose down
```

Add `-v` to wipe the `phoenix-data` volume too if you want a clean slate.

# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-04

### Added

- **`docket demo`** — the full pipeline (classify → cluster → draft →
  report) over 60 bundled synthetic traces with an in-memory backend:
  no API keys, no Docker, no instrumented app. LLM-judge modes run
  under a clearly-labeled deterministic scripted judge (`--live` swaps
  in a real provider with one key); deterministic detectors run for
  real either way. `--rubric` points it at your own taxonomy;
  `--to-phoenix URL` ingests the demo traces into a real Phoenix via
  OTLP instead. The fixture is also available to `run`/`serve` as
  `--backend demo`. A scheduled `demo` workflow runs it in public CI
  with the report rendered in the job summary.
- **Single-API-key clustering paths**: `--clustering mode-only` (one
  cluster per firing mode; no embedding provider at all) and a
  `local:` embedding provider (fastembed/ONNX, no key) behind the new
  `local-embeddings` extra. `--embedding` and `--clustering` are
  available on both `run` and `serve` (serve previously had no
  embedding override), and the missing-OPENAI_API_KEY error now names
  all three ways out.
- **`docket init`** — interactive `docket.yaml` scaffolder (backend,
  tracker, rubric, auto-post threshold; secrets written as `${ENV_VAR}`
  references; refuses to overwrite without `--force`).
- **Rubric registry** (`rubrics/registry/`): six turnkey, self-testing
  taxonomies — support-agent, rag-knowledge-assistant,
  sql-analytics-agent, coding-agent, multi-agent-supervisor,
  voice-ivr-agent — each with a README (trace assumptions, tuning
  knobs, auto-post ratchet path). CI enforces the registry quality
  gate: every `llm_judge` mode ships a positive and a negative
  example, and the eval-rubrics workflow validates + self-tests
  registry rubrics alongside builtins.
- **Docs**: `docs/comparison.md` (dated, capability-level comparison
  with LangSmith Insights/Engine, Galileo/Cisco Signals, Latitude,
  Phoenix, Traceloop, Braintrust, and the eval-framework category);
  `docs/calibration/` (evidence tiers, the staged MAST/MAD live-run
  procedure with a publishing checklist, and a field guide for
  measuring per-mode false-positive rates on your own traffic); README
  reworked around an animated demo recording
  (`scripts/render_demo_svg.py` regenerates it from a real run) and a
  credential-ladder quickstart.

- **Builtin rubric `mast/v1`**: seven multi-agent coordination failure
  modes adapted from the MAST taxonomy (Cemri et al., "Why Do Multi-Agent
  LLM Systems Fail?", arXiv:2503.13657) — step repetition,
  conversation-history loss, termination unawareness, conversation reset,
  missing clarification, ignored agent input, and action-reasoning
  mismatch. Definitions are re-expressed in docket's own words and credited
  to the MAST authors; the subset is limited to modes detectable from a
  trace without ground-truth task outcomes. Compose it via `imports:`
  alongside the other builtins.
- **MAST judge tuning harness** (`scripts/tune_mast_judges.py`): scores the
  `mast/v1` `llm_judge` detectors against the MAD human-labelled dataset
  (Cemri et al., arXiv:2503.13657) and reports per-mode precision/recall/F1
  plus disagreements, for iterating on the judge prompts. Maintainer tool;
  ships no MAD data by default — the dataset (CC-BY-4.0) is fetched/provided
  by the user, with attribution. Documented in `docs/tuning-mast-judges.md`.

### Fixed

- The deterministic pipeline now actually writes `report.md` next to
  the queued drafts, as the quickstart and `agent/report.py` docstring
  always claimed (previously the report was only printed to stdout;
  only the deep-agent mode persisted it, in its virtual filesystem).

## [1.0.0] - 2026-06-12

First stable release. Everything in design `docs/design.md` Phases 0-10,
plus the budget-guardrail work originally scheduled for Phase 11.

### Added

- **Rubric DSL** (`apiVersion: docket.dev/v1`): YAML failure-mode
  taxonomy with JSON Schema validation, semver'd versioning, and
  composable imports (`file://`, packaged builtins). Reference:
  `docs/rubric-spec.md`.
- **Five detection types**: `llm_judge` (structured output enforced via
  provider-native mechanisms), `regex`, `tool_call`, `metric_threshold`,
  `composite`.
- **Four builtin rubrics**: `agents/v1`, `rag/v1`, `routing/v1`,
  `multi-agent/v1`, all synthetic, plus the published example
  `rubrics/examples/sample-support-agent.yaml`.
- **Three trace-backend adapters** — Phoenix, Langfuse, LangSmith — each
  normalizing to OpenInference, with cursor pagination, loud listing
  truncation, and retry-with-backoff. Each ships as an importable
  adapter class and a standalone MCP server binary.
- **Three tracker adapters** — Jira (Cloud + Data Center), Linear,
  GitHub Issues — with label + HTML-comment provenance dedup, also
  exposed as MCP servers.
- **Deterministic triage pipeline** (`docket run`): list →
  classify → annotate (opt-in) → cluster (embeddings + HDBSCAN) → draft
  → report, with deterministic `run_id` and idempotent re-runs.
- **Deep Agents harness mode** (`--agent`): the same six stages exposed
  as tools to a planning LLM, for exploratory runs.
- **Daemon mode** (`docket serve`): runs the pipeline on a fixed
  cadence with exactly-tiling windows; failed ticks retry their window
  instead of dropping it.
- **Budget guardrails**: `max_traces_per_run` hard cap,
  `max_estimated_cost_usd` dollar ceiling, `--dry-run` CI preflight
  gate, `--sample` with uniform / errors-only / stratified strategies,
  `--checkpoint` resumability via backend annotations.
- **Human-in-the-loop**: drafts queue locally by default; `--review`
  walks them through `$EDITOR`; `--auto-post-threshold` is explicit
  opt-in.
- **CLI**: `run`, `serve`, `validate`, `self-test`; six
  `docket-adapter-*` MCP server entry points.
- **Self-instrumentation**: the triage agent emits its own OpenInference
  traces via `--instrument-to`.
- **PII redaction** applied before logs and LLM-judge input.
- Documentation set: quickstart, concepts, rubric DSL reference, adapter
  guide, per-backend/tracker setup guides, benchmarks, security policy.

[1.1.0]: https://github.com/wczaja/docket/releases/tag/v1.1.0
[1.0.0]: https://github.com/wczaja/docket/releases/tag/v1.0.0

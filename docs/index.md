# agent-triage documentation

An observability-platform-agnostic triage runtime for LLM agent traces:
it reads traces from the backend you already run (Phoenix, Langfuse,
LangSmith), classifies them against a YAML failure-mode taxonomy you
control, clusters recurring failures, and drafts deduplicated issues
into your tracker (Jira, Linear, GitHub Issues) with a human in the loop
by default.

## Start here

| If you want to… | Read |
|---|---|
| Get a first run working in minutes | [Quickstart](quickstart.md) (the [README](../README.md) has the 5-minute Phoenix+GitHub path) |
| Learn the vocabulary | [Concepts](concepts.md) |
| Look up a command, flag, or exit code | [CLI reference](cli.md) |
| Configure via file / env vars, check precedence and defaults | [Configuration reference](configuration.md) |
| Embed the pipeline in your own orchestration (Python) | [Python API reference](python-api.md) |
| Drive the adapters from another agent or MCP client | [MCP servers reference](mcp-servers.md) |
| Write or extend a failure-mode taxonomy | [Rubric DSL reference](rubric-spec.md) + [`rubrics/examples/sample-support-agent.yaml`](../rubrics/examples/sample-support-agent.yaml) |
| Understand or add a backend/tracker integration | [Adapters](adapters.md) |
| See performance and cost numbers | [Benchmarks](benchmarks.md) |
| Understand every design decision | [Design document](design.md) |

## Setup guides

Per-backend: [Phoenix](local-phoenix.md) · [Langfuse](local-langfuse.md) ·
[LangSmith](local-langsmith.md).
Per-tracker: [Jira](local-jira.md) · [Linear](local-linear.md) ·
[GitHub Issues](local-github.md).
Testing against live services: [E2E testing](e2e-testing.md).

## Operating it

- `agent-triage run --since 1h` — one-shot pipeline; add `--dry-run` to
  price a window first, `--review` for editor-based draft review,
  `--sample N` for very large windows.
- `agent-triage serve --interval 1h` — daemon mode; tiles consecutive
  windows with no gaps and retries failed windows. Equivalent to cron +
  `run`, with window bookkeeping handled for you.
- `agent-triage validate <rubric>` / `agent-triage self-test <rubric>` —
  schema validation and example-based smoke tests for rubrics.

## Project

[Contributing](../CONTRIBUTING.md) ·
[Security policy](../SECURITY.md) ·
[Code of conduct](../CODE_OF_CONDUCT.md) ·
[Changelog](../CHANGELOG.md)

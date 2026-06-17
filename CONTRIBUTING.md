# Contributing to agent-triage

Thanks for considering a contribution! This page covers setup, the
quality gates CI enforces, the project rules that surprise newcomers,
and the two highest-leverage contribution paths (rubrics and adapters).

## Ground rules (read first)

These come from the project's design (`docs/design.md`) and are enforced
in review:

1. **Scope is bounded.** agent-triage sits *above* observability
   platforms. PRs that turn it into a trace store, an eval framework, a
   web UI, or anything with a local database (SQLite included) will be
   declined regardless of quality. State lives in the user's backend and
   tracker.
2. **OpenInference is the canonical schema.** Backend adapters normalize
   *to* it; backend-specific shapes never leak past the adapter.
3. **No new dependencies without discussion.** `pyproject.toml` is
   curated. Open an issue before adding anything.
4. **Typed errors only.** Raise the most specific subclass from
   `agent_triage.errors`; never bare `Exception`.
5. **Async I/O throughout.** `async def` + `httpx.AsyncClient` for all
   network and disk I/O. Prefer `httpx` over `requests`.
6. **Pydantic v2 for serialized data**; dataclasses only for internal
   value types.
7. **Redact before logging.** Call
   `agent_triage.observability.redact()` on any string that may contain
   user data before it reaches a log line or an LLM judge.
8. **Failure modes live in YAML rubrics**, never hardcoded in Python.
9. **Synthetic data only.** Every rubric, fixture, trace, prompt, and
   example you check in must be invented. Do not paste content derived
   from any real production system. When in
   doubt, leave it out and ask in the PR.

## Development setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/wczaja/agent-triage
cd agent-triage
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Quality gates

CI runs these on every PR across Python 3.11/3.12/3.13; run them locally
before pushing:

```bash
ruff check .                  # lint
ruff format --check .         # formatting
mypy --strict agent_triage    # types
pytest                        # unit tests; coverage gate is 90%
```

Integration tests against a live Phoenix (and tracker sandboxes) are
opt-in: `pytest --run-integration -m integration` — see
`docs/e2e-testing.md`. PRs touching rubric files also trigger the
`eval-rubrics` workflow, which validates every builtin and example
rubric.

## Contributing a rubric

The easiest first contribution. Builtin rubrics live in
`agent_triage/rubric/builtin/<name>/v1/rubric.yaml`; community examples
in `rubrics/examples/`.

1. Start from `rubrics/examples/sample-support-agent.yaml` and the DSL
   reference in `docs/rubric-spec.md`.
2. Keep it synthetic and domain-generic. Include `examples:` on
   `llm_judge` modes — they are the rubric's self-test.
3. `agent-triage validate path/to/rubric.yaml` must exit 0, and
   `agent-triage self-test path/to/rubric.yaml` should pass with a real
   judge model if you can run one.
4. Add the rubric to the tests (`tests/unit/test_builtin_rubrics.py` or
   `tests/unit/test_example_rubrics.py` picks up `rubrics/examples/*.yaml`
   automatically).

## Contributing an adapter

Trace backends and trackers follow a two-layer pattern (design §5.3):

1. A pure-Python async adapter class under
   `agent_triage/adapters/{trace,tracker}/` — owns HTTP calls,
   normalization to OpenInference, retries, error mapping. No MCP
   dependency; unit-testable in-process with a mocked `httpx` transport.
2. A thin MCP server entry point under `agent_triage/mcp_servers/` that
   exposes the class's methods as MCP tools, plus a console script in
   `pyproject.toml`.

Study `agent_triage/adapters/trace/phoenix.py` and its tests as the
reference implementation, and read `docs/adapters.md` for the full tool
contract. New adapters need: unit tests with mocked transports at parity
with the existing ones, retry-with-backoff on 429s, listing pagination
with the loud-truncation contract (`TraceListing.truncated`), and a
`docs/local-<name>.md` setup guide.

## Pull request flow

- Open an issue first for anything beyond a small fix, especially new
  dependencies, new adapters, or DSL changes.
- Keep PRs focused; separate refactors from behavior changes.
- All commits must pass the gates above; pre-commit enforces them.
- By contributing you agree your work is licensed under Apache-2.0 (the
  project license). No CLA.

## Reporting bugs / security issues

Bugs: use the issue templates. Security vulnerabilities: **do not open a
public issue** — see [SECURITY.md](SECURITY.md).

## What & why

<!-- One paragraph: the problem, and how this PR solves it. Link the issue. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy --strict agent_triage` passes
- [ ] `pytest` passes locally (coverage gate: 90%)
- [ ] New behavior is covered by tests
- [ ] Any rubric/fixture/doc content added is **synthetic** (no real
      trace data, internal system names, or proprietary taxonomies)
- [ ] No new dependencies — or the addition was discussed in an issue first
- [ ] Docs updated (`docs/`, README, or `--help` text) if behavior changed

## Scope guard

<!-- Delete if irrelevant. If this touches an explicit non-goal
(web UI, local DB, trace storage, eval framework), explain why it
doesn't cross the line — see docs/design.md §1.5. -->

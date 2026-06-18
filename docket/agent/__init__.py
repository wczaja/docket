"""Top-level triage agent (design §4).

The Deep Agent harness wires the four subagents in `docket.agent.subagents`
into a single workflow per design §4.1. Phase 5 ships the orchestration; the
subagents themselves are pure Python so they're individually unit-testable.
"""

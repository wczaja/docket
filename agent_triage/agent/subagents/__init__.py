"""Subagent implementations per design §4.3.

Each subagent is a pure-Python async class. The Deep Agent harness in
`agent_triage.agent.triage` wraps these as LangChain tools so the top-level
agent can delegate to them.
"""

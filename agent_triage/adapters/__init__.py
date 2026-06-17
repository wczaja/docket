"""Trace-backend and tracker adapters.

Each adapter is a pure-Python async class (no MCP dependency) that
implements one of the ABCs in `agent_triage.adapters.base`. A matching MCP
server in `agent_triage.mcp_servers/` wraps the adapter for use by the
triage agent (Phase 5). The split keeps adapter logic test-friendly while
preserving MCP as the architectural seam between the agent and the world.
"""

from agent_triage.adapters.base import TraceBackend

__all__ = ["TraceBackend"]

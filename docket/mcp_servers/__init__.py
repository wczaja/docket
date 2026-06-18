"""MCP server entry points wrapping pure-Python adapters.

The triage agent (Phase 5) talks to these MCP servers as standard MCP
clients (stdio). The servers themselves are thin: they own an adapter
instance and expose its methods as MCP tools. All real logic lives in
`docket.adapters`; the only thing here is the protocol seam.
"""

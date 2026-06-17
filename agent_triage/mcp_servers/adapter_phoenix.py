"""Stdio MCP server wrapping `PhoenixAdapter`.

Run directly (`python -m agent_triage.mcp_servers.adapter_phoenix`) or via the
installed entry point `agent-triage-adapter-phoenix`. Reads `PHOENIX_URL` (and
optionally `PHOENIX_API_KEY`) from the environment, instantiates a
`PhoenixAdapter`, and exposes its methods as MCP tools.

The shared dispatch layer lives in `agent_triage.mcp_servers._common`.
"""

import asyncio
import os
import sys

from agent_triage.adapters.trace.phoenix import PhoenixAdapter
from agent_triage.mcp_servers._common import TOOLS, build_server, dispatch_tool, serve

SERVER_NAME = "agent-triage-adapter-phoenix"

# Re-exports kept so existing tests + the parity check keep importing from here.
__all__ = ["SERVER_NAME", "TOOLS", "build_server", "cli_main", "dispatch_tool", "serve"]


def _backend_from_env() -> PhoenixAdapter:
    url = os.environ.get("PHOENIX_URL")
    if not url:
        sys.stderr.write(
            "agent-triage-adapter-phoenix: PHOENIX_URL environment variable is required.\n"
        )
        sys.exit(2)
    return PhoenixAdapter(base_url=url, api_key=os.environ.get("PHOENIX_API_KEY"))


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve` closes the backend in the same event loop before this returns.
    asyncio.run(serve(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

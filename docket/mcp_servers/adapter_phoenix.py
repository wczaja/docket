"""Stdio MCP server wrapping `PhoenixAdapter`.

Run directly (`python -m docket.mcp_servers.adapter_phoenix`) or via the
installed entry point `docket-adapter-phoenix`. Reads `PHOENIX_URL` (and
optionally `PHOENIX_API_KEY`) from the environment, instantiates a
`PhoenixAdapter`, and exposes its methods as MCP tools.

The shared dispatch layer lives in `docket.mcp_servers._common`.
"""

import asyncio
import os
import sys

from docket.adapters.trace.phoenix import PhoenixAdapter
from docket.mcp_servers._common import TOOLS, build_server, dispatch_tool, serve

SERVER_NAME = "docket-adapter-phoenix"

# Re-exports kept so existing tests + the parity check keep importing from here.
__all__ = ["SERVER_NAME", "TOOLS", "build_server", "cli_main", "dispatch_tool", "serve"]


def _backend_from_env() -> PhoenixAdapter:
    url = os.environ.get("PHOENIX_URL")
    if not url:
        sys.stderr.write("docket-adapter-phoenix: PHOENIX_URL environment variable is required.\n")
        sys.exit(2)
    return PhoenixAdapter(base_url=url, api_key=os.environ.get("PHOENIX_API_KEY"))


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve` closes the backend in the same event loop before this returns.
    asyncio.run(serve(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

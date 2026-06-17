"""Stdio MCP server wrapping `LangfuseAdapter`.

Run directly (`python -m agent_triage.mcp_servers.adapter_langfuse`) or via
the installed entry point `agent-triage-adapter-langfuse`. Reads
`LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` from the
environment, instantiates a `LangfuseAdapter`, and exposes its methods as
MCP tools.

The shared dispatch layer lives in `agent_triage.mcp_servers._common`.
"""

import asyncio
import os
import sys

from agent_triage.adapters.trace.langfuse import LangfuseAdapter
from agent_triage.mcp_servers._common import TOOLS, build_server, dispatch_tool, serve

SERVER_NAME = "agent-triage-adapter-langfuse"

__all__ = ["SERVER_NAME", "TOOLS", "build_server", "cli_main", "dispatch_tool", "serve"]


def _backend_from_env() -> LangfuseAdapter:
    host = os.environ.get("LANGFUSE_HOST")
    if not host:
        sys.stderr.write(
            "agent-triage-adapter-langfuse: LANGFUSE_HOST environment variable is required.\n"
        )
        sys.exit(2)
    return LangfuseAdapter(
        host=host,
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    )


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve` closes the backend in the same event loop before this returns.
    asyncio.run(serve(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

"""Stdio MCP server wrapping `LangsmithAdapter`.

Run directly (`python -m docket.mcp_servers.adapter_langsmith`) or via
the installed entry point `docket-adapter-langsmith`. Reads
`LANGSMITH_API_KEY` (required), `LANGSMITH_ENDPOINT` (optional, defaults to
LangSmith Cloud), and `LANGSMITH_PROJECT` (optional) from the environment,
instantiates a `LangsmithAdapter`, and exposes its methods as MCP tools.

The shared dispatch layer lives in `docket.mcp_servers._common`.
"""

import asyncio
import os
import sys

from docket.adapters.trace.langsmith import (
    DEFAULT_LANGSMITH_ENDPOINT,
    LangsmithAdapter,
)
from docket.mcp_servers._common import TOOLS, build_server, dispatch_tool, serve

SERVER_NAME = "docket-adapter-langsmith"

__all__ = ["SERVER_NAME", "TOOLS", "build_server", "cli_main", "dispatch_tool", "serve"]


def _backend_from_env() -> LangsmithAdapter:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        sys.stderr.write(
            "docket-adapter-langsmith: LANGSMITH_API_KEY environment variable is required.\n"
        )
        sys.exit(2)
    return LangsmithAdapter(
        endpoint=os.environ.get("LANGSMITH_ENDPOINT") or DEFAULT_LANGSMITH_ENDPOINT,
        api_key=api_key,
        project=os.environ.get("LANGSMITH_PROJECT"),
    )


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve` closes the backend in the same event loop before this returns.
    asyncio.run(serve(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

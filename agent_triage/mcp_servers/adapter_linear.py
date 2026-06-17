"""Stdio MCP server wrapping `LinearAdapter`.

Run directly (`python -m agent_triage.mcp_servers.adapter_linear`) or via the
installed entry point `agent-triage-adapter-linear`. Reads `LINEAR_API_KEY`
(required) and `LINEAR_TEAM_ID` (required) from the environment;
`LINEAR_ENDPOINT` is optional and defaults to Linear's hosted GraphQL
endpoint.

The shared dispatch layer lives in `agent_triage.mcp_servers._tracker_common`.
"""

import asyncio
import os
import sys

from agent_triage.adapters.tracker.linear import (
    DEFAULT_LINEAR_ENDPOINT,
    LinearAdapter,
)
from agent_triage.mcp_servers._tracker_common import (
    TRACKER_TOOLS,
    build_tracker_server,
    dispatch_tracker_tool,
    serve_tracker,
)

SERVER_NAME = "agent-triage-adapter-linear"

__all__ = [
    "SERVER_NAME",
    "TRACKER_TOOLS",
    "build_tracker_server",
    "cli_main",
    "dispatch_tracker_tool",
    "serve_tracker",
]


def _backend_from_env() -> LinearAdapter:
    api_key = os.environ.get("LINEAR_API_KEY")
    team_id = os.environ.get("LINEAR_TEAM_ID")
    if not api_key or not team_id:
        sys.stderr.write(
            "agent-triage-adapter-linear: LINEAR_API_KEY and LINEAR_TEAM_ID "
            "environment variables are required.\n"
        )
        sys.exit(2)
    return LinearAdapter(
        team_id=team_id,
        api_key=api_key,
        endpoint=os.environ.get("LINEAR_ENDPOINT") or DEFAULT_LINEAR_ENDPOINT,
    )


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve_tracker` closes the backend in the same event loop before this returns.
    asyncio.run(serve_tracker(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

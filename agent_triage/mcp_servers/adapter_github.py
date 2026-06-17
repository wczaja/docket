"""Stdio MCP server wrapping `GitHubAdapter`.

Run directly (`python -m agent_triage.mcp_servers.adapter_github`) or via
the installed entry point `agent-triage-adapter-github`. Reads `GITHUB_TOKEN`
(required), `GITHUB_OWNER` (required) and `GITHUB_REPO` (required) from the
environment; `GITHUB_API_URL` is optional and defaults to the public API.

The shared dispatch layer lives in `agent_triage.mcp_servers._tracker_common`.
"""

import asyncio
import os
import sys

from agent_triage.adapters.tracker.github import DEFAULT_GITHUB_API, GitHubAdapter
from agent_triage.mcp_servers._tracker_common import (
    TRACKER_TOOLS,
    build_tracker_server,
    dispatch_tracker_tool,
    serve_tracker,
)

SERVER_NAME = "agent-triage-adapter-github"

__all__ = [
    "SERVER_NAME",
    "TRACKER_TOOLS",
    "build_tracker_server",
    "cli_main",
    "dispatch_tracker_tool",
    "serve_tracker",
]


def _backend_from_env() -> GitHubAdapter:
    token = os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("GITHUB_OWNER")
    repo = os.environ.get("GITHUB_REPO")
    if not token or not owner or not repo:
        sys.stderr.write(
            "agent-triage-adapter-github: GITHUB_TOKEN, GITHUB_OWNER and "
            "GITHUB_REPO environment variables are required.\n"
        )
        sys.exit(2)
    return GitHubAdapter(
        owner=owner,
        repo=repo,
        token=token,
        api_url=os.environ.get("GITHUB_API_URL") or DEFAULT_GITHUB_API,
    )


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve_tracker` closes the backend in the same event loop before this returns.
    asyncio.run(serve_tracker(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

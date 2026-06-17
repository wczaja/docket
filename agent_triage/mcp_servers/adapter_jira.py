"""Stdio MCP server wrapping `JiraAdapter`.

Run directly (`python -m agent_triage.mcp_servers.adapter_jira`) or via the
installed entry point `agent-triage-adapter-jira`. Reads JIRA_HOST (required)
and JIRA_PROJECT (required) from the environment. For Cloud, also requires
JIRA_EMAIL + JIRA_API_TOKEN. For Data Center, requires JIRA_PAT instead.
JIRA_DEPLOYMENT (optional: `cloud` or `datacenter`) overrides hostname-based
auto-detection.

The shared dispatch layer lives in `agent_triage.mcp_servers._tracker_common`.
"""

import asyncio
import os
import sys

from agent_triage.adapters.tracker.jira import Deployment, JiraAdapter
from agent_triage.mcp_servers._tracker_common import (
    TRACKER_TOOLS,
    build_tracker_server,
    dispatch_tracker_tool,
    serve_tracker,
)

SERVER_NAME = "agent-triage-adapter-jira"

__all__ = [
    "SERVER_NAME",
    "TRACKER_TOOLS",
    "build_tracker_server",
    "cli_main",
    "dispatch_tracker_tool",
    "serve_tracker",
]


def _backend_from_env() -> JiraAdapter:
    host = os.environ.get("JIRA_HOST")
    project = os.environ.get("JIRA_PROJECT")
    if not host or not project:
        sys.stderr.write(
            "agent-triage-adapter-jira: JIRA_HOST and JIRA_PROJECT environment "
            "variables are required.\n"
        )
        sys.exit(2)
    deployment_env = os.environ.get("JIRA_DEPLOYMENT")
    deployment: Deployment | None = None
    if deployment_env in ("cloud", "datacenter"):
        deployment = deployment_env  # type: ignore[assignment]
    return JiraAdapter(
        host=host,
        project=project,
        email=os.environ.get("JIRA_EMAIL"),
        api_token=os.environ.get("JIRA_API_TOKEN"),
        pat=os.environ.get("JIRA_PAT"),
        deployment=deployment,
    )


def cli_main() -> None:
    backend = _backend_from_env()
    # `serve_tracker` closes the backend in the same event loop before this returns.
    asyncio.run(serve_tracker(backend, SERVER_NAME))


if __name__ == "__main__":
    cli_main()

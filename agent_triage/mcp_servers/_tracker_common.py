"""Backend-agnostic MCP server scaffolding for tracker adapters.

Mirrors `_common.py` (trace backends) — `dispatch_tracker_tool` translates
incoming MCP tool calls into the right `Tracker` method, `build_tracker_server`
wires it as a stdio MCP server with the same five tools every tracker adapter
exposes. Per-adapter entry points (e.g. `adapter_jira.py`) only provide a
`SERVER_NAME` and a backend constructor.
"""

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from agent_triage.adapters.base import Tracker
from agent_triage.errors import TrackerError
from agent_triage.models.issue import IssueDraft, IssuePatch

TRACKER_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_open_issues",
        description="List open issues, optionally filtered (e.g. by `labels`).",
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": ["object", "null"],
                    "description": (
                        "Backend-specific filter; MUST honor a `labels` array "
                        "meaning all labels present."
                    ),
                }
            },
        },
    ),
    Tool(
        name="search_issues",
        description="Free-text search over issues (may raise NotImplementedError).",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
        },
    ),
    Tool(
        name="create_issue",
        description="Create a new issue from an IssueDraft payload.",
        inputSchema={
            "type": "object",
            "required": ["draft"],
            "properties": {"draft": {"type": "object"}},
        },
    ),
    Tool(
        name="update_issue",
        description="Apply a partial update (IssuePatch) to an existing issue.",
        inputSchema={
            "type": "object",
            "required": ["issue_id", "patch"],
            "properties": {
                "issue_id": {"type": "string"},
                "patch": {"type": "object"},
            },
        },
    ),
    Tool(
        name="comment_on_issue",
        description="Post a comment on an existing issue.",
        inputSchema={
            "type": "object",
            "required": ["issue_id", "comment"],
            "properties": {
                "issue_id": {"type": "string"},
                "comment": {"type": "string"},
            },
        },
    ),
)


def _require(arguments: dict[str, Any], key: str) -> Any:
    try:
        return arguments[key]
    except KeyError as exc:
        raise TrackerError(f"invalid tool argument: missing required argument {key!r}") from exc


def _parse_k(arguments: dict[str, Any]) -> int:
    try:
        return int(arguments.get("k", 10))
    except (TypeError, ValueError) as exc:
        raise TrackerError(
            f"invalid tool argument: 'k' must be an integer, got {arguments.get('k')!r}"
        ) from exc


async def dispatch_tracker_tool(
    tracker: Tracker,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Translate one MCP tool call into the right Tracker method.

    Returns a JSON string (so the MCP TextContent layer can carry it as-is).
    Raises `TrackerError` on adapter failure, malformed arguments, or an
    unknown tool name.
    """
    if name == "list_open_issues":
        issues = await tracker.list_open_issues(arguments.get("filter"))
        return json.dumps([i.model_dump() for i in issues])
    if name == "search_issues":
        results = await tracker.search_issues(
            _require(arguments, "query"),
            _parse_k(arguments),
        )
        return json.dumps([i.model_dump() for i in results])
    if name == "create_issue":
        try:
            draft = IssueDraft.model_validate(_require(arguments, "draft"))
        except ValidationError as exc:
            raise TrackerError(
                f"invalid tool argument: 'draft' is not a valid IssueDraft: {exc}"
            ) from exc
        issue = await tracker.create_issue(draft)
        return json.dumps(issue.model_dump())
    if name == "update_issue":
        try:
            patch = IssuePatch.model_validate(_require(arguments, "patch"))
        except ValidationError as exc:
            raise TrackerError(
                f"invalid tool argument: 'patch' is not a valid IssuePatch: {exc}"
            ) from exc
        issue = await tracker.update_issue(_require(arguments, "issue_id"), patch)
        return json.dumps(issue.model_dump())
    if name == "comment_on_issue":
        await tracker.comment_on_issue(
            _require(arguments, "issue_id"),
            _require(arguments, "comment"),
        )
        return json.dumps({"ok": True})
    raise TrackerError(f"Unknown MCP tool: {name!r}")


def build_tracker_server(tracker: Tracker, server_name: str) -> Server:
    server: Server = Server(server_name)

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[Tool]:
        return list(TRACKER_TOOLS)

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = await dispatch_tracker_tool(tracker, name, arguments)
        return [TextContent(type="text", text=result)]

    return server


async def serve_tracker(tracker: Tracker, server_name: str) -> None:
    """Run the stdio MCP server until the transport closes.

    Closes `tracker` in the same event loop the server ran in, so adapters'
    lazily-created async HTTP clients are released on a live loop.
    """
    server = build_tracker_server(tracker, server_name)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await tracker.close()

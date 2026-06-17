"""Backend-agnostic MCP server scaffolding.

`dispatch_tool` translates incoming MCP tool calls into the right
`TraceBackend` method; `build_server` wires it as a stdio MCP server with
the same six tools every adapter exposes. The per-adapter entry points
(`mcp_servers/adapter_phoenix.py`, `adapter_langfuse.py`) only provide a
`SERVER_NAME` and a backend constructor.
"""

import json
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from agent_triage.adapters.base import TraceBackend
from agent_triage.errors import BackendError
from agent_triage.models.classification import Annotation
from agent_triage.models.otlp import to_otlp

TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_traces",
        description="List trace IDs in a time window.",
        inputSchema={
            "type": "object",
            "required": ["since"],
            "properties": {
                "since": {
                    "type": "string",
                    "description": "ISO 8601 lower bound (inclusive).",
                },
                "until": {
                    "type": ["string", "null"],
                    "description": "ISO 8601 upper bound; null = now.",
                },
                "filter": {
                    "type": ["object", "null"],
                    "description": "Backend-specific filter; may be ignored.",
                },
            },
        },
    ),
    Tool(
        name="get_trace",
        description="Fetch a single trace by ID as OTLP JSON.",
        inputSchema={
            "type": "object",
            "required": ["trace_id"],
            "properties": {"trace_id": {"type": "string"}},
        },
    ),
    Tool(
        name="annotate_trace",
        description="Write an annotation to the backend (upsert by idempotency_key).",
        inputSchema={
            "type": "object",
            "required": ["trace_id", "annotation"],
            "properties": {
                "trace_id": {"type": "string"},
                "annotation": {"type": "object"},
            },
        },
    ),
    Tool(
        name="search_traces",
        description="Semantic search (where supported; may raise NotImplementedError).",
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
        name="mark_trace_processed",
        description=(
            "Write the checkpoint sentinel marking a trace as fully classified "
            "under (run_id, rubric_version)."
        ),
        inputSchema={
            "type": "object",
            "required": ["trace_id", "run_id", "rubric_version"],
            "properties": {
                "trace_id": {"type": "string"},
                "run_id": {"type": "string"},
                "rubric_version": {"type": "string"},
            },
        },
    ),
    Tool(
        name="list_processed_trace_ids",
        description=(
            "List trace IDs already checkpointed for run_id in a time window "
            "(returned as a sorted list)."
        ),
        inputSchema={
            "type": "object",
            "required": ["run_id", "since"],
            "properties": {
                "run_id": {"type": "string"},
                "since": {
                    "type": "string",
                    "description": "ISO 8601 lower bound (inclusive).",
                },
                "until": {
                    "type": ["string", "null"],
                    "description": "ISO 8601 upper bound; null = now.",
                },
            },
        },
    ),
)


def _require(arguments: dict[str, Any], key: str) -> Any:
    try:
        return arguments[key]
    except KeyError as exc:
        raise BackendError(f"invalid tool argument: missing required argument {key!r}") from exc


def _parse_datetime(value: Any, key: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackendError(
            f"invalid tool argument: {key!r} must be an ISO 8601 datetime, got {value!r}"
        ) from exc


def _parse_k(arguments: dict[str, Any]) -> int:
    try:
        return int(arguments.get("k", 10))
    except (TypeError, ValueError) as exc:
        raise BackendError(
            f"invalid tool argument: 'k' must be an integer, got {arguments.get('k')!r}"
        ) from exc


async def dispatch_tool(
    backend: TraceBackend,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Translate one MCP tool call into the right adapter method.

    Returns a JSON string (so the MCP TextContent layer can carry it as-is).
    Raises `BackendError` on adapter failure, malformed arguments, or an
    unknown tool name.
    """
    if name == "list_traces":
        since = _parse_datetime(_require(arguments, "since"), "since")
        until_str = arguments.get("until")
        until = _parse_datetime(until_str, "until") if until_str else None
        ids = await backend.list_traces(since, until, arguments.get("filter"))
        return json.dumps(ids)
    if name == "get_trace":
        trace = await backend.get_trace(_require(arguments, "trace_id"))
        return json.dumps(to_otlp(trace))
    if name == "annotate_trace":
        try:
            annotation = Annotation.model_validate(_require(arguments, "annotation"))
        except ValidationError as exc:
            raise BackendError(
                f"invalid tool argument: 'annotation' is not a valid Annotation: {exc}"
            ) from exc
        await backend.annotate_trace(_require(arguments, "trace_id"), annotation)
        return json.dumps({"ok": True})
    if name == "search_traces":
        results = await backend.search_traces(_require(arguments, "query"), _parse_k(arguments))
        return json.dumps(results)
    if name == "mark_trace_processed":
        await backend.mark_trace_processed(
            _require(arguments, "trace_id"),
            run_id=_require(arguments, "run_id"),
            rubric_version=_require(arguments, "rubric_version"),
        )
        return json.dumps({"ok": True})
    if name == "list_processed_trace_ids":
        since = _parse_datetime(_require(arguments, "since"), "since")
        until_str = arguments.get("until")
        until = _parse_datetime(until_str, "until") if until_str else None
        processed = await backend.list_processed_trace_ids(
            run_id=_require(arguments, "run_id"),
            since=since,
            until=until,
        )
        return json.dumps(sorted(processed))
    raise BackendError(f"Unknown MCP tool: {name!r}")


def build_server(backend: TraceBackend, server_name: str) -> Server:
    server: Server = Server(server_name)

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[Tool]:
        return list(TOOLS)

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = await dispatch_tool(backend, name, arguments)
        return [TextContent(type="text", text=result)]

    return server


async def serve(backend: TraceBackend, server_name: str) -> None:
    """Run the stdio MCP server until the transport closes.

    Closes `backend` in the same event loop the server ran in, so adapters'
    lazily-created async HTTP clients are released on a live loop.
    """
    server = build_server(backend, server_name)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await backend.close()

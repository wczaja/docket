#!/usr/bin/env python3
"""Populate a LangSmith project with the Phase 4 acceptance fixture.

Usage::

    LANGSMITH_API_KEY=ls-... \\
        python scripts/ingest_acceptance_traces_langsmith.py \\
        --project docket-e2e

Posts the 60-trace fixture (20 clean + 40 seeded failures) from
`docket._acceptance` to LangSmith's batch runs endpoint at
`/api/v1/runs/batch`. Prints a one-line manifest per trace plus a summary.

LangSmith's run shape and OpenInference's span shape don't line up 1:1;
this script does a *minimal* mapping that's good enough for the
classifier to do its job — names, inputs, outputs, parent / trace
linkage, run_type. It does NOT round-trip every attribute (no usage
counts, no events). For high-fidelity ingestion, use a real
OpenInference instrumentation in your agent code.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from docket._acceptance import acceptance_summary, build_acceptance_cases
from docket.models.trace import OpenInferenceTrace, Span

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

_KIND_TO_RUN_TYPE: dict[str, str] = {
    "LLM": "llm",
    "TOOL": "tool",
    "CHAIN": "chain",
    "AGENT": "chain",
    "RETRIEVER": "retriever",
    "EMBEDDING": "embedding",
}

# Stable namespace so re-runs map the fixture's string IDs to the same UUIDs
# (LangSmith requires UUIDs for id / trace_id / parent_run_id).
_LS_NAMESPACE = uuid.UUID("8e8c0c3f-3a3b-4e0c-9a0b-1a2b3c4d5e6f")


def _to_uuid(label: str) -> str:
    return str(uuid.uuid5(_LS_NAMESPACE, label))


def _unix_nano_to_iso(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC).isoformat()


def _dotted_segment(start_iso: str, run_id_uuid: str) -> str:
    dt = datetime.fromisoformat(start_iso)
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    compact = dt.strftime("%Y%m%dT%H%M%S") + f"{dt.microsecond:06d}Z"
    return f"{compact}{run_id_uuid}"


def _span_to_run(
    span: Span,
    *,
    project: str,
) -> dict[str, Any]:
    kind = str(span.attributes.get("openinference.span.kind") or "CHAIN").upper()
    run_type = _KIND_TO_RUN_TYPE.get(kind, "chain")
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    if kind == "LLM":
        messages = []
        i = 0
        while True:
            role_key = f"llm.input_messages.{i}.message.role"
            content_key = f"llm.input_messages.{i}.message.content"
            if role_key not in span.attributes:
                break
            messages.append(
                {
                    "role": span.attributes.get(role_key),
                    "content": span.attributes.get(content_key, ""),
                }
            )
            i += 1
        if messages:
            inputs["messages"] = messages
        out_content = span.attributes.get("llm.output_messages.0.message.content")
        if out_content is not None:
            outputs["content"] = out_content
    elif kind == "TOOL":
        if "tool.parameters" in span.attributes:
            inputs["arguments"] = span.attributes["tool.parameters"]
        if "output.value" in span.attributes:
            outputs["result"] = span.attributes["output.value"]
    return {
        "name": span.name,
        "run_type": run_type,
        "start_time": _unix_nano_to_iso(span.start_time_unix_nano),
        "end_time": _unix_nano_to_iso(span.end_time_unix_nano),
        "inputs": inputs,
        "outputs": outputs,
        "session_name": project,
        "extra": {
            "metadata": {
                "openinference.kind": kind,
                **{
                    k: v
                    for k, v in span.attributes.items()
                    if k.startswith(("llm.model_name", "tool.name"))
                },
            }
        },
    }


def _trace_to_runs(trace: OpenInferenceTrace, *, project: str) -> list[dict[str, Any]]:
    """Translate an OpenInferenceTrace into LangSmith batch-API runs.

    LangSmith requires UUIDs for every id, a single-root tree per trace (so
    trace_id == root_run.id), and a `dotted_order` per run. Spans whose
    parent_span_id is None *and* aren't the chosen canonical root get
    reparented under the canonical root so multi-root fixtures still form a
    valid tree.
    """
    if not trace.spans:
        return []

    canonical_root = next((s for s in trace.spans if s.parent_span_id is None), trace.spans[0])
    id_map: dict[str, str] = {s.span_id: _to_uuid(s.span_id) for s in trace.spans}
    root_uuid = id_map[canonical_root.span_id]

    by_id: dict[str, Span] = {s.span_id: s for s in trace.spans}
    effective_parent: dict[str, str | None] = {}
    for s in trace.spans:
        if s.span_id == canonical_root.span_id:
            effective_parent[s.span_id] = None
        elif s.parent_span_id and s.parent_span_id in by_id:
            effective_parent[s.span_id] = s.parent_span_id
        else:
            effective_parent[s.span_id] = canonical_root.span_id

    # Walk parents-first so each span's dotted_order can prepend its parent's.
    dotted_orders: dict[str, str] = {}

    def _build_dotted(span_id: str) -> str:
        if span_id in dotted_orders:
            return dotted_orders[span_id]
        span = by_id[span_id]
        seg = _dotted_segment(_unix_nano_to_iso(span.start_time_unix_nano), id_map[span_id])
        parent_id = effective_parent[span_id]
        dotted = seg if parent_id is None else f"{_build_dotted(parent_id)}.{seg}"
        dotted_orders[span_id] = dotted
        return dotted

    runs: list[dict[str, Any]] = []
    for s in trace.spans:
        run = _span_to_run(s, project=project)
        parent_id = effective_parent[s.span_id]
        run["id"] = id_map[s.span_id]
        run["trace_id"] = root_uuid
        run["parent_run_id"] = id_map[parent_id] if parent_id else None
        run["dotted_order"] = _build_dotted(s.span_id)
        runs.append(run)
    return runs


async def ingest_all(
    *,
    endpoint: str,
    api_key: str,
    project: str,
) -> int:
    cases = build_acceptance_cases()
    failures = 0
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=endpoint, headers=headers, timeout=30.0) as client:
        for label, modes, trace in cases:
            runs = _trace_to_runs(trace, project=project)
            response = await client.post("/api/v1/runs/batch", json={"post": runs})
            modes_label = ",".join(modes) if modes else "clean"
            if response.status_code >= 400:
                sys.stderr.write(
                    f"FAIL  {label:14}  expected={modes_label:24}  "
                    f"trace_id={trace.trace_id}  "
                    f"status={response.status_code}: {response.text[:200]}\n"
                )
                failures += 1
            else:
                sys.stdout.write(
                    f"OK    {label:14}  expected={modes_label:24}  trace_id={trace.trace_id}\n"
                )
    return len(cases) - failures


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_LANGSMITH_ENDPOINT,
        help="LangSmith API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LangSmith API key (default: $LANGSMITH_API_KEY).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="LangSmith project name (session) to ingest traces into.",
    )
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        sys.stderr.write("ERROR: LANGSMITH_API_KEY (or --api-key) is required.\n")
        return 2
    sys.stderr.write("acceptance summary: " + json.dumps(acceptance_summary()) + "\n\n")
    n = await ingest_all(endpoint=args.endpoint, api_key=api_key, project=args.project)
    sys.stderr.write(f"\nIngested {n} traces.\n")
    return 0 if n == len(build_acceptance_cases()) else 1


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()

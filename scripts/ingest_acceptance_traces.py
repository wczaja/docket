#!/usr/bin/env python3
"""Populate a running Phoenix with the acceptance fixture.

Usage::

    python scripts/ingest_acceptance_traces.py --phoenix-url http://localhost:6006

Posts the 60-trace fixture (20 clean + 40 seeded failures) from
`docket._acceptance` to Phoenix's OTLP HTTP endpoint at /v1/traces as
protobuf (application/x-protobuf) — the only OTLP encoding current Phoenix
builds accept. Prints a one-line manifest per trace plus a summary.
"""

import argparse
import asyncio
import json
import sys

import httpx

from docket._acceptance import acceptance_summary, build_acceptance_cases
from docket.models.otlp import to_otlp_protobuf


async def ingest_all(phoenix_url: str) -> int:
    """Ingest the acceptance traces. Returns the number of traces ingested."""
    cases = build_acceptance_cases()
    failures = 0
    async with httpx.AsyncClient(base_url=phoenix_url, timeout=30.0) as client:
        for label, modes, trace in cases:
            response = await client.post(
                "/v1/traces",
                content=to_otlp_protobuf(trace),
                headers={"content-type": "application/x-protobuf"},
            )
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
        "--phoenix-url",
        default="http://localhost:6006",
        help="Phoenix base URL (default: %(default)s)",
    )
    args = parser.parse_args()
    sys.stderr.write("acceptance summary: " + json.dumps(acceptance_summary()) + "\n\n")
    n = await ingest_all(args.phoenix_url)
    sys.stderr.write(f"\nIngested {n} traces.\n")
    return 0 if n == len(build_acceptance_cases()) else 1


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()

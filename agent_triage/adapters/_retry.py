"""Shared HTTP retry/backoff helper for all six adapters.

Phase 11 calls for retry-with-backoff parity across Phoenix, Langfuse,
LangSmith, Jira, Linear, and GitHub. This module is the single
implementation (extracted from the LangSmith adapter's `_request`, which
shipped first) so the six adapters cannot drift.

Semantics:

- HTTP 429 is always retried: the server rejected the request, so nothing
  executed — safe for reads *and* writes (including issue creates).
- 502/503/504 and transport errors (connect/read timeouts) are retried
  only when the caller declares the request idempotent. A create that
  times out after reaching the server may have executed; blind retry
  would double-post.
- `Retry-After` is honored when present — delta-seconds or HTTP-date
  (parsed as GMT per RFC 9110) — and is **capped** at `max_delay` so a
  hostile or buggy header can't stall a run for a day.
- Otherwise: exponential backoff with full jitter, capped at `max_delay`.
- After `max_attempts`, the last response is returned for the caller's
  normal status handling; an exhausted transport-error retry raises the
  caller-supplied typed error instead of leaking raw httpx exceptions.
"""

import asyncio
import random
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from agent_triage.errors import AgentTriageError

DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0

_RETRYABLE_SERVER_STATUSES = frozenset({502, 503, 504})

__all__ = [
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_DELAY",
    "parse_retry_after",
    "request_with_retry",
]


def parse_retry_after(value: str | None, *, max_delay: float = DEFAULT_MAX_DELAY) -> float | None:
    """Parse a Retry-After header into a capped, non-negative delay in seconds.

    Accepts delta-seconds or an HTTP-date. Naive HTTP-dates are GMT per
    RFC 9110 §10.2.3. Unparseable values return None (caller falls back
    to jittered backoff).
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        seconds = (dt - datetime.now(UTC)).total_seconds()
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf guard
        return None
    return min(max(seconds, 0.0), max_delay)


async def request_with_retry(  # noqa: PLR0913 -- retry knobs form one logical unit
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    error_cls: type[AgentTriageError],
    idempotent: bool = False,
    json_body: Any = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    sleep: Callable[[float], Any] | None = None,
) -> httpx.Response:
    """Send a request, retrying per the module-docstring semantics.

    `error_cls` is the adapter's typed error (`BackendError` for trace
    backends, `TrackerError` for trackers) used to wrap transport errors.
    `sleep` is injectable for tests; defaults to `asyncio.sleep`.
    """
    do_sleep = sleep if sleep is not None else asyncio.sleep
    response: httpx.Response | None = None
    last_transport_error: httpx.HTTPError | None = None
    for attempt in range(max_attempts):
        try:
            response = await client.request(
                method, url, json=json_body, params=params, headers=headers
            )
            last_transport_error = None
        except httpx.HTTPError as e:
            if not idempotent:
                raise error_cls(f"{method} {url} failed: {e}") from e
            last_transport_error = e
            response = None
        if response is not None:
            retryable = response.status_code == 429 or (
                idempotent and response.status_code in _RETRYABLE_SERVER_STATUSES
            )
            if not retryable:
                return response
        if attempt == max_attempts - 1:
            break
        delay: float | None = None
        if response is not None:
            delay = parse_retry_after(response.headers.get("Retry-After"), max_delay=max_delay)
        if delay is None:
            delay = min(
                random.uniform(0, base_delay * (2**attempt)),  # noqa: S311 — retry jitter, not crypto
                max_delay,
            )
        await do_sleep(delay)
    if response is None:
        raise error_cls(
            f"{method} {url} failed after {max_attempts} attempts: {last_transport_error}"
        ) from last_transport_error
    return response

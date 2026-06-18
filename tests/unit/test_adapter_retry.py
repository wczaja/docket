"""Tests for the shared adapter retry helper."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from docket.adapters._retry import parse_retry_after, request_with_retry
from docket.errors import BackendError, TrackerError


def _client(handler) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t")


async def _no_sleep(_delay: float) -> None:
    return None


# --- parse_retry_after ------------------------------------------------------


def test_parse_retry_after_seconds() -> None:
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after("0") == 0.0


def test_parse_retry_after_is_capped() -> None:
    assert parse_retry_after("86400", max_delay=60.0) == 60.0


def test_parse_retry_after_rejects_inf_and_nan() -> None:
    assert parse_retry_after("inf") is None
    assert parse_retry_after("nan") is None


def test_parse_retry_after_negative_clamps_to_zero() -> None:
    assert parse_retry_after("-5") == 0.0


def test_parse_retry_after_http_date_is_gmt() -> None:
    # A naive HTTP-date must be treated as GMT (RFC 9110), not local time.
    future = datetime.now(UTC) + timedelta(seconds=30)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    delay = parse_retry_after(header, max_delay=60.0)
    assert delay is not None
    assert 20.0 <= delay <= 60.0


def test_parse_retry_after_garbage_and_none() -> None:
    assert parse_retry_after("soonish") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("  ") is None


# --- request_with_retry -----------------------------------------------------


async def test_429_is_retried_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await request_with_retry(
            client, "GET", "/x", error_cls=BackendError, sleep=_no_sleep
        )
    assert response.status_code == 200
    assert calls == 3


async def test_429_exhaustion_returns_last_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with _client(handler) as client:
        response = await request_with_retry(
            client, "POST", "/x", error_cls=TrackerError, max_attempts=3, sleep=_no_sleep
        )
    assert response.status_code == 429  # caller handles terminal status


async def test_503_not_retried_for_non_idempotent() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with _client(handler) as client:
        response = await request_with_retry(
            client, "POST", "/create", error_cls=TrackerError, sleep=_no_sleep
        )
    assert response.status_code == 503
    assert calls == 1


async def test_503_retried_when_idempotent() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(503)
        return httpx.Response(200)

    async with _client(handler) as client:
        response = await request_with_retry(
            client, "GET", "/x", error_cls=BackendError, idempotent=True, sleep=_no_sleep
        )
    assert response.status_code == 200
    assert calls == 2


async def test_transport_error_raises_typed_error_when_not_idempotent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with _client(handler) as client:
        with pytest.raises(TrackerError, match="failed"):
            await request_with_retry(
                client, "POST", "/create", error_cls=TrackerError, sleep=_no_sleep
            )


async def test_transport_error_retried_then_typed_error_when_idempotent() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("boom")

    async with _client(handler) as client:
        with pytest.raises(BackendError, match="after 3 attempts"):
            await request_with_retry(
                client,
                "GET",
                "/x",
                error_cls=BackendError,
                idempotent=True,
                max_attempts=3,
                sleep=_no_sleep,
            )
    assert calls == 3


async def test_transport_error_recovers_when_idempotent() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200)

    async with _client(handler) as client:
        response = await request_with_retry(
            client, "GET", "/x", error_cls=BackendError, idempotent=True, sleep=_no_sleep
        )
    assert response.status_code == 200


async def test_retry_after_header_caps_sleep() -> None:
    slept: list[float] = []

    async def record_sleep(delay: float) -> None:
        slept.append(delay)

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "86400"})
        return httpx.Response(200)

    async with _client(handler) as client:
        await request_with_retry(
            client, "GET", "/x", error_cls=BackendError, max_delay=60.0, sleep=record_sleep
        )
    assert slept == [60.0]

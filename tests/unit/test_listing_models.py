"""Unit tests for `agent_triage.models.listing` (proposal 001 Specs B + C)."""

from datetime import UTC, datetime

import pytest

from agent_triage.errors import BackendError
from agent_triage.models.listing import (
    RESERVED_FILTER_KEYS,
    TraceFilter,
    TraceListing,
    TraceSummary,
    parse_status_filter,
)


def test_reserved_filter_keys_contains_status() -> None:
    assert "status" in RESERVED_FILTER_KEYS


def test_trace_summary_id_only_is_valid() -> None:
    summary = TraceSummary(trace_id="t-1")
    assert summary.status is None
    assert summary.start_time is None
    assert summary.latency_ms is None
    assert summary.tags == {}


def test_trace_listing_defaults_and_trace_ids() -> None:
    listing = TraceListing(summaries=[TraceSummary(trace_id="a"), TraceSummary(trace_id="b")])
    assert listing.truncated is False
    assert listing.page_limit is None
    assert listing.trace_ids == ["a", "b"]


def test_trace_listing_round_trips_through_json() -> None:
    """The listing crosses the MCP seam as JSON; it must round-trip."""
    listing = TraceListing(
        summaries=[
            TraceSummary(
                trace_id="t-1",
                start_time=datetime(2026, 5, 22, tzinfo=UTC),
                status="error",
                latency_ms=1234.5,
                tags={"tenant_id": "acme"},
            )
        ],
        truncated=True,
        page_limit=50,
    )
    restored = TraceListing.model_validate_json(listing.model_dump_json())
    assert restored == listing


def test_trace_filter_serializes_only_set_keys() -> None:
    assert TraceFilter(status="error").to_filter_dict() == {"status": "error"}
    assert TraceFilter().to_filter_dict() == {}


def test_parse_status_filter_accepts_valid_values() -> None:
    assert parse_status_filter(None) is None
    assert parse_status_filter({}) is None
    assert parse_status_filter({"other": 1}) is None
    assert parse_status_filter({"status": "error"}) == "error"
    assert parse_status_filter({"status": "ok"}) == "ok"


def test_parse_status_filter_rejects_unsupported_values() -> None:
    with pytest.raises(BackendError, match="unsupported value"):
        parse_status_filter({"status": "failed"})

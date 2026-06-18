"""Smoke tests for the TraceBackend ABC contract."""

import pytest

from docket.adapters.base import TraceBackend


def test_trace_backend_cannot_instantiate_directly() -> None:
    with pytest.raises(TypeError, match="abstract"):
        TraceBackend()  # type: ignore[abstract]


async def test_trace_backend_default_close_is_noop() -> None:
    class _Stub(TraceBackend):
        async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
            return []

        async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
            return None

        async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
            return []

        async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
            return None

        async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
            return set()

    stub = _Stub()
    await stub.close()

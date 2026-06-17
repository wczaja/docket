"""Tests for the Annotator subagent: positive-only writeback + retry + per-write abort."""

import pytest

from agent_triage.adapters.base import TraceBackend
from agent_triage.agent.subagents.annotator import Annotator
from agent_triage.errors import BackendError
from agent_triage.models.classification import Annotation, Classification
from agent_triage.rubric.spec import Detection, Mode, Rubric, RubricMetadata


class _FakeBackend(TraceBackend):
    def __init__(self, fail_until_attempt: int = 0) -> None:
        self._fail_until_attempt = fail_until_attempt
        self._attempt = 0
        self.annotations: list[Annotation] = []

    async def list_traces(self, since, until=None, filter=None):  # type: ignore[no-untyped-def]
        return []

    async def get_trace(self, trace_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def annotate_trace(self, trace_id, annotation):  # type: ignore[no-untyped-def]
        self._attempt += 1
        if self._attempt <= self._fail_until_attempt:
            msg = f"transient backend failure attempt {self._attempt}"
            raise BackendError(msg)
        self.annotations.append(annotation)

    async def search_traces(self, query, k=10):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def mark_trace_processed(self, trace_id, *, run_id, rubric_version):  # type: ignore[no-untyped-def]
        pass

    async def list_processed_trace_ids(self, *, run_id, since, until=None):  # type: ignore[no-untyped-def]
        return set()


def _rubric() -> Rubric:
    return Rubric(
        apiVersion="agent-triage.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="t", version="0.1.0"),
        modes=[
            Mode(id="m1", severity="critical", detection=Detection(type="regex", pattern="x")),
            Mode(id="m2", severity="low", detection=Detection(type="regex", pattern="x")),
        ],
    )


def _classification(
    trace_id: str,
    mode_id: str,
    *,
    positive: bool = True,
    error: str | None = None,
) -> Classification:
    return Classification(
        trace_id=trace_id,
        rubric_version="t@0.1.0",
        mode_id=mode_id,
        positive=positive,
        error=error,
    )


async def test_annotator_writes_positive_only() -> None:
    backend = _FakeBackend()
    annotator = Annotator(backend, run_id="r-1")
    rubric = _rubric()
    classifications = [
        _classification("t-1", "m1", positive=True),
        _classification("t-1", "m2", positive=False),
        _classification("t-2", "m1", positive=True, error="failed"),  # has error -> skip
        _classification("t-3", "m1", positive=True),
    ]
    written = await annotator.annotate_positive(classifications, rubric)
    assert written == 2
    assert {a.trace_id for a in backend.annotations} == {"t-1", "t-3"}
    for a in backend.annotations:
        assert a.run_id == "r-1"
        assert a.severity == "critical"


async def test_annotator_retries_transient_failures() -> None:
    backend = _FakeBackend(fail_until_attempt=2)
    annotator = Annotator(
        backend,
        run_id="r-1",
        max_retries=5,
        backoff_base_s=0.0,
    )
    rubric = _rubric()
    classifications = [_classification("t-1", "m1", positive=True)]
    written = await annotator.annotate_positive(classifications, rubric)
    assert written == 1
    assert len(backend.annotations) == 1


async def test_annotator_aborts_after_max_retries() -> None:
    # Fail more times than max_retries allows.
    backend = _FakeBackend(fail_until_attempt=100)
    annotator = Annotator(
        backend,
        run_id="r-1",
        max_retries=3,
        backoff_base_s=0.0,
    )
    rubric = _rubric()
    classifications = [_classification("t-1", "m1", positive=True)]
    with pytest.raises(BackendError, match="after 3 attempts"):
        await annotator.annotate_positive(classifications, rubric)


async def test_annotator_ignores_classifications_for_unknown_mode() -> None:
    backend = _FakeBackend()
    annotator = Annotator(backend, run_id="r-1")
    rubric = _rubric()
    classifications = [_classification("t-1", "ghost", positive=True)]
    written = await annotator.annotate_positive(classifications, rubric)
    assert written == 0
    assert backend.annotations == []


def test_annotator_rejects_zero_retries() -> None:
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="max_retries"):
        Annotator(backend, run_id="r-1", max_retries=0)

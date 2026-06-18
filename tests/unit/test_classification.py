from datetime import UTC, datetime

from docket.models.classification import Annotation, Classification


def test_classification_construct() -> None:
    c = Classification(
        trace_id="t1",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        extra={"excerpt": "..."},
        duration_ms=42.5,
    )
    assert c.positive
    assert c.extra["excerpt"] == "..."


def test_classification_is_frozen() -> None:
    c = Classification(
        trace_id="t",
        rubric_version="v",
        mode_id="m",
        positive=False,
    )
    try:
        c.positive = True  # type: ignore[misc]
    except (TypeError, ValueError, AttributeError):
        return
    msg = "Classification was not frozen"
    raise AssertionError(msg)


def test_annotation_idempotency_key() -> None:
    a = Annotation(
        trace_id="trace-1",
        run_id="run-7",
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        positive=True,
        severity="critical",
    )
    assert a.idempotency_key() == "trace-1|run-7|agents-builtin@1.0.0|hallucination"


def test_annotation_default_created_at_is_recent() -> None:
    a = Annotation(
        trace_id="t",
        run_id="r",
        rubric_version="v",
        mode_id="m",
        positive=False,
        severity="low",
    )
    delta = abs((datetime.now(UTC) - a.created_at).total_seconds())
    assert delta < 5.0

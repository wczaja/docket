from typing import Any

import pytest

from agent_triage.detectors import get_detector
from agent_triage.detectors.composite import CompositeDetector
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import TraceLike
from agent_triage.rubric.spec import Detection, Mode


class _CountingProvider(ModelProvider):
    def __init__(self, positive: bool = True) -> None:
        self.model = "mock:1"
        self.calls = 0
        self._positive = positive

    async def structured_complete(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        return {"positive": self._positive}


def _judge_operand() -> Detection:
    return Detection(
        type="llm_judge",
        prompt="Did the handoff fail?",
        output_schema={
            "type": "object",
            "required": ["positive"],
            "properties": {"positive": {"type": "boolean"}},
        },
    )


def _composite_mode(operator: str, operands: list[Detection]) -> Mode:
    return Mode(
        id="m",
        severity="high",
        detection=Detection(type="composite", operator=operator, operands=operands),  # type: ignore[arg-type]
    )


async def test_composite_and_both_true() -> None:
    mode = _composite_mode(
        "and",
        [Detection(type="regex", pattern="hello"), Detection(type="regex", pattern="world")],
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello world"))
    assert verdict.positive
    assert len(verdict.extra["sub_verdicts"]) == 2


async def test_composite_and_one_false() -> None:
    mode = _composite_mode(
        "and",
        [Detection(type="regex", pattern="hello"), Detection(type="regex", pattern="world")],
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello there"))
    assert not verdict.positive


async def test_composite_or_one_true() -> None:
    mode = _composite_mode(
        "or",
        [Detection(type="regex", pattern="hello"), Detection(type="regex", pattern="world")],
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello there"))
    assert verdict.positive


async def test_composite_or_all_false() -> None:
    mode = _composite_mode(
        "or",
        [Detection(type="regex", pattern="hello"), Detection(type="regex", pattern="world")],
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    verdict = await detector.evaluate(mode, TraceLike(full_text="goodbye"))
    assert not verdict.positive


async def test_composite_and_short_circuits_before_llm_judge() -> None:
    """A negative cheap operand in an AND must spare the judge entirely."""
    provider = _CountingProvider()
    mode = _composite_mode(
        "and",
        [Detection(type="regex", pattern="no-such-text"), _judge_operand()],
    )
    detector = CompositeDetector(lambda t: get_detector(t, llm_provider=provider))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello"))
    assert not verdict.positive
    assert provider.calls == 0
    # Evidence covers evaluated operands only.
    assert len(verdict.extra["sub_verdicts"]) == 1


async def test_composite_and_runs_llm_judge_when_gate_passes() -> None:
    provider = _CountingProvider(positive=True)
    mode = _composite_mode(
        "and",
        [Detection(type="regex", pattern="hello"), _judge_operand()],
    )
    detector = CompositeDetector(lambda t: get_detector(t, llm_provider=provider))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello"))
    assert verdict.positive
    assert provider.calls == 1


async def test_composite_and_orders_cheap_operand_before_llm_judge() -> None:
    """Even when the rubric lists the judge first, the deterministic operand
    is evaluated first and a negative gate spares the judge."""
    provider = _CountingProvider()
    mode = _composite_mode(
        "and",
        [_judge_operand(), Detection(type="regex", pattern="no-such-text")],
    )
    detector = CompositeDetector(lambda t: get_detector(t, llm_provider=provider))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello"))
    assert not verdict.positive
    assert provider.calls == 0


async def test_composite_or_short_circuits_before_llm_judge() -> None:
    """A positive cheap operand in an OR must spare the judge entirely."""
    provider = _CountingProvider()
    mode = _composite_mode(
        "or",
        [Detection(type="regex", pattern="hello"), _judge_operand()],
    )
    detector = CompositeDetector(lambda t: get_detector(t, llm_provider=provider))
    verdict = await detector.evaluate(mode, TraceLike(full_text="hello"))
    assert verdict.positive
    assert provider.calls == 0


async def test_composite_nested_deterministic_composite_sorts_before_judge() -> None:
    """A nested composite built only from deterministic operands counts as
    cheap and is evaluated before the judge."""
    provider = _CountingProvider()
    nested = Detection(
        type="composite",
        operator="and",
        operands=[Detection(type="regex", pattern="a"), Detection(type="regex", pattern="zzz")],
    )
    mode = _composite_mode("and", [_judge_operand(), nested])
    detector = CompositeDetector(lambda t: get_detector(t, llm_provider=provider))
    verdict = await detector.evaluate(mode, TraceLike(full_text="a"))
    assert not verdict.positive
    assert provider.calls == 0


@pytest.mark.parametrize("operator", ["and", "or"])
@pytest.mark.parametrize(
    "operand_outcomes",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        (True, True, False),
        (False, False, False),
    ],
)
async def test_composite_short_circuit_preserves_verdict_semantics(
    operator: str, operand_outcomes: tuple[bool, ...]
) -> None:
    """Truth table: the short-circuiting verdict equals the old all-evaluate
    fold (all() for AND, any() for OR)."""
    operands = [
        Detection(type="regex", pattern="match" if outcome else "no-such-text")
        for outcome in operand_outcomes
    ]
    mode = _composite_mode(operator, operands)
    detector = CompositeDetector(lambda t: get_detector(t))
    verdict = await detector.evaluate(mode, TraceLike(full_text="match"))
    expected = all(operand_outcomes) if operator == "and" else any(operand_outcomes)
    assert verdict.positive == expected


async def test_composite_missing_operands_raises() -> None:
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(type="composite", operator="and"),
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    with pytest.raises(DetectionError, match="requires `operands`"):
        await detector.evaluate(mode, TraceLike(full_text=""))


async def test_composite_bad_operator_raises() -> None:
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(
            type="composite",
            operator=">",
            operands=[Detection(type="regex", pattern="x")],
        ),
    )
    detector = CompositeDetector(lambda t: get_detector(t))
    with pytest.raises(DetectionError, match="must be 'and' or 'or'"):
        await detector.evaluate(mode, TraceLike(full_text=""))

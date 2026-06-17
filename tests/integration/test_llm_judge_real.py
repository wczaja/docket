"""One real-LLM smoke test that covers the §7 Phase 2 acceptance criterion:

  "One integration test runs llm_judge against a real model with a
   deterministic test rubric and verifies positive + negative cases."

Gated behind `pytest --run-integration` AND `ANTHROPIC_API_KEY` in the env so
default runs (including CI without secrets) skip it. The rubric is synthetic
and the trace excerpts are obvious-enough that Haiku 4.5 should classify them
correctly under normal conditions.
"""

import pytest

from agent_triage.detectors.llm_judge import LLMJudgeDetector
from agent_triage.llm._anthropic import AnthropicProvider
from agent_triage.models.trace import TraceLike
from agent_triage.rubric.spec import Detection, Mode

pytestmark = pytest.mark.integration


_HALLUCINATION_MODE = Mode(
    id="hallucination",
    severity="critical",
    detection=Detection(
        type="llm_judge",
        prompt=(
            "The trace contains an agent's response. Decide whether the response "
            "contains an obviously fabricated or factually-impossible claim about "
            "well-known reality (geography, basic arithmetic, established facts). "
            "Return positive=true if it does."
        ),
        output_schema={
            "type": "object",
            "required": ["positive"],
            "properties": {"positive": {"type": "boolean"}},
        },
    ),
)


async def test_real_anthropic_classifies_obvious_hallucination_as_positive() -> None:
    detector = LLMJudgeDetector(AnthropicProvider())
    trace = TraceLike(full_text="The agent said: 'The capital of France is Tokyo.'")
    verdict = await detector.evaluate(_HALLUCINATION_MODE, trace)
    assert verdict.positive, f"expected positive verdict, got {verdict.extra!r}"


async def test_real_anthropic_classifies_correct_statement_as_negative() -> None:
    detector = LLMJudgeDetector(AnthropicProvider())
    trace = TraceLike(full_text="The agent said: 'The capital of France is Paris.'")
    verdict = await detector.evaluate(_HALLUCINATION_MODE, trace)
    assert not verdict.positive, f"expected negative verdict, got {verdict.extra!r}"

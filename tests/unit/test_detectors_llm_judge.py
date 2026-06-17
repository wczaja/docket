from typing import Any
from unittest.mock import patch

import pytest

from agent_triage.detectors import llm_judge as llm_judge_module
from agent_triage.detectors.llm_judge import LLMJudgeDetector
from agent_triage.errors import DetectionError
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import TraceLike
from agent_triage.rubric.spec import Detection, Mode


class MockProvider(ModelProvider):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.model = "mock:1"
        self._responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise RuntimeError("MockProvider has no more queued responses")
        return self._responses.pop(0)


def _llm_judge_mode(model_uri: str | None = None) -> Mode:
    return Mode(
        id="hallucination",
        severity="critical",
        detection=Detection(
            type="llm_judge",
            prompt="Detect hallucination.",
            output_schema={
                "type": "object",
                "required": ["positive"],
                "properties": {"positive": {"type": "boolean"}},
            },
            model=model_uri,
        ),
    )


async def test_llm_judge_positive() -> None:
    provider = MockProvider([{"positive": True}])
    detector = LLMJudgeDetector(provider)
    verdict = await detector.evaluate(_llm_judge_mode(), TraceLike(full_text="suspicious"))
    assert verdict.positive
    assert verdict.extra == {"positive": True}


async def test_llm_judge_negative() -> None:
    provider = MockProvider([{"positive": False}])
    detector = LLMJudgeDetector(provider)
    verdict = await detector.evaluate(_llm_judge_mode(), TraceLike(full_text="clean trace"))
    assert not verdict.positive


async def test_llm_judge_redacts_pii_before_sending() -> None:
    provider = MockProvider([{"positive": False}])
    detector = LLMJudgeDetector(provider)
    await detector.evaluate(
        _llm_judge_mode(),
        TraceLike(full_text="please email user@example.com about it"),
    )
    _system, user, _schema = provider.calls[0]
    assert "user@example.com" not in user
    assert "[REDACTED_EMAIL]" in user


async def test_llm_judge_redacts_context_single() -> None:
    provider = MockProvider([{"positive": False}])
    detector = LLMJudgeDetector(provider)
    await detector.evaluate(
        _llm_judge_mode(),
        TraceLike(full_text="clean trace", context="Customer phone: (555) 867-5309"),
    )
    _system, user, _schema = provider.calls[0]
    assert "867-5309" not in user
    assert "[REDACTED_PHONE]" in user


async def test_llm_judge_redacts_context_batch() -> None:
    provider = MockProvider([{"verdicts": [{"positive": False}, {"positive": False}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    await detector.evaluate_batch(
        _llm_judge_mode(),
        [
            TraceLike(full_text="a", context="call +1 555-867-5309"),
            TraceLike(full_text="b", context="email bob@example.com"),
        ],
    )
    _system, user, _schema = provider.calls[0]
    assert "555-867-5309" not in user
    assert "[REDACTED_PHONE]" in user
    assert "bob@example.com" not in user
    assert "[REDACTED_EMAIL]" in user


async def test_llm_judge_raises_when_prompt_missing() -> None:
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(
            type="llm_judge",
            output_schema={"type": "object", "properties": {"positive": {"type": "boolean"}}},
        ),
    )
    detector = LLMJudgeDetector(MockProvider([]))
    with pytest.raises(DetectionError, match="requires `prompt`"):
        await detector.evaluate(mode, TraceLike(full_text="x"))


async def test_llm_judge_raises_when_schema_missing() -> None:
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(type="llm_judge", prompt="Detect."),
    )
    detector = LLMJudgeDetector(MockProvider([]))
    with pytest.raises(DetectionError, match="requires `prompt` and `output_schema`"):
        await detector.evaluate(mode, TraceLike(full_text="x"))


async def test_llm_judge_missing_positive_raises() -> None:
    """A response without `positive` raises instead of silently scoring negative."""
    provider = MockProvider([{"something_else": 1}])
    detector = LLMJudgeDetector(provider)
    with pytest.raises(DetectionError, match="violates `output_schema`"):
        await detector.evaluate(_llm_judge_mode(), TraceLike(full_text="x"))


async def test_llm_judge_non_bool_positive_raises() -> None:
    provider = MockProvider([{"positive": "yes"}])
    detector = LLMJudgeDetector(provider)
    with pytest.raises(DetectionError, match="violates `output_schema`"):
        await detector.evaluate(_llm_judge_mode(), TraceLike(full_text="x"))


async def test_llm_judge_non_bool_positive_raises_even_with_permissive_schema() -> None:
    """Even when the schema does not constrain `positive`, a non-boolean value
    must not be coerced via bool()."""
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(
            type="llm_judge",
            prompt="Detect.",
            output_schema={"type": "object"},
        ),
    )
    provider = MockProvider([{"positive": "yes"}])
    detector = LLMJudgeDetector(provider)
    with pytest.raises(DetectionError, match="missing a boolean `positive`"):
        await detector.evaluate(mode, TraceLike(full_text="x"))


async def test_llm_judge_schema_violating_response_raises() -> None:
    mode = Mode(
        id="m",
        severity="low",
        detection=Detection(
            type="llm_judge",
            prompt="Detect.",
            output_schema={
                "type": "object",
                "required": ["positive", "confidence"],
                "properties": {
                    "positive": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        ),
    )
    provider = MockProvider([{"positive": True, "confidence": "high"}])
    detector = LLMJudgeDetector(provider)
    with pytest.raises(DetectionError, match="violates `output_schema`"):
        await detector.evaluate(mode, TraceLike(full_text="x"))


async def test_llm_judge_batch_missing_positive_raises() -> None:
    provider = MockProvider([{"verdicts": [{"positive": True}, {"oops": 1}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    with pytest.raises(DetectionError, match="violates `output_schema`"):
        await detector.evaluate_batch(
            _llm_judge_mode(), [TraceLike(full_text="a"), TraceLike(full_text="b")]
        )


async def test_llm_judge_batch_non_bool_positive_raises() -> None:
    provider = MockProvider([{"verdicts": [{"positive": 1}, {"positive": False}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    with pytest.raises(DetectionError, match="violates `output_schema`"):
        await detector.evaluate_batch(
            _llm_judge_mode(), [TraceLike(full_text="a"), TraceLike(full_text="b")]
        )


async def test_llm_judge_batched_call_count() -> None:
    """20 traces with batch_size=8 should call the provider exactly 3 times
    (8 + 8 + 4). Verifies the §7 budget-mode acceptance criterion."""
    provider = MockProvider(
        [
            {"verdicts": [{"positive": True}] * 8},
            {"verdicts": [{"positive": True}] * 8},
            {"verdicts": [{"positive": True}] * 4},
        ]
    )
    detector = LLMJudgeDetector(provider, batch_size=8)
    traces = [TraceLike(full_text=f"trace-{i}") for i in range(20)]
    verdicts = await detector.evaluate_batch(_llm_judge_mode(), traces)
    assert len(verdicts) == 20
    assert all(v.positive for v in verdicts)
    assert len(provider.calls) == 3


async def test_llm_judge_batch_wraps_schema() -> None:
    provider = MockProvider([{"verdicts": [{"positive": True}, {"positive": False}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    traces = [TraceLike(full_text="a"), TraceLike(full_text="b")]
    verdicts = await detector.evaluate_batch(_llm_judge_mode(), traces)
    assert [v.positive for v in verdicts] == [True, False]
    _system, _user, schema = provider.calls[0]
    assert schema["properties"]["verdicts"]["minItems"] == 2
    assert schema["properties"]["verdicts"]["maxItems"] == 2


async def test_llm_judge_batch_wrong_count_raises() -> None:
    provider = MockProvider([{"verdicts": [{"positive": True}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    traces = [TraceLike(full_text="a"), TraceLike(full_text="b")]
    with pytest.raises(DetectionError, match="expected 2 verdicts"):
        await detector.evaluate_batch(_llm_judge_mode(), traces)


async def test_llm_judge_batch_missing_verdicts_raises() -> None:
    provider = MockProvider([{"wrong_key": []}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    with pytest.raises(DetectionError, match="did not return a `verdicts` array"):
        await detector.evaluate_batch(
            _llm_judge_mode(), [TraceLike(full_text="a"), TraceLike(full_text="b")]
        )


async def test_llm_judge_batch_non_object_entry_raises() -> None:
    provider = MockProvider([{"verdicts": ["not-an-object", {"positive": True}]}])
    detector = LLMJudgeDetector(provider, batch_size=8)
    with pytest.raises(DetectionError, match="batched verdict was not an object"):
        await detector.evaluate_batch(
            _llm_judge_mode(), [TraceLike(full_text="a"), TraceLike(full_text="b")]
        )


async def test_llm_judge_per_mode_model_override() -> None:
    """A mode with `model: openai:gpt-4o-mini` should resolve through
    build_provider, not the default provider injected into the detector.
    """
    default = MockProvider([{"positive": False}])
    override = MockProvider([{"positive": True}])
    detector = LLMJudgeDetector(default)
    mode = _llm_judge_mode(model_uri="openai:gpt-4o-mini")
    with patch.object(llm_judge_module, "build_provider", return_value=override):
        verdict = await detector.evaluate(mode, TraceLike(full_text="x"))
    assert verdict.positive
    assert default.calls == []
    assert len(override.calls) == 1


def test_llm_judge_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be"):
        LLMJudgeDetector(MockProvider([]), batch_size=0)


async def test_llm_judge_batch_size_one_uses_single_path() -> None:
    """batch_size=1 with 3 traces still produces 3 separate single-trace calls
    (NOT a batched call). Verifies the unwrapped-schema path is taken."""
    provider = MockProvider([{"positive": True}, {"positive": False}, {"positive": True}])
    detector = LLMJudgeDetector(provider, batch_size=1)
    traces = [TraceLike(full_text=f"t{i}") for i in range(3)]
    verdicts = await detector.evaluate_batch(_llm_judge_mode(), traces)
    assert [v.positive for v in verdicts] == [True, False, True]
    assert len(provider.calls) == 3
    # Each call used the unwrapped schema, not the batched wrapper.
    for _system, _user, schema in provider.calls:
        assert "verdicts" not in schema.get("properties", {})


async def test_per_mode_model_override_provider_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m-15: one provider per override URI per detector, not one per evaluation."""
    built: list[str] = []
    canned = MockProvider([{"positive": False}] * 3)

    def fake_build(uri: str) -> ModelProvider:
        built.append(uri)
        return canned

    monkeypatch.setattr(llm_judge_module, "build_provider", fake_build)
    detector = LLMJudgeDetector(MockProvider([]))
    mode = _llm_judge_mode(model_uri="anthropic:claude-test")
    trace = TraceLike(full_text="nothing to see")
    await detector.evaluate(mode, trace)
    await detector.evaluate(mode, trace)
    await detector.evaluate(mode, trace)
    assert built == ["anthropic:claude-test"]

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from docket.detectors import llm_judge as llm_judge_module
from docket.detectors.llm_judge import LLMJudgeDetector
from docket.errors import DetectionError
from docket.llm.base import ModelProvider
from docket.models.trace import TraceLike
from docket.rubric.spec import Detection, Mode, Rubric


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


# --- evidence normalization + excerpt verification ------------------------------

_ANSWER = "[user] Which version added feature X?\n[assistant] Feature X was added in v2.1."


def _evidence_mode(properties: dict[str, Any]) -> Mode:
    return Mode(
        id="off-corpus-answer",
        severity="critical",
        detection=Detection(
            type="llm_judge",
            prompt="Detect ungrounded claims.",
            output_schema={
                "type": "object",
                "required": ["positive"],
                "properties": {"positive": {"type": "boolean"}, **properties},
            },
        ),
    )


_UNGROUNDED_MODE = _evidence_mode(
    {
        "ungrounded_excerpt": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
)


async def test_llm_judge_copies_mode_evidence_property_into_excerpt() -> None:
    response = {
        "positive": True,
        "ungrounded_excerpt": "Feature X was added in v2.1.",
        "confidence": 0.9,
    }
    detector = LLMJudgeDetector(MockProvider([dict(response)]))
    verdict = await detector.evaluate(_UNGROUNDED_MODE, TraceLike(full_text=_ANSWER))
    assert verdict.extra == {
        **response,
        "excerpt": "Feature X was added in v2.1.",
        "excerpt_verified": True,
    }


async def test_llm_judge_does_not_mutate_the_provider_response() -> None:
    response = {"positive": True, "ungrounded_excerpt": "Feature X was added in v2.1."}
    detector = LLMJudgeDetector(MockProvider([response]))
    await detector.evaluate(_UNGROUNDED_MODE, TraceLike(full_text=_ANSWER))
    assert response == {"positive": True, "ungrounded_excerpt": "Feature X was added in v2.1."}


async def test_llm_judge_excerpt_from_judge_wins_over_schema_evidence() -> None:
    """A judge-returned `excerpt` is kept even when the schema declares a
    different evidence property (providers don't forbid undeclared keys)."""
    mode = _evidence_mode({"reason": {"type": "string"}})
    response = {
        "positive": True,
        "excerpt": "Feature X was added in v2.1.",
        "reason": "claim absent from retrieval",
    }
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(mode, TraceLike(full_text=_ANSWER))
    assert verdict.extra["excerpt"] == "Feature X was added in v2.1."
    assert verdict.extra["reason"] == "claim absent from retrieval"


async def test_llm_judge_evidence_is_first_nonblank_string_in_declaration_order() -> None:
    mode = _evidence_mode(
        {
            "staleness_days": {"type": ["integer", "null"]},
            "chosen_specialist": {"type": ["string", "null"]},
            "expected_specialist": {"type": ["string", "null"]},
        }
    )
    response = {
        "positive": True,
        "excerpt": "  ",
        "staleness_days": 400,
        "chosen_specialist": None,
        "expected_specialist": "refunds_specialist",
    }
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(mode, TraceLike(full_text="routed to refunds_specialist"))
    assert verdict.extra["excerpt"] == "refunds_specialist"
    assert verdict.extra["excerpt_verified"] is True


async def test_llm_judge_positive_without_text_evidence_adds_no_keys() -> None:
    mode = _evidence_mode({"staleness_days": {"type": ["integer", "null"]}})
    response = {"positive": True, "staleness_days": 540}
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(mode, TraceLike(full_text="pricing page dated 2024"))
    assert verdict.extra == {"positive": True, "staleness_days": 540}


async def test_llm_judge_negative_verdict_gets_no_evidence_keys() -> None:
    response = {"positive": False, "ungrounded_excerpt": "not a failure"}
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(_UNGROUNDED_MODE, TraceLike(full_text=_ANSWER))
    assert verdict.extra == {"positive": False, "ungrounded_excerpt": "not a failure"}


async def test_llm_judge_flags_but_keeps_excerpt_missing_from_trace() -> None:
    """A hallucinated quote is flagged, never silently dropped."""
    response = {"positive": True, "ungrounded_excerpt": "Feature X shipped in v3.0."}
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(_UNGROUNDED_MODE, TraceLike(full_text=_ANSWER))
    assert verdict.extra["excerpt"] == "Feature X shipped in v3.0."
    assert verdict.extra["excerpt_verified"] is False


@pytest.mark.parametrize(
    ("excerpt", "verified"),
    [
        ('"feature x WAS  added\nin v2.1"', True),  # case, whitespace, wrapping quotes
        ("Which version … Feature X was added", True),  # unicode ellipsis elision
        ("Which version [...] v2.1", True),  # bracketed elision
        ("Which version ... v2.1 ... feature X", False),  # elided parts out of order
        ("It’s Feature X", False),  # curly apostrophe normalized, still absent
        ("...", False),  # no content left to check
    ],
)
async def test_llm_judge_excerpt_check_is_lenient_on_form_strict_on_content(
    excerpt: str, verified: bool
) -> None:
    response = {"positive": True, "ungrounded_excerpt": excerpt}
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(_UNGROUNDED_MODE, TraceLike(full_text=_ANSWER))
    assert verdict.extra["excerpt_verified"] is verified


async def test_llm_judge_excerpt_check_uses_the_redacted_text_the_judge_saw() -> None:
    """The judge only ever sees redacted text, so it quotes placeholders; a
    PII-shaped string in the excerpt gets the same `redact()` pass."""
    trace = TraceLike(full_text="[assistant] I emailed jane.doe@example.com the refund.")
    responses = [
        {"positive": True, "ungrounded_excerpt": "I emailed [REDACTED_EMAIL] the refund"},
        {"positive": True, "ungrounded_excerpt": "I emailed someone@else.org the refund"},
    ]
    detector = LLMJudgeDetector(MockProvider(responses))
    placeholder = await detector.evaluate(_UNGROUNDED_MODE, trace)
    raw_pii = await detector.evaluate(_UNGROUNDED_MODE, trace)
    assert placeholder.extra["excerpt_verified"] is True
    assert raw_pii.extra["excerpt_verified"] is True


async def test_llm_judge_excerpt_may_quote_the_context() -> None:
    """The context block is part of what the judge saw."""
    response = {"positive": True, "ungrounded_excerpt": "release notes for v3.4"}
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(
        _UNGROUNDED_MODE,
        TraceLike(full_text=_ANSWER, context="Retrieved: release notes for v3.4 and v3.5."),
    )
    assert verdict.extra["excerpt_verified"] is True


async def test_llm_judge_batch_checks_each_excerpt_against_its_own_trace() -> None:
    provider = MockProvider(
        [
            {
                "verdicts": [
                    {"positive": True, "ungrounded_excerpt": "Feature Y was added in v9"},
                    {"positive": True, "ungrounded_excerpt": "Feature Y was added in v9"},
                    {"positive": False, "ungrounded_excerpt": None},
                ]
            }
        ]
    )
    detector = LLMJudgeDetector(provider, batch_size=8)
    verdicts = await detector.evaluate_batch(
        _UNGROUNDED_MODE,
        [
            TraceLike(full_text=_ANSWER),
            TraceLike(full_text="[assistant] Feature Y was added in v9."),
            TraceLike(full_text="[assistant] Feature Z is on the roadmap."),
        ],
    )
    assert [v.extra.get("excerpt_verified") for v in verdicts] == [False, True, None]
    assert verdicts[0].extra["excerpt"] == "Feature Y was added in v9"
    assert "excerpt" not in verdicts[2].extra


# --- every shipped rubric surfaces its judge's evidence, unedited ---------------

_REPO = Path(__file__).resolve().parents[2]
_SHIPPED_RUBRICS = [
    *sorted((_REPO / "docket" / "rubric" / "builtin").glob("*/v1/rubric.yaml")),
    *sorted((_REPO / "rubrics" / "registry").glob("*/v1/rubric.yaml")),
    *sorted((_REPO / "rubrics" / "examples").glob("*.yaml")),
]


def _judges(detection: Detection) -> list[Detection]:
    if detection.type == "llm_judge":
        return [detection]
    return [judge for operand in detection.operands or [] for judge in _judges(operand)]


_SHIPPED_JUDGE_MODES = [
    pytest.param(
        mode.model_copy(update={"detection": judge}),
        id=f"{path.relative_to(_REPO).parent}:{mode.id}",
    )
    for path in _SHIPPED_RUBRICS
    for mode in Rubric.model_validate(yaml.safe_load(path.read_text())).modes
    for judge in _judges(mode.detection)
]


def _positive_response(output_schema: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """A schema-valid positive verdict with every property filled in, and the
    evidence docket should surface: the first string-typed property's value."""
    response: dict[str, Any] = {}
    evidence: str | None = None
    for name, spec in output_schema["properties"].items():
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if name == "positive":
            response[name] = True
        elif "string" in types:
            response[name] = f"{name} evidence"
            evidence = evidence or response[name]
        elif "integer" in types:
            response[name] = 7
        elif "number" in types:
            response[name] = 0.5
    return response, evidence


def test_sweep_covers_every_shipped_judge() -> None:
    assert len(_SHIPPED_JUDGE_MODES) == 43


@pytest.mark.parametrize("mode", _SHIPPED_JUDGE_MODES)
async def test_shipped_rubric_judge_evidence_reaches_excerpt(mode: Mode) -> None:
    assert mode.detection.output_schema is not None
    response, evidence = _positive_response(mode.detection.output_schema)
    detector = LLMJudgeDetector(MockProvider([response]))
    verdict = await detector.evaluate(mode, TraceLike(full_text="trace"))
    assert verdict.positive
    assert verdict.extra.get("excerpt") == evidence

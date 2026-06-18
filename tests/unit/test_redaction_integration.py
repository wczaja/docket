"""End-to-end redaction test: PII in a trace is scrubbed before reaching the LLM judge.

Covers §7 Phase 3 acceptance #3: "Redaction hook example: scrubs email
addresses, phone numbers, account numbers before classifier sees the trace."
"""

import copy
import json
from pathlib import Path
from typing import Any

from docket.agent.subagents.clusterer import cluster_per_mode
from docket.agent.subagents.drafter import draft_issues
from docket.detectors.llm_judge import LLMJudgeDetector
from docket.llm.base import ModelProvider
from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Classification
from docket.models.otlp import from_otlp
from docket.rubric.spec import Clustering, Detection, Mode, Rubric, RubricMetadata


class _RecordingProvider(ModelProvider):
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.model = "recording:1"
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._response = response if response is not None else {"positive": False}

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((system, user, schema))
        return dict(self._response)


class _RecordingEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = "recording-embed"
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def _llm_judge_mode() -> Mode:
    return Mode(
        id="any-failure",
        severity="high",
        detection=Detection(
            type="llm_judge",
            prompt="Detect any failure mode.",
            output_schema={
                "type": "object",
                "required": ["positive"],
                "properties": {"positive": {"type": "boolean"}},
            },
        ),
    )


def _inject_pii(otlp: dict[str, Any]) -> dict[str, Any]:
    """Replace the user message content in simple_llm_call.json with PII-rich text."""
    result = copy.deepcopy(otlp)
    spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
    for attr in spans[0]["attributes"]:
        if attr["key"] == "llm.input_messages.0.message.content":
            attr["value"]["stringValue"] = (
                "Contact me at jane.doe@example.com or call 555-867-5309. "
                "My SSN is 123-45-6789 and my account #4567890123 is overdue."
            )
            break
    return result


async def test_trace_to_judge_path_redacts_pii(traces_dir: Path) -> None:
    """End-to-end: OTLP trace with PII -> OpenInferenceTrace -> TraceLike ->
    LLMJudgeDetector. Assert the PII appears in the TraceLike (no redaction
    at the trace bridge) but is scrubbed in the prompt sent to the provider.
    """
    raw_otlp = json.loads((traces_dir / "simple_llm_call.json").read_text())
    otlp_with_pii = _inject_pii(raw_otlp)
    trace = from_otlp(otlp_with_pii)
    trace_like = trace.to_trace_like()

    assert "jane.doe@example.com" in trace_like.full_text
    assert "555-867-5309" in trace_like.full_text
    assert "123-45-6789" in trace_like.full_text
    assert "#4567890123" in trace_like.full_text

    provider = _RecordingProvider()
    detector = LLMJudgeDetector(provider)
    await detector.evaluate(_llm_judge_mode(), trace_like)

    assert len(provider.calls) == 1
    _system, user_prompt, _schema = provider.calls[0]

    assert "jane.doe@example.com" not in user_prompt
    assert "[REDACTED_EMAIL]" in user_prompt
    assert "555-867-5309" not in user_prompt
    assert "[REDACTED_PHONE]" in user_prompt
    assert "123-45-6789" not in user_prompt
    assert "[REDACTED_SSN]" in user_prompt
    assert "4567890123" not in user_prompt
    assert "[REDACTED_ACCOUNT]" in user_prompt


async def test_clusterer_to_drafter_path_inherits_redaction(tmp_path: Path) -> None:
    """End-to-end-ish: PII in classification evidence -> clusterer (redacts at
    the choke point) -> drafter. The embedding provider, the drafter LLM
    prompt, and the queued issue files all see only redacted text.
    """
    mode = Mode(id="leak", severity="medium", detection=Detection(type="regex", pattern="x"))
    rubric = Rubric(
        apiVersion="docket.dev/v1",
        kind="Rubric",
        metadata=RubricMetadata(name="redaction-test", version="0.0.1"),
        modes=[mode],
        clustering=Clustering(
            embedding_model="recording-embed",
            similarity_threshold=0.82,
            min_cluster_size=3,
        ),
    )
    raw_excerpt = "Refund jane.doe@example.com, call (555) 867-5309"
    classifications = [
        Classification(
            trace_id=f"t-{i}",
            rubric_version="redaction-test@0.0.1",
            mode_id="leak",
            positive=True,
            extra={"excerpt": raw_excerpt, "confidence": 0.9},
        )
        for i in range(3)
    ]
    embedding_provider = _RecordingEmbeddingProvider()

    clusters = await cluster_per_mode(
        classifications,
        rubric=rubric,
        embedding_provider=embedding_provider,
    )
    # The embeddings API never saw the raw PII.
    for call in embedding_provider.calls:
        for text in call:
            assert "jane.doe@example.com" not in text
            assert "867-5309" not in text
    assert len(clusters) == 1
    assert "[REDACTED_EMAIL]" in clusters[0].representative_excerpt
    assert "[REDACTED_PHONE]" in clusters[0].representative_excerpt

    llm_provider = _RecordingProvider(
        response={"title": "Agent leaks PII", "body": "Representative evidence shows a leak."}
    )
    drafts = await draft_issues(
        clusters,
        rubric=rubric,
        llm_provider=llm_provider,
        run_id="run-1",
        output_dir=tmp_path,
    )
    # The drafter prompt inherited the redacted excerpt.
    assert len(llm_provider.calls) == 1
    _system, drafter_prompt, _schema = llm_provider.calls[0]
    assert "jane.doe@example.com" not in drafter_prompt
    assert "867-5309" not in drafter_prompt
    assert "[REDACTED_EMAIL]" in drafter_prompt
    # And so did the queued issue files on disk.
    assert len(drafts) == 1
    cluster_id = clusters[0].cluster_id
    for queued_file in (tmp_path / f"{cluster_id}.json", tmp_path / f"{cluster_id}.md"):
        content = queued_file.read_text()
        assert "jane.doe@example.com" not in content
        assert "867-5309" not in content

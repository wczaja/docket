"""Zero-credential demo: in-memory trace backend + scripted providers.

`docket demo` runs the *real* triage pipeline — classify → cluster →
draft → report — over this module so a first run needs no API keys, no
Docker, and no instrumented app:

  - `DemoBackend` serves the 60-trace seeded-failure fixture from
    `docket._acceptance` (20 clean + 40 seeded across five `agents/v1`
    modes) entirely in memory, timestamps within the last hour.
  - `DemoJudgeProvider` stands in for the LLM: a deterministic, scripted
    judge that recognizes the fixture's seeded markers and templates the
    drafter's issue bodies. It is clearly labeled in output and is NOT a
    real model — `docket demo --live` swaps in a real provider.
  - `DemoEmbeddingProvider` embeds via hashed character trigrams so
    similar excerpts cluster without an embeddings API.

The deterministic detectors (`regex`, `tool_call`, `metric_threshold`)
run for real in both modes; only `llm_judge` modes and the drafter are
scripted until `--live`.

`ingest_to_phoenix` posts the same fixture to a real Phoenix OTLP
endpoint (`docket demo --to-phoenix`) so the second touch — docket
against a live backend — needs no instrumented app either.
"""

import math
import re
import zlib
from datetime import UTC, datetime
from typing import Any

import httpx

from docket._acceptance import (
    _HALLUCINATION_FALSEHOODS,
    TraceCase,
    acceptance_summary,
    build_acceptance_cases,
)
from docket.adapters.base import TraceBackend
from docket.errors import BackendError
from docket.llm.base import ModelProvider
from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Annotation
from docket.models.otlp import to_otlp
from docket.models.trace import OpenInferenceTrace

DEMO_BACKEND_ID = "demo"
DEMO_JUDGE_MODEL = "demo-scripted-judge"
DEMO_EMBEDDING_MODEL = "demo-hashed-trigrams"

__all__ = [
    "DEMO_BACKEND_ID",
    "DEMO_EMBEDDING_MODEL",
    "DEMO_JUDGE_MODEL",
    "DemoBackend",
    "DemoEmbeddingProvider",
    "DemoJudgeProvider",
    "build_demo_cases",
    "demo_summary",
    "ingest_to_phoenix",
]


def build_demo_cases() -> list[TraceCase]:
    """The demo trace set: `(label, expected_mode_ids, trace)` triples.

    Reuses the Phase 4/5 acceptance fixture so the demo exercises exactly
    the traces the acceptance tests gate on (20 clean + 40 seeded).
    """
    return build_acceptance_cases()


def demo_summary() -> dict[str, Any]:
    """Counts + seeded mode ids for banner/output purposes."""
    return acceptance_summary()


class DemoBackend(TraceBackend):
    """In-memory `TraceBackend` over the demo fixture.

    Supports the full contract the pipeline uses — listing, fetching,
    annotation upsert, and checkpoint sentinels — so `--annotate` and
    `--checkpoint` behave exactly as they would against a real backend
    (state just lives in this process).
    """

    def __init__(self, cases: list[TraceCase] | None = None) -> None:
        self._cases = cases if cases is not None else build_demo_cases()
        self._traces: dict[str, OpenInferenceTrace] = {
            trace.trace_id: trace for _, _, trace in self._cases
        }
        self.expected_modes: dict[str, list[str]] = {
            trace.trace_id: list(modes) for _, modes, trace in self._cases
        }
        self._annotations: dict[str, Annotation] = {}
        self._sentinels: dict[tuple[str, str], str] = {}

    @staticmethod
    def _trace_start(trace: OpenInferenceTrace) -> datetime:
        start_ns = min(s.start_time_unix_nano for s in trace.spans)
        return datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)

    async def list_traces(
        self,
        since: datetime,
        until: datetime | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[str]:
        del filter  # the demo fixture is small enough to never need filtering
        out = []
        for trace_id, trace in self._traces.items():
            if not trace.spans:
                continue
            start = self._trace_start(trace)
            if start >= since and (until is None or start <= until):
                out.append(trace_id)
        return sorted(out)

    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        trace = self._traces.get(trace_id)
        if trace is None:
            raise BackendError(f"demo backend has no trace {trace_id!r}")
        return trace

    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        if trace_id not in self._traces:
            raise BackendError(f"demo backend has no trace {trace_id!r}")
        self._annotations[annotation.idempotency_key()] = annotation

    @property
    def annotations(self) -> list[Annotation]:
        """Annotations written this process (for tests and demo output)."""
        return list(self._annotations.values())

    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        raise NotImplementedError(
            "The demo backend has no semantic search; run against a real "
            "backend (phoenix, langfuse, langsmith) for search_traces."
        )

    async def mark_trace_processed(
        self,
        trace_id: str,
        *,
        run_id: str,
        rubric_version: str,
    ) -> None:
        if trace_id not in self._traces:
            raise BackendError(f"demo backend has no trace {trace_id!r}")
        self._sentinels[(run_id, trace_id)] = rubric_version

    async def list_processed_trace_ids(
        self,
        *,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> set[str]:
        return {tid for (rid, tid) in self._sentinels if rid == run_id}


# --- scripted judge ---------------------------------------------------------

_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot help",
    "i can't help",
    "i cannot fulfill",
    "i can't do that",
    "i'm not able",
    "i won't help",
    "i can't comply",
    "sorry, i can't",
    "i'm unable",
)

# Dispatch markers: substrings of the builtin `agents/v1` judge prompts. A
# custom rubric's llm_judge modes won't match any of these and score negative
# in scripted mode (counted in `unknown_judge_calls`; the CLI warns).
_HALLUCINATION_PROMPT_MARKERS = ("factual claims", "not supported")
_PREMATURE_PROMPT_MARKERS = ("terminate", "gave up prematurely", "prematurely")
_HANDOFF_PROMPT_MARKERS = ("handoff",)

_FALSEHOOD_RESPONSES: tuple[str, ...] = tuple(response for _, response in _HALLUCINATION_FALSEHOODS)


class DemoJudgeProvider(ModelProvider):
    """Deterministic stand-in for the judge + drafter LLM calls.

    NOT a model: a scripted classifier that recognizes the demo fixture's
    seeded failure markers, plus a template renderer for issue drafts. It
    exists so `docket demo` runs free, offline, and reproducibly; pass
    `--live` for a real provider. Output everywhere labels it "scripted".
    """

    model = DEMO_JUDGE_MODEL

    def __init__(self) -> None:
        self.calls = 0
        self.unknown_judge_calls = 0

    async def structured_complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        properties = schema.get("properties", {})
        if "title" in properties and "body" in properties:
            return self._draft(user)
        if "verdicts" in properties:
            item_schema = properties["verdicts"].get("items", {})
            blocks = _split_batch_traces(user)
            return {"verdicts": [self._judge(user, block, item_schema) for block in blocks]}
        return self._judge(user, _extract_trace_text(user), schema)

    # -- judge ----------------------------------------------------------

    def _judge(self, full_prompt: str, trace_text: str, schema: dict[str, Any]) -> dict[str, Any]:
        instructions = _extract_instructions(full_prompt).lower()
        if any(m in instructions for m in _HALLUCINATION_PROMPT_MARKERS):
            return self._judge_hallucination(trace_text)
        if any(m in instructions for m in _PREMATURE_PROMPT_MARKERS):
            return self._judge_premature(trace_text)
        if any(m in instructions for m in _HANDOFF_PROMPT_MARKERS):
            return _negative("scripted judge: no failed handoff detected")
        self.unknown_judge_calls += 1
        return _negative(
            "scripted demo judge does not evaluate custom llm_judge modes; "
            "pass --live for a real model"
        )

    @staticmethod
    def _judge_hallucination(trace_text: str) -> dict[str, Any]:
        for falsehood in _FALSEHOOD_RESPONSES:
            if falsehood in trace_text:
                return {
                    "positive": True,
                    "excerpt": falsehood,
                    "confidence": 0.97,
                    "reason": "response asserts a fact contradicted by common knowledge",
                }
        return _negative("no unsupported factual claim recognized")

    @staticmethod
    def _judge_premature(trace_text: str) -> dict[str, Any]:
        final = _final_assistant_line(trace_text)
        refused = any(marker in final.lower() for marker in _REFUSAL_MARKERS)
        used_tools = "[tool:" in trace_text
        # < 60 chars: the response is ONLY the refusal. Longer refusals carry
        # payload (e.g. a leaked system prompt) and belong to other modes.
        if refused and not used_tools and len(final) < 60:
            return {
                "positive": True,
                "excerpt": final,
                "confidence": 0.92,
                "reason": "bare refusal of an answerable request, no tool use attempted",
            }
        return _negative("agent made a reasonable attempt at the task")

    # -- drafter ----------------------------------------------------------

    def _draft(self, user: str) -> dict[str, Any]:
        mode_id = _search(r"Failure mode: (\S+)", user) or "unknown-mode"
        mode_name = _search(r"Mode name: (.+)", user) or mode_id
        severity = _search(r"Severity: (\w+)", user) or "unknown"
        size = _search(r"Cluster size: (\d+)", user) or "?"
        evidence = _search(r"Representative evidence:\n---\n(.*?)\n---", user, flags=re.DOTALL)
        title = _DRAFT_TITLES.get(mode_id, f"{mode_name}: {size} similar failures").format(n=size)
        next_step = _DRAFT_NEXT_STEPS.get(
            mode_id,
            f"Review the member traces in the provenance block and decide whether "
            f"`{mode_id}` needs a guardrail or a rubric threshold change.",
        )
        body_parts = [
            f"{size} traces in this window hit the `{mode_id}` failure mode "
            f"(severity: {severity}). The cluster groups semantically similar "
            "occurrences, so this is one recurring behavior, not isolated noise.",
        ]
        if evidence:
            body_parts += ["", "**Representative evidence:**", "", f"> {evidence.strip()}"]
        body_parts += [
            "",
            "**Suggested next step:** " + next_step,
            "",
            "_Drafted by the scripted demo judge (deterministic). Run "
            "`docket demo --live` for real-model drafting._",
        ]
        return {"title": title[:120], "body": "\n".join(body_parts)}


_DRAFT_TITLES: dict[str, str] = {
    "refusal-leakage": "Agent pastes its system prompt into refusal responses ({n} traces)",
    "unsafe-tool-call": "Destructive tool calls executed without user confirmation ({n} traces)",
    "infinite-loop": "Agent loops past the span budget without terminating ({n} traces)",
    "premature-termination": "Agent abandons answerable requests with a bare refusal ({n} traces)",
    "hallucination": "Agent asserts facts contradicted by its own context ({n} traces)",
}

_DRAFT_NEXT_STEPS: dict[str, str] = {
    "refusal-leakage": (
        "Add an output filter (or system-prompt instruction) that strips internal "
        "instructions from refusal text, then spot-check the member traces."
    ),
    "unsafe-tool-call": (
        "Gate the destructive tools behind an explicit user-confirmation step in the "
        "agent loop; audit the member traces for actual data impact."
    ),
    "infinite-loop": (
        "Add a hard iteration cap and a loop detector to the agent runtime; the member "
        "traces show where the planner fails to converge."
    ),
    "premature-termination": (
        "Inspect the refused requests for a mis-tuned safety filter or missing tool "
        "wiring; these were answerable tasks."
    ),
    "hallucination": (
        "Require citation of retrieved context for factual claims, or add a "
        "verification tool call before the final response."
    ),
}


def _negative(reason: str) -> dict[str, Any]:
    return {"positive": False, "excerpt": None, "confidence": 0.9, "reason": reason}


def _search(pattern: str, text: str, *, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def _extract_instructions(prompt: str) -> str:
    """The `Instructions:` block the LLMJudgeDetector prepends to every call."""
    m = re.search(
        r"Instructions:\n(.*?)(?:\n\n(?:Context:|Trace:|=== Trace)|\Z)", prompt, re.DOTALL
    )
    return m.group(1) if m else prompt


def _extract_trace_text(prompt: str) -> str:
    idx = prompt.rfind("Trace:\n")
    return prompt[idx + len("Trace:\n") :] if idx != -1 else prompt


def _split_batch_traces(prompt: str) -> list[str]:
    """Split the batched-judge prompt into per-trace text blocks."""
    parts = re.split(r"=== Trace \d+ ===\n", prompt)
    if len(parts) <= 1:
        return [_extract_trace_text(prompt)]
    # parts[0] is the Instructions block; the trailing "Return an object..."
    # sentence rides along with the last trace and matches no markers.
    return parts[1:]


def _final_assistant_line(trace_text: str) -> str:
    final = ""
    for line in trace_text.splitlines():
        if line.startswith("[assistant] "):
            final = line[len("[assistant] ") :]
    return final


# --- scripted embeddings ------------------------------------------------------


class DemoEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings from hashed word unigrams.

    Free and offline. Excerpts that share most of their words (the
    fixture's seeded variants) land within the rubric's similarity
    threshold and cluster; unrelated excerpts don't. Not a semantic
    model — pass a real `--embedding` URI when judging real traces.
    """

    model = DEMO_EMBEDDING_MODEL

    _DIM = 512

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9_']+", text.lower()) or ["<empty>"]
        v = [0.0] * cls._DIM
        for token in tokens:
            v[zlib.crc32(token.encode("utf-8")) % cls._DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


# --- Phoenix seeding ----------------------------------------------------------


async def ingest_to_phoenix(
    phoenix_url: str,
    cases: list[TraceCase] | None = None,
) -> tuple[int, list[str]]:
    """POST the demo fixture to a Phoenix OTLP HTTP endpoint.

    Returns `(ingested_count, failure_messages)`. Mirrors
    `scripts/ingest_acceptance_traces.py`, packaged so `docket demo
    --to-phoenix` works from a wheel install.
    """
    resolved = cases if cases is not None else build_demo_cases()
    failures: list[str] = []
    async with httpx.AsyncClient(base_url=phoenix_url, timeout=30.0) as client:
        for label, _modes, trace in resolved:
            try:
                response = await client.post(
                    "/v1/traces",
                    json=to_otlp(trace),
                    headers={"content-type": "application/json"},
                )
            except httpx.HTTPError as e:
                failures.append(f"{label}: {e}")
                continue
            if response.status_code >= 400:
                failures.append(f"{label}: HTTP {response.status_code} {response.text[:120]}")
    return len(resolved) - len(failures), failures

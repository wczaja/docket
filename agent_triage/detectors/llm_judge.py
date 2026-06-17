"""LLM-judge detector.

Constructs a single (or batched) structured-output completion against the
configured provider. The rubric's `output_schema` controls the required shape;
providers request it via their native mechanism, and this detector enforces
conformance by validating every response against the schema (JSON Schema
draft 2020-12). A malformed response raises `DetectionError` — it is never
silently scored negative. PII is redacted from trace text before the
completion runs (design §4).

Batching: `evaluate_batch` groups up to `batch_size` traces per provider call.
For a batch of N, the schema is wrapped as `{verdicts: [item_schema] * N}` and
the prompt blocks the traces with explicit numbering so the model emits one
verdict per trace in order.
"""

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from agent_triage.detectors.base import Detector
from agent_triage.errors import DetectionError
from agent_triage.llm import build_provider
from agent_triage.llm.base import ModelProvider
from agent_triage.models.trace import TraceLike, Verdict
from agent_triage.observability import redact
from agent_triage.rubric.spec import Mode

_SYSTEM_PROMPT = (
    "You are an LLM agent failure-mode classifier. Given a trace and a failure "
    "mode definition, decide whether the failure mode is present. Return strictly "
    "the JSON object described by the schema. The `positive` field is true when "
    "the failure mode is present in the trace."
)


class LLMJudgeDetector(Detector):
    def __init__(self, default_provider: ModelProvider, batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
        self._default = default_provider
        self._batch_size = batch_size
        self._override_providers: dict[str, ModelProvider] = {}

    async def evaluate(self, mode: Mode, trace: TraceLike) -> Verdict:
        verdicts = await self._evaluate_chunk(mode, [trace])
        return verdicts[0]

    async def evaluate_batch(
        self,
        mode: Mode,
        traces: list[TraceLike],
    ) -> list[Verdict]:
        all_verdicts: list[Verdict] = []
        for start in range(0, len(traces), self._batch_size):
            chunk = traces[start : start + self._batch_size]
            all_verdicts.extend(await self._evaluate_chunk(mode, chunk))
        return all_verdicts

    def _resolve_provider(self, mode: Mode) -> ModelProvider:
        if mode.detection.model is None:
            return self._default
        # Cache per-mode override providers: each provider holds an SDK HTTP
        # client, and a run evaluates the same mode against every trace.
        cached = self._override_providers.get(mode.detection.model)
        if cached is None:
            cached = build_provider(mode.detection.model)
            self._override_providers[mode.detection.model] = cached
        return cached

    async def _evaluate_chunk(self, mode: Mode, traces: list[TraceLike]) -> list[Verdict]:
        d = mode.detection
        if d.prompt is None or d.output_schema is None:
            raise DetectionError(
                f"Mode {mode.id!r}: llm_judge requires `prompt` and `output_schema`"
            )
        provider = self._resolve_provider(mode)
        redacted = [redact(t.full_text) for t in traces]
        # Contexts also reach the external provider; redact them like the trace text.
        contexts = [redact(t.context) if t.context is not None else None for t in traces]
        if len(traces) == 1:
            user = self._build_single_prompt(d.prompt, redacted[0], contexts[0])
            result = await provider.structured_complete(
                system=_SYSTEM_PROMPT,
                user=user,
                schema=d.output_schema,
            )
            return [Verdict(positive=self._validated_positive(mode, result), extra=result)]
        batch_schema = self._wrap_schema_for_batch(d.output_schema, len(traces))
        user = self._build_batch_prompt(d.prompt, redacted, contexts)
        result = await provider.structured_complete(
            system=_SYSTEM_PROMPT,
            user=user,
            schema=batch_schema,
        )
        verdicts_data = result.get("verdicts")
        if not isinstance(verdicts_data, list):
            raise DetectionError(
                f"Mode {mode.id!r}: batched llm_judge did not return a `verdicts` array"
            )
        if len(verdicts_data) != len(traces):
            raise DetectionError(
                f"Mode {mode.id!r}: expected {len(traces)} verdicts, "
                f"provider returned {len(verdicts_data)}"
            )
        out: list[Verdict] = []
        for v in verdicts_data:
            if not isinstance(v, dict):
                raise DetectionError(f"Mode {mode.id!r}: batched verdict was not an object")
            out.append(Verdict(positive=self._validated_positive(mode, v), extra=v))
        return out

    @staticmethod
    def _validated_positive(mode: Mode, result: dict[str, Any]) -> bool:
        """Validate one judge response against the mode's `output_schema`.

        A response that violates the schema or lacks a boolean `positive`
        raises `DetectionError`, feeding the classifier's retry/`unprocessed`
        path. It is never defaulted to a negative verdict.
        """
        schema = mode.detection.output_schema
        if schema is None:  # unreachable after the _evaluate_chunk guard
            raise DetectionError(f"Mode {mode.id!r}: llm_judge requires `output_schema`")
        try:
            jsonschema_validate(result, schema, cls=Draft202012Validator)
        except JSONSchemaValidationError as e:
            raise DetectionError(
                f"Mode {mode.id!r}: judge response violates `output_schema`: {e.message}"
            ) from e
        except SchemaError as e:
            raise DetectionError(
                f"Mode {mode.id!r}: `output_schema` is not a valid JSON Schema: {e.message}"
            ) from e
        positive = result.get("positive")
        if not isinstance(positive, bool):
            raise DetectionError(
                f"Mode {mode.id!r}: judge response is missing a boolean `positive` field"
            )
        return positive

    @staticmethod
    def _build_single_prompt(instruction: str, trace_text: str, context: str | None) -> str:
        parts: list[str] = [f"Instructions:\n{instruction}"]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Trace:\n{trace_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_batch_prompt(
        instruction: str,
        trace_texts: list[str],
        contexts: list[str | None],
    ) -> str:
        parts: list[str] = [f"Instructions:\n{instruction}"]
        for i, (text, ctx) in enumerate(zip(trace_texts, contexts, strict=True)):
            block = f"=== Trace {i + 1} ==="
            if ctx:
                block += f"\nContext: {ctx}"
            block += f"\n{text}"
            parts.append(block)
        parts.append(
            "Return an object with a `verdicts` array containing exactly one "
            "entry per trace, in the same order."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _wrap_schema_for_batch(item_schema: dict[str, Any], n: int) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["verdicts"],
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": item_schema,
                    "minItems": n,
                    "maxItems": n,
                }
            },
        }

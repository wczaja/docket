"""Self-instrumentation: emit OpenInference traces for the triage agent's own runs.

Per design §4.5:

  The triage agent's own runs MUST emit OpenInference traces to a configurable
  backend (`instrumentation_backend` in `docket.yaml`; default: Phoenix,
  on the assumption that an OSS-first user is already running Phoenix locally).

Phase 5 wires this via `arize-phoenix-otel` (the canonical OTLP exporter for
Phoenix) plus `openinference-instrumentation-{anthropic,openai}` so the LLM
provider calls inside detectors and the drafter automatically generate spans.

`configure_instrumentation` is opt-in — the CLI passes `--instrument-to URL`
or the config sets `instrumentation_backend`. Default is off, so unit tests
and CI runs don't try to reach a Phoenix that isn't there.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from docket.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_INSTRUMENTATION_ENDPOINT = "http://localhost:6006"
DEFAULT_PROJECT_NAME = "docket"


@contextmanager
def configure_instrumentation(
    *,
    endpoint: str = DEFAULT_INSTRUMENTATION_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    instrument_anthropic: bool = True,
    instrument_openai: bool = True,
) -> "Iterator[Any]":
    """Wire OpenInference instrumentation for the duration of a `with` block.

    The instrumentation imports are lazy so a CLI run without
    `--instrument-to` doesn't pull in the heavy transitive deps. When the
    operator explicitly asked for instrumentation but the packages are
    missing or broken, that's raised as a ConfigError — silently running
    uninstrumented would hide the misconfiguration. On enter, returns the
    tracer provider so callers can attach custom spans if they want.
    """
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError as e:
        raise ConfigError(
            f"Instrumentation was requested but its dependencies failed to import: {e}. "
            "Reinstall docket-runtime with the openinference-instrumentation-* and "
            "arize-phoenix-otel packages available."
        ) from e

    tracer_provider = register(
        endpoint=endpoint,
        project_name=project_name,
    )
    anthropic_instrumentor: Any | None = None
    openai_instrumentor: Any | None = None
    if instrument_anthropic:
        anthropic_instrumentor = AnthropicInstrumentor()
        anthropic_instrumentor.instrument(tracer_provider=tracer_provider)
    if instrument_openai:
        openai_instrumentor = OpenAIInstrumentor()
        openai_instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        yield tracer_provider
    finally:
        if anthropic_instrumentor is not None:
            anthropic_instrumentor.uninstrument()
        if openai_instrumentor is not None:
            openai_instrumentor.uninstrument()

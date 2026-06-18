"""Tests for the instrumentation context manager.

The instrumentation pulls in `phoenix.otel.register` which expects to send
OTLP to a real endpoint. We mock the imports so the test runs without a
Phoenix server.
"""

from unittest.mock import MagicMock, patch

import pytest

from docket.observability.instrumentation import (
    DEFAULT_INSTRUMENTATION_ENDPOINT,
    DEFAULT_PROJECT_NAME,
)


def test_default_constants() -> None:
    assert DEFAULT_INSTRUMENTATION_ENDPOINT.startswith("http")
    assert DEFAULT_PROJECT_NAME == "docket"


def test_configure_instrumentation_instruments_and_uninstruments() -> None:
    from docket.observability.instrumentation import configure_instrumentation

    fake_provider = MagicMock(name="tracer_provider")
    fake_anthropic = MagicMock(name="AnthropicInstrumentor")
    fake_openai = MagicMock(name="OpenAIInstrumentor")

    with (
        patch(
            "phoenix.otel.register",
            return_value=fake_provider,
        ) as register_mock,
        patch(
            "openinference.instrumentation.anthropic.AnthropicInstrumentor",
            return_value=fake_anthropic,
        ),
        patch(
            "openinference.instrumentation.openai.OpenAIInstrumentor",
            return_value=fake_openai,
        ),
    ):
        with configure_instrumentation(endpoint="http://test", project_name="test-proj"):
            register_mock.assert_called_once()
            fake_anthropic.instrument.assert_called_once()
            fake_openai.instrument.assert_called_once()
        fake_anthropic.uninstrument.assert_called_once()
        fake_openai.uninstrument.assert_called_once()


def test_configure_instrumentation_can_disable_per_provider() -> None:
    from docket.observability.instrumentation import configure_instrumentation

    fake_provider = MagicMock(name="tracer_provider")
    fake_anthropic = MagicMock(name="AnthropicInstrumentor")
    fake_openai = MagicMock(name="OpenAIInstrumentor")

    with (
        patch("phoenix.otel.register", return_value=fake_provider),
        patch(
            "openinference.instrumentation.anthropic.AnthropicInstrumentor",
            return_value=fake_anthropic,
        ),
        patch(
            "openinference.instrumentation.openai.OpenAIInstrumentor",
            return_value=fake_openai,
        ),
        configure_instrumentation(
            endpoint="http://test",
            instrument_anthropic=True,
            instrument_openai=False,
        ),
    ):
        fake_anthropic.instrument.assert_called_once()
        fake_openai.instrument.assert_not_called()


def test_configure_instrumentation_cleans_up_on_exception() -> None:
    """If the with-block raises, instrumentors still uninstrument on the way out."""
    from docket.observability.instrumentation import configure_instrumentation

    fake_provider = MagicMock(name="tracer_provider")
    fake_anthropic = MagicMock(name="AnthropicInstrumentor")
    fake_openai = MagicMock(name="OpenAIInstrumentor")

    with (
        patch("phoenix.otel.register", return_value=fake_provider),
        patch(
            "openinference.instrumentation.anthropic.AnthropicInstrumentor",
            return_value=fake_anthropic,
        ),
        patch(
            "openinference.instrumentation.openai.OpenAIInstrumentor",
            return_value=fake_openai,
        ),
        pytest.raises(RuntimeError, match="boom"),
        configure_instrumentation(endpoint="http://test"),
    ):
        raise RuntimeError("boom")
    fake_anthropic.uninstrument.assert_called_once()
    fake_openai.uninstrument.assert_called_once()


def test_configure_instrumentation_import_failure_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken instrumentation install surfaces as ConfigError, not ImportError."""
    import sys

    from docket.errors import ConfigError
    from docket.observability.instrumentation import configure_instrumentation

    # Setting a sys.modules entry to None makes its import raise ImportError.
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    with (
        pytest.raises(ConfigError, match="dependencies failed to import"),
        configure_instrumentation(endpoint="http://test"),
    ):
        pass  # pragma: no cover

"""Typed exception hierarchy. All raised exceptions inherit from AgentTriageError."""


class AgentTriageError(Exception):
    """Base class for all agent-triage errors. Never raise directly; raise a subclass."""


class ConfigError(AgentTriageError):
    """Failed to load or validate agent-triage.yaml."""


class CredentialError(AgentTriageError):
    """A required credential (API key, token) is missing or invalid."""


class RubricError(AgentTriageError):
    """Base class for rubric-related errors."""


class RubricValidationError(RubricError):
    """A rubric failed schema or semantic validation."""


class RubricImportError(RubricError):
    """A rubric's `imports:` could not be resolved."""


class DetectionError(AgentTriageError):
    """A detector failed to evaluate a trace."""


class BackendError(AgentTriageError):
    """A trace backend adapter failed."""


class TrackerError(AgentTriageError):
    """A tracker adapter failed."""


class BudgetExceededError(AgentTriageError):
    """A run exceeded the configured `max_traces_per_run` cap."""

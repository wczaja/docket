"""Typed exception hierarchy. All raised exceptions inherit from DocketError."""


class DocketError(Exception):
    """Base class for all docket errors. Never raise directly; raise a subclass."""


class ConfigError(DocketError):
    """Failed to load or validate docket.yaml."""


class CredentialError(DocketError):
    """A required credential (API key, token) is missing or invalid."""


class RubricError(DocketError):
    """Base class for rubric-related errors."""


class RubricValidationError(RubricError):
    """A rubric failed schema or semantic validation."""


class RubricImportError(RubricError):
    """A rubric's `imports:` could not be resolved."""


class DetectionError(DocketError):
    """A detector failed to evaluate a trace."""


class BackendError(DocketError):
    """A trace backend adapter failed."""


class TrackerError(DocketError):
    """A tracker adapter failed."""


class BudgetExceededError(DocketError):
    """A run exceeded the configured `max_traces_per_run` cap."""

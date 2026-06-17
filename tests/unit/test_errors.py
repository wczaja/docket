from agent_triage.errors import (
    AgentTriageError,
    BackendError,
    BudgetExceededError,
    ConfigError,
    CredentialError,
    DetectionError,
    RubricError,
    RubricImportError,
    RubricValidationError,
    TrackerError,
)


def test_all_inherit_from_base() -> None:
    for cls in (
        ConfigError,
        CredentialError,
        RubricError,
        RubricValidationError,
        RubricImportError,
        DetectionError,
        BackendError,
        TrackerError,
        BudgetExceededError,
    ):
        assert issubclass(cls, AgentTriageError)


def test_rubric_validation_is_a_rubric_error() -> None:
    assert issubclass(RubricValidationError, RubricError)
    assert issubclass(RubricImportError, RubricError)

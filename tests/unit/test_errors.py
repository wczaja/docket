from docket.errors import (
    BackendError,
    BudgetExceededError,
    ConfigError,
    CredentialError,
    DetectionError,
    DocketError,
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
        assert issubclass(cls, DocketError)


def test_rubric_validation_is_a_rubric_error() -> None:
    assert issubclass(RubricValidationError, RubricError)
    assert issubclass(RubricImportError, RubricError)

"""Rubric DSL: Pydantic spec, YAML loader with import resolution,
JSON Schema validator, and built-in rubric registry."""

from docket.rubric.loader import load_rubric
from docket.rubric.registry import (
    BUILTIN_URI_PREFIX,
    is_builtin_uri,
    list_builtins,
    resolve_builtin,
)
from docket.rubric.spec import Mode, Rubric
from docket.rubric.validator import validate_rubric_yaml

__all__ = [
    "BUILTIN_URI_PREFIX",
    "Mode",
    "Rubric",
    "is_builtin_uri",
    "list_builtins",
    "load_rubric",
    "resolve_builtin",
    "validate_rubric_yaml",
]

"""Rubric DSL: Pydantic spec, YAML loader with import resolution,
JSON Schema validator, and built-in rubric registry."""

from agent_triage.rubric.loader import load_rubric
from agent_triage.rubric.registry import (
    BUILTIN_URI_PREFIX,
    is_builtin_uri,
    list_builtins,
    resolve_builtin,
)
from agent_triage.rubric.spec import Mode, Rubric
from agent_triage.rubric.validator import validate_rubric_yaml

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

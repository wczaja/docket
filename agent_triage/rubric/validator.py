"""JSON Schema validation for rubric files.

Pydantic gives structural validation matched to Python types; JSON Schema gives
DSL-level validation as a portable, language-agnostic check. We run both so
contributors writing rubrics get the same errors regardless of tooling.

Single-file: imports are not walked here; the recursive loader catches import
issues during its merge pass. This keeps JSON Schema concerns scoped to the
DSL surface.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaError

from agent_triage.errors import RubricValidationError
from agent_triage.rubric._sources import read_source

SUPPORTED_API_VERSIONS = frozenset({"agent-triage.dev/v1"})


def validate_rubric_yaml(source: Path | str) -> None:
    """Validate a single rubric source against the JSON Schema for its apiVersion.

    `source` may be a `Path`, a plain path string, a `file://` URI, or an
    `agent-triage.dev/builtin/<name>/<version>` URI. Imports are not walked.
    """
    text, origin, _canonical = read_source(source)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RubricValidationError(f"Failed to parse YAML at {origin}: {e}") from e

    if not isinstance(data, dict):
        raise RubricValidationError(f"Rubric at {origin} must be a YAML mapping")

    api_version = data.get("apiVersion")
    if api_version not in SUPPORTED_API_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_API_VERSIONS))
        raise RubricValidationError(
            f"Unsupported apiVersion {api_version!r} at {origin}; "
            f"this runtime supports: {supported}"
        )

    schema = _load_schema_v1()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: tuple(str(p) for p in e.path))
    if errors:
        messages = "\n".join(f"  - {_format_error(e)}" for e in errors)
        raise RubricValidationError(
            f"Rubric at {origin} failed JSON Schema validation:\n{messages}"
        )


def _load_schema_v1() -> dict[str, Any]:
    schema_text = files("agent_triage.rubric.schemas").joinpath("v1.json").read_text()
    parsed = json.loads(schema_text)
    if not isinstance(parsed, dict):
        raise RubricValidationError("v1.json schema is not a JSON object")
    return parsed


def _format_error(error: JSONSchemaError) -> str:
    path = ".".join(str(p) for p in error.path) or "<root>"
    return f"{path}: {error.message}"

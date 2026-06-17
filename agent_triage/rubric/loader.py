"""YAML loader for rubric files with recursive import resolution.

Supports the two import schemes defined for v1.0 (design §3.2):
  - `file://<path>` (absolute, or relative to the importing rubric file)
  - `agent-triage.dev/builtin/<name>/<version>` (resolved against packaged data)

Merge semantics: imports are walked left-to-right; later imports override
earlier; the importing rubric's own modes override imports. Cycle detection
uses the canonical URIs of in-progress sources.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

from agent_triage.errors import RubricError, RubricImportError, RubricValidationError
from agent_triage.rubric._sources import read_source
from agent_triage.rubric.spec import Detection, Mode, Rubric

SUPPORTED_API_VERSIONS = frozenset({"agent-triage.dev/v1"})


def load_rubric(source: Path | str) -> Rubric:
    """Load a rubric and recursively resolve and merge its imports.

    `source` may be a `Path`, a plain path string, a `file://` URI, or an
    `agent-triage.dev/builtin/<name>/<version>` URI.

    After import resolution and merging, the merged mode set is validated
    against the design §3.5 semantic rules (see `_validate_modes_semantic`),
    so broken modes fail here rather than per-trace at classification time.
    """
    rubric, mode_origins = _load_recursive(source, base_uri=None, in_progress=())
    _validate_modes_semantic(rubric, mode_origins)
    return rubric


def _load_recursive(
    source: Path | str,
    base_uri: str | None,
    in_progress: tuple[str, ...],
) -> tuple[Rubric, dict[str, str]]:
    text, origin, canonical = read_source(source, base_uri)
    if canonical in in_progress:
        cycle = " -> ".join((*in_progress, canonical))
        raise RubricImportError(f"Import cycle detected: {cycle}")
    data = _parse_yaml(text, origin)
    raw_imports = data.pop("imports", None) or []
    if not isinstance(raw_imports, list) or not all(isinstance(s, str) for s in raw_imports):
        raise RubricValidationError(
            f"Rubric {origin!r} has invalid `imports`: expected a list of URI strings"
        )
    own = _parse_rubric(data, origin)
    if own.apiVersion not in SUPPORTED_API_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_API_VERSIONS))
        raise RubricValidationError(
            f"Rubric {origin!r} uses unsupported apiVersion {own.apiVersion!r}. "
            f"This runtime supports: {supported}"
        )
    new_in_progress = (*in_progress, canonical)
    imported: list[Rubric] = []
    mode_origins: dict[str, str] = {}
    for imp_uri in raw_imports:
        sub, sub_origins = _load_recursive(imp_uri, base_uri=canonical, in_progress=new_in_progress)
        imported.append(sub)
        mode_origins.update(sub_origins)
    mode_origins.update({mode.id: origin for mode in own.modes})
    merged_modes = _merge_modes([r.modes for r in imported] + [own.modes])
    final = own.model_copy(update={"modes": merged_modes, "imports": raw_imports})
    _validate_post_merge(final, origin, had_imports=bool(raw_imports))
    return final, mode_origins


def _parse_yaml(text: str, origin: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RubricError(f"Failed to parse YAML at {origin}: {e}") from e
    if not isinstance(data, dict):
        raise RubricError(f"Rubric at {origin} must be a YAML mapping at the top level")
    return data


def _parse_rubric(data: dict[str, Any], origin: str) -> Rubric:
    try:
        return Rubric.model_validate(data)
    except ValidationError as e:
        raise RubricValidationError(f"Rubric at {origin} failed validation:\n{e}") from e


def _merge_modes(mode_lists: list[list[Mode]]) -> list[Mode]:
    merged: dict[str, Mode] = {}
    for modes in mode_lists:
        for mode in modes:
            merged[mode.id] = mode
    return list(merged.values())


def _validate_modes_semantic(rubric: Rubric, mode_origins: dict[str, str]) -> None:
    """Enforce the design §3.5 rules on the merged mode set.

    Every `llm_judge` detection (including composite operands) must have a
    `prompt` and an `output_schema`; the `output_schema` must be a valid JSON
    Schema (draft 2020-12) with `type: object` at the root that requires a
    boolean `positive` property.
    """
    for mode in rubric.modes:
        origin = mode_origins.get(mode.id)
        where = f"Mode {mode.id!r}" + (f" (defined in {origin})" if origin else "")
        for detection in _walk_detections(mode.detection):
            if detection.type != "llm_judge":
                continue
            if not detection.prompt or detection.output_schema is None:
                raise RubricValidationError(
                    f"{where}: llm_judge detection requires `prompt` and "
                    f"`output_schema` (design §3.5)"
                )
            _check_output_schema(detection.output_schema, where)


def _walk_detections(detection: Detection) -> Iterator[Detection]:
    yield detection
    for operand in detection.operands or []:
        yield from _walk_detections(operand)


def _check_output_schema(schema: dict[str, Any], where: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise RubricValidationError(
            f"{where}: `output_schema` is not a valid JSON Schema (draft 2020-12): {e.message}"
        ) from e
    if schema.get("type") != "object":
        raise RubricValidationError(
            f"{where}: `output_schema` must declare `type: object` at the root (design §3.5)"
        )
    required = schema.get("required")
    properties = schema.get("properties")
    positive = properties.get("positive") if isinstance(properties, dict) else None
    if (
        not isinstance(required, list)
        or "positive" not in required
        or not isinstance(positive, dict)
        or positive.get("type") != "boolean"
    ):
        raise RubricValidationError(
            f"{where}: `output_schema` must require a boolean `positive` property (design §3.5)"
        )


def _validate_post_merge(rubric: Rubric, origin: str, *, had_imports: bool) -> None:
    if rubric.modes:
        return
    if had_imports:
        raise RubricValidationError(
            f"Rubric {origin!r} has imports but zero modes after merging. "
            f"Imports may have resolved to empty rubrics."
        )
    raise RubricValidationError(
        f"Rubric {origin!r} has no modes and no imports. "
        f"A rubric must contain at least one mode after merging."
    )

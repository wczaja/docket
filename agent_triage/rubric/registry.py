"""Built-in rubric registry.

Built-in rubrics live inside `agent_triage.rubric.builtin` and are addressed by
the URI scheme `agent-triage.dev/builtin/<name>/<version>`. They are resolved
via `importlib.resources` so they work uniformly from editable installs and
wheels.
"""

from importlib.resources import files
from importlib.resources.abc import Traversable

from agent_triage.errors import RubricImportError

BUILTIN_URI_PREFIX = "agent-triage.dev/builtin/"
_BUILTIN_PACKAGE = "agent_triage.rubric.builtin"


def is_builtin_uri(uri: str) -> bool:
    return uri.startswith(BUILTIN_URI_PREFIX)


def resolve_builtin(uri: str) -> Traversable:
    """Resolve a builtin URI to a Traversable for its rubric.yaml.

    Raises `RubricImportError` if the URI is malformed or the rubric is unknown.
    """
    if not is_builtin_uri(uri):
        raise RubricImportError(
            f"Not a builtin URI: {uri!r} (expected prefix {BUILTIN_URI_PREFIX!r})"
        )
    suffix = uri[len(BUILTIN_URI_PREFIX) :]
    parts = suffix.split("/")
    if len(parts) != 2 or not all(parts):
        raise RubricImportError(
            f"Builtin URI must be of the form {BUILTIN_URI_PREFIX!r}<name>/<version>: {uri!r}"
        )
    name, version = parts
    root = files(_BUILTIN_PACKAGE)
    target = root / name / version / "rubric.yaml"
    if not target.is_file():
        available = sorted(list_builtins())
        raise RubricImportError(
            f"Unknown builtin rubric {uri!r}. "
            f"Available: {available if available else '(none registered)'}"
        )
    return target


def list_builtins() -> list[str]:
    """Return the URIs of all packaged builtin rubrics."""
    root = files(_BUILTIN_PACKAGE)
    builtins: list[str] = []
    for name_entry in root.iterdir():
        if not name_entry.is_dir() or name_entry.name.startswith("_"):
            continue
        name = name_entry.name
        for version_entry in name_entry.iterdir():
            if not version_entry.is_dir():
                continue
            version = version_entry.name
            if (version_entry / "rubric.yaml").is_file():
                builtins.append(f"{BUILTIN_URI_PREFIX}{name}/{version}")
    return builtins

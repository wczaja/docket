"""Source reading shared by loader and validator.

Resolves `file://`, plain paths, and `docket.dev/builtin/...` URIs into
readable text. Used internally by both the recursive loader and the JSON Schema
validator so URI semantics are consistent across them.
"""

from pathlib import Path

from docket.errors import RubricImportError
from docket.rubric.registry import is_builtin_uri, resolve_builtin

UNSUPPORTED_SCHEMES = ("https://", "http://", "registry://")
FILE_URI_PREFIX = "file://"


def read_source(
    source: Path | str,
    base_uri: str | None = None,
) -> tuple[str, str, str]:
    """Read text from `source`.

    Returns `(text, origin_for_errors, canonical_uri)`. `base_uri` supplies the
    directory context for resolving relative file paths in imports.
    """
    if isinstance(source, str) and is_builtin_uri(source):
        target = resolve_builtin(source)
        return target.read_text(), source, source

    if isinstance(source, str):
        if any(source.startswith(s) for s in UNSUPPORTED_SCHEMES):
            raise RubricImportError(
                f"Import scheme not supported in v1.0: {source!r}. "
                f"https:// and registry:// are reserved for v1.1+."
            )
        if source.startswith(FILE_URI_PREFIX):
            path = _resolve_file_uri(source, base_uri)
        else:
            path = _resolve_relative_path(Path(source), base_uri)
    else:
        path = source

    try:
        text = path.read_text()
    except FileNotFoundError as e:
        raise RubricImportError(f"Rubric file not found: {path}") from e
    canonical = f"{FILE_URI_PREFIX}{path.resolve()}"
    return text, str(path), canonical


def _resolve_file_uri(uri: str, base_uri: str | None) -> Path:
    path = Path(uri[len(FILE_URI_PREFIX) :])
    return _resolve_relative_path(path, base_uri)


def _resolve_relative_path(path: Path, base_uri: str | None) -> Path:
    if path.is_absolute():
        return path
    if base_uri is None:
        return path
    if not base_uri.startswith(FILE_URI_PREFIX):
        raise RubricImportError(
            f"Cannot resolve relative path {str(path)!r} from non-file source {base_uri!r}"
        )
    base = Path(base_uri[len(FILE_URI_PREFIX) :])
    return base.parent / path

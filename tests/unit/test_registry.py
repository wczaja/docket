import pytest

from agent_triage.errors import RubricImportError
from agent_triage.rubric.registry import (
    BUILTIN_URI_PREFIX,
    is_builtin_uri,
    list_builtins,
    resolve_builtin,
)


def test_is_builtin_uri_recognizes_prefix() -> None:
    assert is_builtin_uri("agent-triage.dev/builtin/foo/v1")
    assert is_builtin_uri(BUILTIN_URI_PREFIX + "x/y")


def test_is_builtin_uri_rejects_others() -> None:
    assert not is_builtin_uri("file:///foo")
    assert not is_builtin_uri("/foo")
    assert not is_builtin_uri("./foo.yaml")
    assert not is_builtin_uri("https://example.com/x")


def test_resolve_builtin_agents_v1_exists() -> None:
    target = resolve_builtin("agent-triage.dev/builtin/agents/v1")
    assert target.is_file()
    text = target.read_text()
    assert "apiVersion: agent-triage.dev/v1" in text


def test_resolve_builtin_rejects_non_builtin_prefix() -> None:
    with pytest.raises(RubricImportError, match="Not a builtin URI"):
        resolve_builtin("file:///foo")


def test_resolve_builtin_rejects_missing_version() -> None:
    with pytest.raises(RubricImportError, match="must be of the form"):
        resolve_builtin("agent-triage.dev/builtin/agents")


def test_resolve_builtin_rejects_extra_segments() -> None:
    with pytest.raises(RubricImportError, match="must be of the form"):
        resolve_builtin("agent-triage.dev/builtin/agents/v1/extra")


def test_resolve_builtin_rejects_empty_parts() -> None:
    with pytest.raises(RubricImportError, match="must be of the form"):
        resolve_builtin("agent-triage.dev/builtin//v1")
    with pytest.raises(RubricImportError, match="must be of the form"):
        resolve_builtin("agent-triage.dev/builtin/agents/")


def test_resolve_builtin_unknown_rubric() -> None:
    with pytest.raises(RubricImportError, match="Unknown builtin"):
        resolve_builtin("agent-triage.dev/builtin/nonexistent/v1")


def test_list_builtins_contains_agents_v1() -> None:
    builtins = list_builtins()
    assert "agent-triage.dev/builtin/agents/v1" in builtins

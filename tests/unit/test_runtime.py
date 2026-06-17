"""Tests for `runtime.build_backend`.

Phase 4's `run_triage` orchestrator moved to `agent.triage.run_triage_pipeline`
in Phase 5; see `test_triage_pipeline.py` for those tests. What remains here
is the trace-backend factory used by the CLI to wire adapters from
config + flags.
"""

import pytest

from agent_triage.config import Config, MCPServerConfig
from agent_triage.errors import ConfigError, TrackerError
from agent_triage.runtime import build_backend, build_tracker


def test_build_backend_from_phoenix_url() -> None:
    backend = build_backend(
        backend_name="phoenix",
        config=None,
        phoenix_url="http://test:6006",
    )
    assert backend.__class__.__name__ == "PhoenixAdapter"


def test_build_backend_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://config:6006"},
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    backend = build_backend(backend_name=None, config=config)
    assert backend.__class__.__name__ == "PhoenixAdapter"


def test_build_backend_cli_url_overrides_config_env() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://config:6006"},
        ),
        rubric="x",
    )
    backend = build_backend(
        backend_name=None,
        config=config,
        phoenix_url="http://override:6006",
    )
    assert backend.__class__.__name__ == "PhoenixAdapter"
    # The override URL is what reaches the adapter.
    assert "override" in backend._base_url  # type: ignore[attr-defined]  # noqa: SLF001


def test_build_backend_requires_phoenix_url() -> None:
    with pytest.raises(ConfigError, match="PHOENIX_URL"):
        build_backend(backend_name="phoenix", config=None)


def test_build_backend_from_langfuse_args() -> None:
    backend = build_backend(
        backend_name="langfuse",
        config=None,
        langfuse_host="http://langfuse.test",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",  # noqa: S106
    )
    assert backend.__class__.__name__ == "LangfuseAdapter"


def test_build_backend_langfuse_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-langfuse",
            env={
                "LANGFUSE_HOST": "http://langfuse.test",
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
            },
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    backend = build_backend(backend_name=None, config=config)
    assert backend.__class__.__name__ == "LangfuseAdapter"


def test_build_backend_requires_langfuse_host() -> None:
    with pytest.raises(ConfigError, match="LANGFUSE_HOST"):
        build_backend(backend_name="langfuse", config=None)


def test_build_backend_from_langsmith_args() -> None:
    backend = build_backend(
        backend_name="langsmith",
        config=None,
        langsmith_api_key="ls-test",  # noqa: S106
    )
    assert backend.__class__.__name__ == "LangsmithAdapter"


def test_build_backend_langsmith_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-langsmith",
            env={
                "LANGSMITH_API_KEY": "ls-test",
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.test",
                "LANGSMITH_PROJECT": "agents-v1",
            },
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    backend = build_backend(backend_name=None, config=config)
    assert backend.__class__.__name__ == "LangsmithAdapter"


def test_build_backend_requires_langsmith_api_key() -> None:
    with pytest.raises(ConfigError, match="LANGSMITH_API_KEY"):
        build_backend(backend_name="langsmith", config=None)


def test_build_backend_unknown_name() -> None:
    with pytest.raises(ConfigError, match="Unknown trace backend"):
        build_backend(backend_name="datadog", config=None)


def test_build_backend_requires_some_name() -> None:
    with pytest.raises(ConfigError, match="No trace backend specified"):
        build_backend(backend_name=None, config=None)


# -- tracker factory --------------------------------------------------------


def test_build_tracker_returns_none_when_unconfigured() -> None:
    assert build_tracker(tracker_name=None, config=None) is None


def test_build_tracker_jira_from_cli_args() -> None:
    tracker = build_tracker(
        tracker_name="jira",
        config=None,
        jira_host="https://example.atlassian.net",
        jira_project="AGT",
        jira_email="bot@example.com",
        jira_api_token="cloud-token",  # noqa: S106
    )
    assert tracker is not None
    assert tracker.__class__.__name__ == "JiraAdapter"
    assert tracker.deployment == "cloud"  # type: ignore[attr-defined]


def test_build_tracker_jira_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://test:6006"},
        ),
        tracker=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-jira",
            env={
                "JIRA_HOST": "https://jira.internal.example.com",
                "JIRA_PROJECT": "AGT",
                "JIRA_PAT": "pat-token",
            },
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    tracker = build_tracker(tracker_name=None, config=config)
    assert tracker is not None
    assert tracker.deployment == "datacenter"  # type: ignore[attr-defined]


def test_build_tracker_cli_overrides_config_env() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://test:6006"},
        ),
        tracker=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-jira",
            env={
                "JIRA_HOST": "https://example.atlassian.net",
                "JIRA_PROJECT": "AGT",
                "JIRA_EMAIL": "old@example.com",
                "JIRA_API_TOKEN": "old-token",
            },
        ),
        rubric="x",
    )
    tracker = build_tracker(
        tracker_name=None,
        config=config,
        jira_email="new@example.com",
        jira_api_token="new-token",  # noqa: S106
    )
    assert tracker is not None
    assert tracker._email == "new@example.com"  # type: ignore[attr-defined]  # noqa: SLF001


def test_build_tracker_jira_requires_host_and_project() -> None:
    with pytest.raises(ConfigError, match="JIRA_HOST"):
        build_tracker(
            tracker_name="jira",
            config=None,
            jira_email="bot@example.com",
            jira_api_token="t",  # noqa: S106
        )


def test_build_tracker_jira_validates_credentials_at_construction() -> None:
    """Missing credentials surface as TrackerError from the adapter ctor."""
    with pytest.raises(TrackerError, match="email and api_token"):
        build_tracker(
            tracker_name="jira",
            config=None,
            jira_host="https://example.atlassian.net",
            jira_project="AGT",
        )


def test_build_tracker_jira_explicit_deployment_override() -> None:
    tracker = build_tracker(
        tracker_name="jira",
        config=None,
        jira_host="https://example.atlassian.net",  # hostname says cloud
        jira_project="AGT",
        jira_pat="pat",  # noqa: S106
        jira_deployment="datacenter",  # but we explicitly pick DC
    )
    assert tracker is not None
    assert tracker.deployment == "datacenter"  # type: ignore[attr-defined]


def test_build_tracker_rejects_bad_deployment_string() -> None:
    with pytest.raises(ConfigError, match="Unknown Jira deployment"):
        build_tracker(
            tracker_name="jira",
            config=None,
            jira_host="https://example.atlassian.net",
            jira_project="AGT",
            jira_email="bot@example.com",
            jira_api_token="t",  # noqa: S106
            jira_deployment="onprem",
        )


def test_build_tracker_unknown_name() -> None:
    with pytest.raises(ConfigError, match="Unknown tracker"):
        build_tracker(tracker_name="asana", config=None)


def test_build_tracker_linear_from_cli_args() -> None:
    tracker = build_tracker(
        tracker_name="linear",
        config=None,
        linear_api_key="ln-test",  # noqa: S106
        linear_team_id="team-uuid",
    )
    assert tracker is not None
    assert tracker.__class__.__name__ == "LinearAdapter"


def test_build_tracker_linear_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://test:6006"},
        ),
        tracker=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-linear",
            env={
                "LINEAR_API_KEY": "ln-key",
                "LINEAR_TEAM_ID": "team-uuid",
            },
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    tracker = build_tracker(tracker_name=None, config=config)
    assert tracker is not None
    assert tracker.__class__.__name__ == "LinearAdapter"


def test_build_tracker_linear_requires_api_key() -> None:
    with pytest.raises(ConfigError, match="LINEAR_API_KEY"):
        build_tracker(
            tracker_name="linear",
            config=None,
            linear_team_id="team-uuid",
        )


def test_build_tracker_linear_requires_team_id() -> None:
    with pytest.raises(ConfigError, match="LINEAR_TEAM_ID"):
        build_tracker(
            tracker_name="linear",
            config=None,
            linear_api_key="ln-test",  # noqa: S106
        )


def test_build_tracker_github_from_cli_args() -> None:
    tracker = build_tracker(
        tracker_name="github",
        config=None,
        github_token="ghp_test",  # noqa: S106
        github_owner="acme",
        github_repo="widgets",
    )
    assert tracker is not None
    assert tracker.__class__.__name__ == "GitHubAdapter"
    assert tracker.repo_path == "acme/widgets"  # type: ignore[attr-defined]


def test_build_tracker_github_from_config() -> None:
    config = Config(
        trace_backend=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-phoenix",
            env={"PHOENIX_URL": "http://test:6006"},
        ),
        tracker=MCPServerConfig(
            type="mcp",
            command="agent-triage-adapter-github",
            env={
                "GITHUB_TOKEN": "ghp_test",
                "GITHUB_OWNER": "acme",
                "GITHUB_REPO": "widgets",
            },
        ),
        rubric="agent-triage.dev/builtin/agents/v1",
    )
    tracker = build_tracker(tracker_name=None, config=config)
    assert tracker is not None
    assert tracker.__class__.__name__ == "GitHubAdapter"


def test_build_tracker_github_requires_token() -> None:
    with pytest.raises(ConfigError, match="GITHUB_TOKEN"):
        build_tracker(
            tracker_name="github",
            config=None,
            github_owner="acme",
            github_repo="widgets",
        )


def test_build_tracker_github_requires_owner_and_repo() -> None:
    with pytest.raises(ConfigError, match="GITHUB_OWNER"):
        build_tracker(
            tracker_name="github",
            config=None,
            github_token="t",  # noqa: S106
        )


def test_build_tracker_github_supports_enterprise_api_url() -> None:
    """GitHub Enterprise Server users override the API URL via flag or env."""
    tracker = build_tracker(
        tracker_name="github",
        config=None,
        github_token="t",  # noqa: S106
        github_owner="acme",
        github_repo="widgets",
        github_api_url="https://github.acme.internal/api/v3",
    )
    assert tracker is not None
    # The adapter records the base URL after stripping trailing slash.
    assert tracker._api_url == "https://github.acme.internal/api/v3"  # type: ignore[attr-defined]  # noqa: SLF001

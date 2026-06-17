"""Adapter factory and CLI-facing convenience.

Phase 4 owned the orchestration loop here; Phase 5 moved it into
`agent_triage.agent.triage.run_triage_pipeline`. What survives in this module
is the trace-backend factory used by the CLI (and by the integration tests).

Phase 6 added Langfuse; Phase 7 added LangSmith; Phase 8 added the tracker
factory for Jira (and the per-tracker `build_tracker` selector).
"""

from agent_triage.adapters.base import TraceBackend, Tracker
from agent_triage.adapters.trace.langfuse import LangfuseAdapter
from agent_triage.adapters.trace.langsmith import (
    DEFAULT_LANGSMITH_ENDPOINT,
    LangsmithAdapter,
)
from agent_triage.adapters.trace.phoenix import PhoenixAdapter
from agent_triage.adapters.tracker.github import DEFAULT_GITHUB_API, GitHubAdapter
from agent_triage.adapters.tracker.jira import Deployment, JiraAdapter
from agent_triage.adapters.tracker.linear import DEFAULT_LINEAR_ENDPOINT, LinearAdapter
from agent_triage.config import Config
from agent_triage.errors import ConfigError

_BACKEND_ALIASES: dict[str, str] = {
    "phoenix": "phoenix",
    "agent-triage-adapter-phoenix": "phoenix",
    "langfuse": "langfuse",
    "agent-triage-adapter-langfuse": "langfuse",
    "langsmith": "langsmith",
    "agent-triage-adapter-langsmith": "langsmith",
}


def resolve_backend_id(backend_name: str | None, config: Config | None) -> str:
    """Return the canonical backend identifier (`phoenix`, `langfuse`,
    `langsmith`) used in run_id derivation and log output. Mirrors
    `build_backend`'s name-resolution rules but returns just the label.
    """
    name = backend_name
    if config is not None and not name:
        name = _strip_adapter_prefix(config.trace_backend.command)
    if not name:
        return "phoenix"
    return _BACKEND_ALIASES.get(name, name)


def build_backend(  # noqa: PLR0913 -- per-backend kwargs form one logical surface
    backend_name: str | None,
    config: Config | None,
    *,
    phoenix_url: str | None = None,
    phoenix_api_key: str | None = None,
    langfuse_host: str | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
) -> TraceBackend:
    """Construct a `TraceBackend` from CLI args + config.

    Phase 4 added Phoenix; Phase 6 added Langfuse; Phase 7 added LangSmith.
    """
    name = backend_name
    env: dict[str, str] = {}
    if config is not None:
        name = name or _strip_adapter_prefix(config.trace_backend.command)
        env = dict(config.trace_backend.env)
    if not name:
        raise ConfigError(
            "No trace backend specified. Pass --backend or set trace_backend in the config."
        )
    if name in ("phoenix", "agent-triage-adapter-phoenix"):
        url = phoenix_url or env.get("PHOENIX_URL")
        if not url:
            raise ConfigError(
                "Phoenix backend requires --phoenix-url or PHOENIX_URL in the config env."
            )
        return PhoenixAdapter(
            base_url=url,
            api_key=phoenix_api_key or env.get("PHOENIX_API_KEY"),
        )
    if name in ("langfuse", "agent-triage-adapter-langfuse"):
        host = langfuse_host or env.get("LANGFUSE_HOST")
        if not host:
            raise ConfigError(
                "Langfuse backend requires --langfuse-host or LANGFUSE_HOST in the config env."
            )
        return LangfuseAdapter(
            host=host,
            public_key=langfuse_public_key or env.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=langfuse_secret_key or env.get("LANGFUSE_SECRET_KEY"),
        )
    if name in ("langsmith", "agent-triage-adapter-langsmith"):
        api_key = langsmith_api_key or env.get("LANGSMITH_API_KEY")
        if not api_key:
            raise ConfigError(
                "LangSmith backend requires --langsmith-api-key or LANGSMITH_API_KEY in "
                "the config env."
            )
        return LangsmithAdapter(
            endpoint=langsmith_endpoint
            or env.get("LANGSMITH_ENDPOINT")
            or DEFAULT_LANGSMITH_ENDPOINT,
            api_key=api_key,
            project=langsmith_project or env.get("LANGSMITH_PROJECT"),
        )
    raise ConfigError(f"Unknown trace backend {name!r}. Supported: phoenix, langfuse, langsmith.")


def _strip_adapter_prefix(command: str) -> str:
    return command.removeprefix("agent-triage-adapter-")


def build_tracker(  # noqa: PLR0913 -- per-tracker kwargs form one logical surface
    tracker_name: str | None,
    config: Config | None,
    *,
    jira_host: str | None = None,
    jira_project: str | None = None,
    jira_email: str | None = None,
    jira_api_token: str | None = None,
    jira_pat: str | None = None,
    jira_deployment: str | None = None,
    linear_api_key: str | None = None,
    linear_team_id: str | None = None,
    linear_endpoint: str | None = None,
    github_token: str | None = None,
    github_owner: str | None = None,
    github_repo: str | None = None,
    github_api_url: str | None = None,
) -> Tracker | None:
    """Construct a `Tracker` from CLI args + config, or None if unconfigured.

    Returning None means "no tracker configured" — the pipeline then keeps
    its Phase 5 behavior (queue drafts to local files; no dedup).
    """
    name = tracker_name
    env: dict[str, str] = {}
    if config is not None and config.tracker is not None:
        name = name or _strip_adapter_prefix(config.tracker.command)
        env = dict(config.tracker.env)
    if not name:
        return None
    if name in ("jira", "agent-triage-adapter-jira"):
        host = jira_host or env.get("JIRA_HOST")
        project = jira_project or env.get("JIRA_PROJECT")
        if not host or not project:
            raise ConfigError(
                "Jira tracker requires --jira-host and --jira-project (or JIRA_HOST + "
                "JIRA_PROJECT in the config env)."
            )
        deployment_raw = jira_deployment or env.get("JIRA_DEPLOYMENT")
        deployment: Deployment | None = None
        if deployment_raw in ("cloud", "datacenter"):
            deployment = deployment_raw  # type: ignore[assignment]
        elif deployment_raw is not None:
            raise ConfigError(
                f"Unknown Jira deployment {deployment_raw!r}; expected 'cloud' or 'datacenter'."
            )
        return JiraAdapter(
            host=host,
            project=project,
            email=jira_email or env.get("JIRA_EMAIL"),
            api_token=jira_api_token or env.get("JIRA_API_TOKEN"),
            pat=jira_pat or env.get("JIRA_PAT"),
            deployment=deployment,
        )
    if name in ("linear", "agent-triage-adapter-linear"):
        api_key = linear_api_key or env.get("LINEAR_API_KEY")
        team_id = linear_team_id or env.get("LINEAR_TEAM_ID")
        if not api_key:
            raise ConfigError(
                "Linear tracker requires --linear-api-key or LINEAR_API_KEY in the config env."
            )
        if not team_id:
            raise ConfigError(
                "Linear tracker requires --linear-team or LINEAR_TEAM_ID in the config env."
            )
        return LinearAdapter(
            team_id=team_id,
            api_key=api_key,
            endpoint=linear_endpoint or env.get("LINEAR_ENDPOINT") or DEFAULT_LINEAR_ENDPOINT,
        )
    if name in ("github", "agent-triage-adapter-github"):
        token = github_token or env.get("GITHUB_TOKEN")
        owner = github_owner or env.get("GITHUB_OWNER")
        repo = github_repo or env.get("GITHUB_REPO")
        if not token:
            raise ConfigError(
                "GitHub tracker requires --github-token or GITHUB_TOKEN in the config env."
            )
        if not owner or not repo:
            raise ConfigError(
                "GitHub tracker requires --github-owner and --github-repo "
                "(or GITHUB_OWNER + GITHUB_REPO in the config env)."
            )
        return GitHubAdapter(
            owner=owner,
            repo=repo,
            token=token,
            api_url=github_api_url or env.get("GITHUB_API_URL") or DEFAULT_GITHUB_API,
        )
    raise ConfigError(f"Unknown tracker {name!r}. Supported: jira, linear, github.")

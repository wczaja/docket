"""agent-triage: observability-platform-agnostic triage runtime for LLM agent traces."""

from importlib.metadata import PackageNotFoundError, version

from agent_triage.errors import AgentTriageError

try:
    __version__ = version("agent-triage")
except PackageNotFoundError:  # pragma: no cover - source tree without an installed dist
    __version__ = "0.0.0+uninstalled"

__all__ = ["AgentTriageError", "__version__"]

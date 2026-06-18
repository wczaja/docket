"""docket: observability-platform-agnostic triage runtime for LLM agent traces."""

from importlib.metadata import PackageNotFoundError, version

from docket.errors import DocketError

try:
    __version__ = version("docket")
except PackageNotFoundError:  # pragma: no cover - source tree without an installed dist
    __version__ = "0.0.0+uninstalled"

__all__ = ["DocketError", "__version__"]

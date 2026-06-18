"""docket.yaml config loading via Pydantic."""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from docket.errors import ConfigError

AutoPostThreshold = Literal["critical", "high", "medium", "low", "never"]

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: object, path: Path) -> object:
    """Recursively expand ``${VAR}`` references in string values.

    Secrets belong in the environment, not in YAML; an unset variable is a
    configuration error, not an empty string.
    """
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise ConfigError(
                    f"Config at {path} references ${{{name}}} but {name} is not set "
                    "in the environment"
                )
            return resolved

        return _ENV_REF_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, path) for v in value]
    return value


class MCPServerConfig(BaseModel):
    type: Literal["mcp"]
    command: str
    env: dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    trace_backend: MCPServerConfig
    tracker: MCPServerConfig | None = None
    rubric: str
    max_traces_per_run: int = Field(default=1000, gt=0)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)
    auto_post_threshold: AutoPostThreshold = "never"
    instrumentation_backend: MCPServerConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        try:
            text = path.read_text()
        except FileNotFoundError as e:
            raise ConfigError(f"Config file not found: {path}") from e
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML at {path}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"Config at {path} must be a YAML mapping at the top level")
        data = _expand_env(data, path)
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ConfigError(f"Config at {path} failed validation: {e}") from e

"""Pydantic models for the rubric DSL (apiVersion agent-triage.dev/v1)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["critical", "high", "medium", "low"]
DetectionType = Literal["llm_judge", "regex", "tool_call", "metric_threshold", "composite"]
AutoPostThreshold = Literal["critical", "high", "medium", "low", "never"]


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: DetectionType
    prompt: str | None = None
    output_schema: dict[str, Any] | None = None
    pattern: str | None = None
    tool_calls: list[str] | None = None
    metric: str | None = None
    threshold: float | None = None
    operator: Literal["==", "!=", "<", "<=", ">", ">=", "and", "or"] | None = None
    operands: list["Detection"] | None = None
    model: str | None = None


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_excerpt: str
    context: str | None = None
    expected: Literal["positive", "negative"]


class Mode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str | None = None
    description: str | None = None
    severity: Severity
    detection: Detection
    examples: list[Example] = Field(default_factory=list)


class Clustering(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["per_mode_embedding"] = "per_mode_embedding"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    min_cluster_size: int = Field(default=3, ge=2)


class TriageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_post_threshold: AutoPostThreshold = "never"
    default_severity_to_tracker: dict[Severity, str] = Field(default_factory=dict)


class RubricMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    authors: list[str] = Field(default_factory=list)
    description: str | None = None


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    apiVersion: str = Field(pattern=r"^agent-triage\.dev/v\d+$")  # noqa: N815
    kind: Literal["Rubric"]
    metadata: RubricMetadata
    imports: list[str] = Field(default_factory=list)
    modes: list[Mode] = Field(default_factory=list)
    clustering: Clustering | None = None
    triage: TriageConfig | None = None

    @field_validator("modes")
    @classmethod
    def _unique_mode_ids(cls, modes: list[Mode]) -> list[Mode]:
        seen: dict[str, int] = {}
        for i, mode in enumerate(modes):
            if mode.id in seen:
                raise ValueError(
                    f"Duplicate mode id {mode.id!r} at indices {seen[mode.id]} and {i}"
                )
            seen[mode.id] = i
        return modes


Detection.model_rebuild()

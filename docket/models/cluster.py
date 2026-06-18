"""Cluster value types.

Per design §4.3 (Clusterer subagent): per-mode clusters of classified-positive
traces. The cluster_id is computed deterministically from `(mode_id, sorted
member trace IDs)` so re-runs against the same membership produce the same
id — important for the drafter's dedup logic in Phase 8.
"""

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from docket.models.classification import Severity


class ClusterStats(BaseModel):
    """Aggregate confidence statistics over a cluster's members."""

    model_config = ConfigDict(frozen=True)

    size: int
    min_confidence: float | None = None
    max_confidence: float | None = None
    mean_confidence: float | None = None


class Cluster(BaseModel):
    """A group of semantically-similar classified-positive traces for one mode."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    mode_id: str
    severity: Severity
    member_trace_ids: list[str]
    representative_trace_id: str
    representative_excerpt: str | None = None
    stats: ClusterStats = Field(default_factory=lambda: ClusterStats(size=0))


def compute_cluster_id(mode_id: str, trace_ids: list[str]) -> str:
    """Deterministic cluster id from mode_id + sorted member trace IDs."""
    h = hashlib.sha256()
    h.update(mode_id.encode("utf-8"))
    for tid in sorted(trace_ids):
        h.update(b"|")
        h.update(tid.encode("utf-8"))
    return h.hexdigest()[:12]

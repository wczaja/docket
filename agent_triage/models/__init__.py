"""Internal value types used across the runtime.

Phase 2 shipped `TraceLike` as a minimal stub for what detectors operate on.
Phase 3 added the canonical `OpenInferenceTrace` + `Span` types and the OTLP
normalizer; `TraceLike` stays as the detector input, with
`OpenInferenceTrace.to_trace_like()` bridging the two.

Phase 4 adds `Classification` and `Annotation` — the in-process verdict shape
and the trace-backend-bound view, respectively.
"""

from agent_triage.models.classification import Annotation, Classification, Severity
from agent_triage.models.cluster import Cluster, ClusterStats, compute_cluster_id
from agent_triage.models.otlp import from_otlp, to_otlp
from agent_triage.models.report import ModeStats, RunReport, TraceResult
from agent_triage.models.trace import (
    Document,
    Embedding,
    Event,
    Message,
    OpenInferenceTrace,
    Span,
    SpanKind,
    Status,
    ToolCall,
    TraceLike,
    Verdict,
)

__all__ = [
    "Annotation",
    "Classification",
    "Cluster",
    "ClusterStats",
    "Document",
    "Embedding",
    "Event",
    "Message",
    "ModeStats",
    "OpenInferenceTrace",
    "RunReport",
    "Severity",
    "Span",
    "SpanKind",
    "Status",
    "ToolCall",
    "TraceLike",
    "TraceResult",
    "Verdict",
    "compute_cluster_id",
    "from_otlp",
    "to_otlp",
]

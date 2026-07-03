"""Clusterer subagent: per-mode HDBSCAN over evidence-excerpt embeddings.

Per design §4.3:
  Input  - classified traces (those with at least one positive mode), the rubric
  Output - per-mode clusters
  Impl   - per-mode, embed the evidence excerpts; HDBSCAN cluster with the
           rubric's threshold; for each cluster, select the highest-confidence
           trace as the representative

HDBSCAN runs over a precomputed cosine-distance matrix (sklearn's HDBSCAN
supports "precomputed" metric). The rubric's `clustering.similarity_threshold`
maps to `1.0 - threshold` in distance space via `cluster_selection_epsilon`.

Clusters smaller than `min_cluster_size` are dropped (treated as HDBSCAN noise
and discarded).
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

from docket.llm.embeddings import EmbeddingProvider
from docket.models.classification import Classification
from docket.models.cluster import Cluster, ClusterStats, compute_cluster_id
from docket.models.trace import OpenInferenceTrace
from docket.observability import redact
from docket.rubric.spec import Mode, Rubric

log = logging.getLogger(__name__)

_DEFAULT_SIMILARITY_THRESHOLD = 0.82
_DEFAULT_MIN_CLUSTER_SIZE = 3


async def cluster_per_mode(
    classifications: list[Classification],
    *,
    rubric: Rubric,
    embedding_provider: EmbeddingProvider,
    traces_by_id: dict[str, OpenInferenceTrace] | None = None,
) -> list[Cluster]:
    """For each mode in `rubric`, cluster the classified-positive traces.

    `traces_by_id` is optional context the clusterer uses when a classification
    has no `excerpt` / `match` in its `extra` — it falls back to the trace's
    final response. Without it, classifications missing an excerpt embed as
    synthetic placeholder strings (which still group together but lose
    semantic resolution).
    """
    threshold = (
        rubric.clustering.similarity_threshold
        if rubric.clustering
        else _DEFAULT_SIMILARITY_THRESHOLD
    )
    min_size = (
        rubric.clustering.min_cluster_size if rubric.clustering else _DEFAULT_MIN_CLUSTER_SIZE
    )
    modes_by_id = {m.id: m for m in rubric.modes}
    positives_by_mode = _positives_by_mode(classifications, modes_by_id)

    clusters: list[Cluster] = []
    for mode_id, positives in positives_by_mode.items():
        if len(positives) < min_size:
            continue
        mode = modes_by_id[mode_id]
        # Redact PII once at this choke point: excerpts (classification
        # evidence or the trace's final response — both via _text_for) feed
        # the external embeddings API here and become `representative_excerpt`
        # on the Cluster, which the drafter prompt, report.md, queued issue
        # files, and eval cases all inherit verbatim.
        excerpts = [
            redact(_text_for(c, traces_by_id.get(c.trace_id) if traces_by_id else None))
            for c in positives
        ]
        embeddings = await embedding_provider.embed(excerpts)
        if len(embeddings) != len(positives):
            log.warning(
                "mode %s: embedding provider returned %d vectors for %d excerpts; skipping mode",
                mode_id,
                len(embeddings),
                len(positives),
            )
            continue  # provider failure; defensive
        labels = _hdbscan_labels(embeddings, min_size=min_size, threshold=threshold)
        clusters.extend(_build_clusters(mode, positives, excerpts, labels, min_size))
    return clusters


def cluster_mode_only(
    classifications: list[Classification],
    *,
    rubric: Rubric,
    traces_by_id: dict[str, OpenInferenceTrace] | None = None,
) -> list[Cluster]:
    """One cluster per mode containing every positive — no embeddings needed.

    The deliberately lossy fallback behind `--clustering mode-only`: it
    trades sub-pattern separation within a mode for zero embedding-provider
    requirements, so a single-API-key deployment still gets one draft per
    firing failure mode. `min_cluster_size` still gates drafting, and the
    representative is still the highest-confidence member.
    """
    min_size = (
        rubric.clustering.min_cluster_size if rubric.clustering else _DEFAULT_MIN_CLUSTER_SIZE
    )
    modes_by_id = {m.id: m for m in rubric.modes}
    clusters: list[Cluster] = []
    for mode_id, positives in _positives_by_mode(classifications, modes_by_id).items():
        if len(positives) < min_size:
            continue
        mode = modes_by_id[mode_id]
        # Same redaction choke point as the embedding path: excerpts become
        # `representative_excerpt` and flow into drafts, reports, and evals.
        excerpts = [
            redact(_text_for(c, traces_by_id.get(c.trace_id) if traces_by_id else None))
            for c in positives
        ]
        clusters.extend(_build_clusters(mode, positives, excerpts, [0] * len(positives), min_size))
    clusters.sort(key=lambda c: (c.mode_id, c.cluster_id))
    return clusters


def _positives_by_mode(
    classifications: list[Classification],
    modes_by_id: dict[str, Mode],
) -> dict[str, list[Classification]]:
    positives_by_mode: dict[str, list[Classification]] = {}
    for c in classifications:
        if not c.positive or c.error is not None:
            continue
        if c.mode_id not in modes_by_id:
            continue
        positives_by_mode.setdefault(c.mode_id, []).append(c)
    return positives_by_mode


def _text_for(
    classification: Classification,
    trace: OpenInferenceTrace | None,
) -> str:
    extra = classification.extra or {}
    excerpt = extra.get("excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        return excerpt
    match = extra.get("match")
    if isinstance(match, str) and match.strip():
        return match
    if trace is not None:
        final = trace.get_final_response()
        if final:
            return final
    return f"trace:{classification.trace_id} mode:{classification.mode_id}"


def _hdbscan_labels(
    embeddings: list[list[float]],
    *,
    min_size: int,
    threshold: float,
) -> list[int]:
    arr = np.asarray(embeddings, dtype=float)
    distances = cosine_distances(arr)
    # min_samples=1 + allow_single_cluster=True: HDBSCAN's "density variation"
    # assumption breaks down for tightly-similar input (all-zero distances).
    # These two settings make it behave like DBSCAN-with-epsilon: any point
    # within epsilon of another is in the same cluster; min_cluster_size still
    # discards groups too small to draft for.
    clusterer = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,
        metric="precomputed",
        cluster_selection_epsilon=max(0.0, 1.0 - threshold),
        allow_single_cluster=True,
        # `distances` is a throwaway local that nothing reads after this, so let
        # HDBSCAN consume it in place (the param only applies to
        # metric="precomputed"). Set explicitly to silence sklearn's
        # FutureWarning about the default flipping False->True in 1.10.
        copy=False,
    )
    try:
        labels: Any = clusterer.fit_predict(distances)
    except (TypeError, IndexError) as exc:
        # Known sklearn HDBSCAN bug: with `allow_single_cluster=True` and
        # `cluster_selection_epsilon > 0`, `epsilon_search` crashes inside
        # `traverse_upwards` when the EOM step picks the root as the only
        # cluster. NumPy >=2.4 surfaces it as TypeError; older NumPy raised
        # IndexError. The algorithm's intended output in that case is a single
        # cluster containing every point, which is what we emit here.
        # See https://github.com/scikit-learn-contrib/hdbscan/issues/370.
        if "0-dimensional" not in str(exc) and "out of bounds" not in str(exc):
            raise
        return [0] * arr.shape[0]
    return [int(label) for label in labels]


def _build_clusters(
    mode: Mode,
    positives: list[Classification],
    excerpts: list[str],
    labels: list[int],
    min_size: int,
) -> list[Cluster]:
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # noise
            continue
        groups.setdefault(label, []).append(idx)

    out: list[Cluster] = []
    for _label_id, indices in groups.items():
        if len(indices) < min_size:
            continue
        members = [positives[i] for i in indices]
        member_excerpts = [excerpts[i] for i in indices]
        rep_local_idx = _pick_representative_index(members)
        rep = members[rep_local_idx]
        confidences = [_confidence_of(m) for m in members]
        non_null = [c for c in confidences if c is not None]
        stats = ClusterStats(
            size=len(members),
            min_confidence=min(non_null) if non_null else None,
            max_confidence=max(non_null) if non_null else None,
            mean_confidence=(sum(non_null) / len(non_null)) if non_null else None,
        )
        out.append(
            Cluster(
                cluster_id=compute_cluster_id(mode.id, [m.trace_id for m in members]),
                mode_id=mode.id,
                severity=mode.severity,
                member_trace_ids=[m.trace_id for m in members],
                representative_trace_id=rep.trace_id,
                representative_excerpt=member_excerpts[rep_local_idx],
                stats=stats,
            )
        )
    # Sort for deterministic output ordering.
    out.sort(key=lambda c: (c.mode_id, c.cluster_id))
    return out


def _pick_representative_index(members: list[Classification]) -> int:
    best_idx = 0
    best_conf = _confidence_of(members[0]) or -1.0
    for i, m in enumerate(members[1:], start=1):
        conf = _confidence_of(m)
        if conf is not None and conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx


def _confidence_of(classification: Classification) -> float | None:
    extra = classification.extra or {}
    v = extra.get("confidence")
    if isinstance(v, (int, float)):
        return float(v)
    return None

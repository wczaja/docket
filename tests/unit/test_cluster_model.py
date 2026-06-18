from docket.models.cluster import Cluster, ClusterStats, compute_cluster_id


def test_compute_cluster_id_is_deterministic() -> None:
    a = compute_cluster_id("hallucination", ["t-1", "t-2", "t-3"])
    b = compute_cluster_id("hallucination", ["t-3", "t-1", "t-2"])
    assert a == b


def test_compute_cluster_id_changes_with_membership() -> None:
    a = compute_cluster_id("hallucination", ["t-1", "t-2"])
    b = compute_cluster_id("hallucination", ["t-1", "t-3"])
    assert a != b


def test_compute_cluster_id_changes_with_mode() -> None:
    a = compute_cluster_id("hallucination", ["t-1", "t-2"])
    b = compute_cluster_id("infinite-loop", ["t-1", "t-2"])
    assert a != b


def test_cluster_is_frozen() -> None:
    cluster = Cluster(
        cluster_id="abc",
        mode_id="hallucination",
        severity="critical",
        member_trace_ids=["t-1", "t-2", "t-3"],
        representative_trace_id="t-1",
    )
    try:
        cluster.member_trace_ids = ["x"]  # type: ignore[misc]
    except (TypeError, ValueError, AttributeError):
        return
    msg = "Cluster was not frozen"
    raise AssertionError(msg)


def test_cluster_stats_default_size_zero() -> None:
    stats = ClusterStats(size=0)
    assert stats.min_confidence is None
    assert stats.max_confidence is None
    assert stats.mean_confidence is None

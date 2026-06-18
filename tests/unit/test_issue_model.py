import json

from docket.models.issue import (
    Issue,
    IssueDraft,
    IssuePatch,
    IssueProvenance,
    make_labels,
)


def test_provenance_html_comment_is_parseable() -> None:
    prov = IssueProvenance(
        rubric_version="agents-builtin@1.0.0",
        mode_id="hallucination",
        cluster_id="abc123",
        representative_trace_id="t-1",
        run_id="r-1",
    )
    comment = prov.to_html_comment()
    assert comment.startswith("<!-- docket:provenance ")
    assert comment.endswith(" -->")
    body = comment[len("<!-- docket:provenance ") : -len(" -->")]
    parsed = json.loads(body)
    assert parsed["cluster_id"] == "abc123"


def test_make_labels_includes_required_three() -> None:
    labels = make_labels("hallucination", "agents-builtin@1.0.0")
    assert "docket" in labels
    assert "mode:hallucination" in labels
    assert "rubric:agents-builtin@1.0.0" in labels


def test_make_labels_truncates_long_rubric_label_deterministically() -> None:
    """A rubric label over 50 chars (the GitHub cap) is truncated to a
    41-char prefix + "-" + 8 hex chars of sha256, so creation and the dedup
    query always agree."""
    long_version = "a-very-long-rubric-name-for-overflow-testing@10.20.30-rc.1"
    first = make_labels("hallucination", long_version)
    second = make_labels("hallucination", long_version)
    assert first == second  # deterministic
    rubric_label = first[2]
    assert len(rubric_label) <= 50
    assert rubric_label.startswith(f"rubric:{long_version}"[:41] + "-")
    suffix = rubric_label.rsplit("-", 1)[-1]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_make_labels_replaces_spaces() -> None:
    labels = make_labels("hallucination", "my rubric@1.0.0")
    assert "rubric:my-rubric@1.0.0" in labels
    assert not any(" " in label for label in labels)


def test_issue_draft_renders_markdown() -> None:
    draft = IssueDraft(
        cluster_id="c-1",
        mode_id="hallucination",
        rubric_version="agents-builtin@1.0.0",
        run_id="r-1",
        severity="critical",
        representative_trace_id="t-1",
        member_trace_ids=["t-1", "t-2", "t-3"],
        title="Repeated hallucinations in customer responses",
        body="Body of the issue here.",
        labels=make_labels("hallucination", "agents-builtin@1.0.0"),
    )
    md = draft.to_markdown()
    assert "# Repeated hallucinations" in md
    assert "**Severity**: critical" in md
    assert "**Cluster**: `c-1`" in md
    assert "`t-1`" in md
    assert "`t-2`" in md


def test_issue_draft_json_record_round_trips() -> None:
    draft = IssueDraft(
        cluster_id="c-1",
        mode_id="hallucination",
        rubric_version="v",
        run_id="r",
        severity="critical",
        representative_trace_id="t-1",
        member_trace_ids=["t-1"],
        title="title",
        body="body",
    )
    serialized = json.dumps(draft.to_json_record())
    restored = IssueDraft.model_validate(json.loads(serialized))
    assert restored == draft


def test_provenance_round_trips_through_body_extraction() -> None:
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="cl-1",
        representative_trace_id="t-rep",
        run_id="r-1",
        member_trace_ids=["t-1", "t-2", "t-3"],
    )
    body = f"Some body text.\n\n{prov.to_html_comment()}"
    parsed = IssueProvenance.parse_from_body(body)
    assert parsed == prov


def test_parse_from_body_uses_last_provenance_block() -> None:
    """A drafted body quoting an example block earlier must not shadow the
    real trailing provenance comment."""
    quoted = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="quoted-example",
        representative_trace_id="t-x",
        run_id="r-x",
    )
    real = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="real-cluster",
        representative_trace_id="t-rep",
        run_id="r-1",
        member_trace_ids=["t-1"],
    )
    body = (
        "Bodies look like this:\n\n"
        f"> {quoted.to_html_comment()}\n\n"
        "Actual issue text.\n\n"
        f"{real.to_html_comment()}"
    )
    parsed = IssueProvenance.parse_from_body(body)
    assert parsed is not None
    assert parsed.cluster_id == "real-cluster"


def test_parse_from_body_tolerates_braces_inside_values() -> None:
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id='cl-{"weird"}-}brace{',
        representative_trace_id="t-}rep",
        run_id="r-1",
        member_trace_ids=["t-{1}"],
    )
    body = f"Some body.\n\n{prov.to_html_comment()}"
    parsed = IssueProvenance.parse_from_body(body)
    assert parsed == prov


def test_provenance_caps_member_trace_ids_at_100_with_flag() -> None:
    members = [f"t-{i:04d}" for i in range(150)]
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="cl-big",
        representative_trace_id="t-0000",
        run_id="r-1",
        member_trace_ids=members,
    )
    comment = prov.to_html_comment()
    payload = json.loads(comment[len("<!-- docket:provenance ") : -len(" -->")])
    assert len(payload["member_trace_ids"]) == 100
    assert payload["member_trace_ids"] == members[:100]
    assert payload["member_trace_ids_truncated"] is True
    # Round-trips through extraction: dedup overlap still works on first 100.
    parsed = IssueProvenance.parse_from_body(f"body\n\n{comment}")
    assert parsed is not None
    assert parsed.member_trace_ids == members[:100]


def test_provenance_under_cap_has_no_truncation_flag() -> None:
    prov = IssueProvenance(
        rubric_version="agents@1.0.0",
        mode_id="hallucination",
        cluster_id="cl-small",
        representative_trace_id="t-1",
        run_id="r-1",
        member_trace_ids=["t-1", "t-2"],
    )
    comment = prov.to_html_comment()
    payload = json.loads(comment[len("<!-- docket:provenance ") : -len(" -->")])
    assert "member_trace_ids_truncated" not in payload
    assert IssueProvenance.parse_from_body(f"body\n\n{comment}") == prov


def test_parse_from_body_returns_none_when_no_comment() -> None:
    assert IssueProvenance.parse_from_body("just a plain body, no provenance") is None


def test_parse_from_body_returns_none_when_payload_is_not_json() -> None:
    bad = "before <!-- docket:provenance not_json{} --> after"
    assert IssueProvenance.parse_from_body(bad) is None


def test_parse_from_body_returns_none_when_payload_is_json_but_wrong_shape() -> None:
    bad = 'before <!-- docket:provenance {"missing_required": true} --> after'
    assert IssueProvenance.parse_from_body(bad) is None


def test_issue_value_type_carries_state_and_url() -> None:
    issue = Issue(
        id="10001",
        key="AGT-1",
        url="https://example.atlassian.net/browse/AGT-1",
        title="existing",
        body="existing body",
        labels=["docket"],
        state="open",
    )
    assert issue.state == "open"
    assert issue.url is not None


def test_issue_patch_defaults_leave_fields_unset() -> None:
    patch = IssuePatch()
    assert patch.title is None
    assert patch.body is None
    assert patch.labels is None
    assert patch.state is None

"""Unit tests for `JiraAdapter` against a mocked HTTP transport."""

import base64
import json
from typing import Any

import httpx
import pytest

from docket.adapters.tracker.jira import (
    JiraAdapter,
    _adf_to_text,
    _detect_deployment,
    _markdown_to_adf,
)
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssuePatch, IssueProvenance


def _make_cloud_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    project: str = "AGT",
) -> JiraAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.atlassian.net")
    return JiraAdapter(
        host="https://example.atlassian.net",
        project=project,
        email="bot@example.com",
        api_token="cloud-token",  # noqa: S106
        client=client,
    )


def _make_dc_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    project: str = "AGT",
) -> JiraAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://jira.internal.example.com")
    return JiraAdapter(
        host="https://jira.internal.example.com",
        project=project,
        pat="dc-pat-token",  # noqa: S106
        client=client,
    )


def _draft(**overrides: Any) -> IssueDraft:
    base: dict[str, Any] = {
        "cluster_id": "cl-1",
        "mode_id": "hallucination",
        "rubric_version": "agents@1.0.0",
        "run_id": "run-abcdef0123456789",
        "severity": "high",
        "representative_trace_id": "t-rep",
        "member_trace_ids": ["t-1", "t-2", "t-3"],
        "title": "Agent hallucinates capital cities",
        "body": "Body of the issue.\n\nA second paragraph.",
        "labels": ["docket", "mode:hallucination", "rubric:agents@1.0.0"],
    }
    base.update(overrides)
    return IssueDraft(**base)


# -- deployment auto-detection ----------------------------------------------


def test_detect_deployment_for_cloud_host() -> None:
    assert _detect_deployment("https://example.atlassian.net") == "cloud"


def test_detect_deployment_for_internal_host() -> None:
    assert _detect_deployment("https://jira.internal.example.com") == "datacenter"


def test_explicit_deployment_overrides_detection() -> None:
    adapter = JiraAdapter(
        host="https://example.atlassian.net",
        project="AGT",
        pat="x",  # noqa: S106
        deployment="datacenter",
    )
    assert adapter.deployment == "datacenter"
    assert adapter.api_version == "2"


# -- auth shape -------------------------------------------------------------


def test_cloud_requires_email_and_api_token() -> None:
    with pytest.raises(TrackerError, match="email and api_token"):
        JiraAdapter(host="https://example.atlassian.net", project="AGT")


def test_cloud_partial_credentials_rejected() -> None:
    with pytest.raises(TrackerError, match="email and api_token"):
        JiraAdapter(
            host="https://example.atlassian.net",
            project="AGT",
            email="bot@example.com",
        )


def test_datacenter_requires_pat() -> None:
    with pytest.raises(TrackerError, match="Personal Access Token"):
        JiraAdapter(host="https://jira.internal.example.com", project="AGT")


async def test_default_client_sets_basic_auth_for_cloud() -> None:
    adapter = JiraAdapter(
        host="https://example.atlassian.net",
        project="AGT",
        email="bot@example.com",
        api_token="cloud-token",  # noqa: S106
    )
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    expected = base64.b64encode(b"bot@example.com:cloud-token").decode("ascii")
    assert client.headers.get("Authorization") == f"Basic {expected}"
    await adapter.close()


async def test_default_client_sets_bearer_auth_for_datacenter() -> None:
    adapter = JiraAdapter(
        host="https://jira.internal.example.com",
        project="AGT",
        pat="dc-pat-token",  # noqa: S106
    )
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert client.headers.get("Authorization") == "Bearer dc-pat-token"
    await adapter.close()


async def test_default_client_constructed_lazily() -> None:
    adapter = JiraAdapter(
        host="https://example.atlassian.net",
        project="AGT",
        email="bot@example.com",
        api_token="cloud-token",  # noqa: S106
    )
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001
    adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert adapter._client is not None  # type: ignore[attr-defined]  # noqa: SLF001
    await adapter.close()
    assert adapter._client is None  # type: ignore[attr-defined]  # noqa: SLF001


async def test_close_when_never_used_is_safe() -> None:
    adapter = JiraAdapter(
        host="https://example.atlassian.net",
        project="AGT",
        email="bot@example.com",
        api_token="cloud-token",  # noqa: S106
    )
    await adapter.close()


# -- list_open_issues -------------------------------------------------------


async def test_list_open_issues_filters_by_labels_via_jql_on_cloud() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "issues": [
                    {
                        "id": "10001",
                        "key": "AGT-1",
                        "fields": {
                            "summary": "Existing issue",
                            "description": {
                                "type": "doc",
                                "version": 1,
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [{"type": "text", "text": "existing body"}],
                                    }
                                ],
                            },
                            "labels": ["docket", "mode:hallucination"],
                            "status": {"name": "Open"},
                        },
                    }
                ],
                "total": 1,
            },
        )

    adapter = _make_cloud_adapter(handler)
    issues = await adapter.list_open_issues(
        filter={"labels": ["docket", "mode:hallucination"]},
    )
    assert captured["path"] == "/rest/api/3/search/jql"
    jql = captured["params"]["jql"]
    assert 'project = "AGT"' in jql
    assert "resolution = Unresolved" in jql
    assert 'labels = "docket"' in jql
    assert 'labels = "mode:hallucination"' in jql
    assert len(issues) == 1
    assert issues[0].id == "10001"
    assert issues[0].key == "AGT-1"
    assert issues[0].url == "https://example.atlassian.net/browse/AGT-1"
    assert issues[0].body == "existing body"
    assert issues[0].state == "open"


async def test_list_open_issues_uses_v2_path_for_datacenter() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"issues": [], "total": 0})

    adapter = _make_dc_adapter(handler)
    await adapter.list_open_issues(filter={"labels": ["docket"]})
    assert captured["path"] == "/rest/api/2/search"


async def test_list_open_issues_paginates_with_next_page_token_on_cloud() -> None:
    """Cloud uses `/rest/api/3/search/jql`: the token is echoed back on page 2
    and omitted by the server on the last page."""
    page_count = 0
    tokens_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_count
        page_count += 1
        assert request.url.path == "/rest/api/3/search/jql"
        assert "startAt" not in request.url.params
        tokens_seen.append(request.url.params.get("nextPageToken"))
        if page_count == 1:
            return httpx.Response(
                200,
                json={
                    "issues": [{"id": str(i), "key": f"AGT-{i}", "fields": {}} for i in range(50)],
                    "nextPageToken": "tok-2",
                },
            )
        return httpx.Response(
            200,
            json={
                "issues": [{"id": str(i), "key": f"AGT-{i}", "fields": {}} for i in range(50, 60)],
            },
        )

    adapter = _make_cloud_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["docket"]})
    assert len(issues) == 60
    assert page_count == 2
    assert tokens_seen == [None, "tok-2"]


async def test_list_open_issues_paginates_with_start_at_on_datacenter() -> None:
    """Data Center keeps the classic `/rest/api/2/search` startAt/total shape."""
    page_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_count
        page_count += 1
        assert request.url.path == "/rest/api/2/search"
        start_at = int(request.url.params.get("startAt", "0"))
        if start_at == 0:
            return httpx.Response(
                200,
                json={
                    "issues": [{"id": str(i), "key": f"AGT-{i}", "fields": {}} for i in range(50)],
                    "total": 60,
                },
            )
        return httpx.Response(
            200,
            json={
                "issues": [{"id": str(i), "key": f"AGT-{i}", "fields": {}} for i in range(50, 60)],
                "total": 60,
            },
        )

    adapter = _make_dc_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["docket"]})
    assert len(issues) == 60
    assert page_count == 2


async def test_cloud_search_page_cap_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Hitting the page cap must warn loudly (dedup may miss matches), not raise."""

    def handler(_request: httpx.Request) -> httpx.Response:
        # Always claims another page exists.
        return httpx.Response(
            200,
            json={
                "issues": [{"id": "1", "key": "AGT-1", "fields": {}}],
                "nextPageToken": "tok-more",
            },
        )

    adapter = _make_cloud_adapter(handler)
    with caplog.at_level("WARNING", logger="docket.adapters.tracker.jira"):
        issues = await adapter.list_open_issues(filter={"labels": ["docket"]})
    assert len(issues) == 20  # one issue per page, capped at _MAX_PAGES
    assert any("dedup may miss matches" in r.message for r in caplog.records)
    listing = await adapter.list_open_issues_v2(filter={"labels": ["docket"]})
    assert listing.truncated is True
    assert listing.page_limit == 20


async def test_datacenter_search_page_cap_flags_truncation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        start_at = int(request.url.params.get("startAt", "0"))
        # Claims 10k results; serves one per page.
        return httpx.Response(
            200,
            json={
                "issues": [{"id": str(start_at), "key": f"AGT-{start_at}", "fields": {}}],
                "total": 10_000,
            },
        )

    listing = await _make_dc_adapter(handler).list_open_issues_v2(filter={"labels": ["docket"]})
    assert listing.truncated is True
    assert len(listing.issues) == 20


async def test_list_open_issues_v2_is_complete_when_pages_run_out() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issues": [{"id": "1", "key": "AGT-1", "fields": {}}]})

    listing = await _make_cloud_adapter(handler).list_open_issues_v2(filter={"labels": ["docket"]})
    assert listing.truncated is False
    assert [i.key for i in listing.issues] == ["AGT-1"]


async def test_list_open_issues_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 403"):
        await adapter.list_open_issues(filter={"labels": ["docket"]})


async def test_search_returns_non_dict_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[1,2,3]", headers={"content-type": "application/json"})

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="non-object"):
        await adapter.list_open_issues(filter={"labels": ["docket"]})


async def test_search_returns_non_list_issues_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issues": {"weird": "shape"}, "total": 1})

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="non-list `issues`"):
        await adapter.list_open_issues(filter={"labels": ["docket"]})


async def test_search_returns_non_json_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>nope</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="non-JSON"):
        await adapter.list_open_issues(filter={"labels": ["docket"]})


# -- search_issues ----------------------------------------------------------


async def test_search_issues_builds_text_jql_and_respects_k() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "issues": [{"id": str(i), "key": f"AGT-{i}", "fields": {}} for i in range(5)],
                "total": 5,
            },
        )

    adapter = _make_cloud_adapter(handler)
    results = await adapter.search_issues("paris capital", k=5)
    assert len(results) == 5
    jql = captured["params"]["jql"]
    assert 'project = "AGT"' in jql
    assert 'text ~ "paris capital"' in jql


# -- create_issue -----------------------------------------------------------


async def test_create_issue_posts_adf_body_on_cloud_and_returns_typed_issue() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201,
            json={
                "id": "10042",
                "key": "AGT-42",
                "self": "https://example.atlassian.net/rest/api/3/issue/10042",
            },
        )

    adapter = _make_cloud_adapter(handler)
    draft = _draft()
    issue = await adapter.create_issue(draft)

    assert captured["path"] == "/rest/api/3/issue"
    fields = captured["body"]["fields"]
    assert fields["project"] == {"key": "AGT"}
    assert fields["summary"] == draft.title
    assert fields["labels"] == draft.labels
    # ADF check: top-level doc with paragraph(s) of text.
    desc = fields["description"]
    assert desc["type"] == "doc"
    assert desc["version"] == 1
    para_texts = [node["content"][0]["text"] for node in desc["content"] if node.get("content")]
    assert "Body of the issue." in para_texts
    assert "A second paragraph." in para_texts

    # Returned Issue is properly populated.
    assert issue.id == "10042"
    assert issue.key == "AGT-42"
    assert issue.url == "https://example.atlassian.net/browse/AGT-42"
    assert issue.title == draft.title
    assert issue.body == draft.body
    assert issue.labels == draft.labels


async def test_create_issue_sends_plain_text_body_on_datacenter() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "777", "key": "AGT-777"})

    adapter = _make_dc_adapter(handler)
    draft = _draft()
    await adapter.create_issue(draft)
    assert captured["path"] == "/rest/api/2/issue"
    assert captured["body"]["fields"]["description"] == draft.body


async def test_create_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad project")

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 400"):
        await adapter.create_issue(_draft())


async def test_create_issue_raises_when_response_missing_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={})

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="no id/key"):
        await adapter.create_issue(_draft())


# -- update_issue -----------------------------------------------------------


async def test_update_issue_patches_only_provided_fields() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            calls.append(
                {
                    "path": request.url.path,
                    "body": json.loads(request.read().decode()),
                }
            )
            return httpx.Response(204)
        # GET refetch
        return httpx.Response(
            200,
            json={
                "id": "10042",
                "key": "AGT-42",
                "fields": {
                    "summary": "New title",
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Body of the issue."}],
                            }
                        ],
                    },
                    "labels": ["docket"],
                    "status": {"name": "Open"},
                },
            },
        )

    adapter = _make_cloud_adapter(handler)
    patch = IssuePatch(title="New title")
    issue = await adapter.update_issue("10042", patch)

    assert len(calls) == 1
    assert calls[0]["path"] == "/rest/api/3/issue/10042"
    assert calls[0]["body"]["fields"] == {"summary": "New title"}
    assert issue.title == "New title"


async def test_update_issue_with_only_state_change_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="state transitions are not supported"):
        await adapter.update_issue("10042", IssuePatch(state="closed"))


async def test_update_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 500"):
        await adapter.update_issue("10042", IssuePatch(title="x"))


# -- comment_on_issue -------------------------------------------------------


async def test_comment_on_issue_posts_adf_on_cloud() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "c-1"})

    adapter = _make_cloud_adapter(handler)
    await adapter.comment_on_issue("10042", "New cluster members:\n\n- t-7\n- t-8")
    assert captured["path"] == "/rest/api/3/issue/10042/comment"
    body = captured["body"]["body"]
    assert body["type"] == "doc"
    # The bullet list lands as one paragraph in v1.0 (no rich formatting).
    texts = [n["content"][0]["text"] for n in body["content"] if n.get("content")]
    assert any("New cluster members" in t for t in texts)


async def test_comment_on_issue_posts_plain_text_on_datacenter() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "c-1"})

    adapter = _make_dc_adapter(handler)
    await adapter.comment_on_issue("10042", "plain text comment")
    assert captured["body"]["body"] == "plain text comment"


async def test_comment_on_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 404"):
        await adapter.comment_on_issue("10042", "hi")


# -- retry behavior ----------------------------------------------------------


async def test_list_open_issues_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"issues": [], "total": 0})

    adapter = _make_cloud_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["docket"]})
    assert issues == []
    assert calls == 2


async def test_create_issue_retries_429_then_succeeds() -> None:
    """A 429'd create was rejected by the server, so retrying it is safe."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(201, json={"id": "10042", "key": "AGT-42"})

    adapter = _make_cloud_adapter(handler)
    issue = await adapter.create_issue(_draft())
    assert issue.key == "AGT-42"
    assert calls == 2


async def test_create_issue_connect_error_raises_tracker_error_without_retry() -> None:
    """Transport errors on a non-idempotent create must surface as TrackerError
    immediately — a blind retry could double-post."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    adapter = _make_cloud_adapter(handler)
    with pytest.raises(TrackerError, match="failed"):
        await adapter.create_issue(_draft())
    assert calls == 1


# -- severity → priority -----------------------------------------------------


async def test_create_issue_maps_p_levels_to_jira_priority_names() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "1", "key": "AGT-1"})

    adapter = _make_cloud_adapter(handler)
    await adapter.create_issue(_draft(priority="P2"))
    assert captured["body"]["fields"]["priority"] == {"name": "High"}


async def test_create_issue_passes_unknown_priority_name_verbatim() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "1", "key": "AGT-1"})

    adapter = _make_cloud_adapter(handler)
    await adapter.create_issue(_draft(priority="Sev-1"))
    assert captured["body"]["fields"]["priority"] == {"name": "Sev-1"}


async def test_create_issue_omits_priority_when_draft_has_none() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": "1", "key": "AGT-1"})

    adapter = _make_cloud_adapter(handler)
    await adapter.create_issue(_draft())
    assert "priority" not in captured["body"]["fields"]


# -- ADF helpers ------------------------------------------------------------


def test_markdown_to_adf_splits_on_blank_lines() -> None:
    doc = _markdown_to_adf("first paragraph\n\nsecond paragraph\n\nthird")
    paras = [n["content"][0]["text"] for n in doc["content"] if n.get("content")]
    assert paras == ["first paragraph", "second paragraph", "third"]


def test_markdown_to_adf_empty_input_yields_empty_paragraph() -> None:
    doc = _markdown_to_adf("")
    assert doc["content"] == [{"type": "paragraph", "content": []}]


def test_adf_to_text_round_trips_paragraphs() -> None:
    doc = _markdown_to_adf("alpha\n\nbeta")
    assert _adf_to_text(doc) == "alpha\n\nbeta"


def test_adf_to_text_handles_none_and_non_dict() -> None:
    assert _adf_to_text(None) == ""
    assert _adf_to_text("already text") == "already text"
    assert _adf_to_text([1, 2, 3]) == ""


def _search_handler_returning(issue: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issues": [issue]})

    return handler


async def test_status_category_beats_localized_name() -> None:
    """m-6: a localized/custom done-status maps to closed via statusCategory."""
    issue = {
        "id": "1",
        "key": "AGT-9",
        "fields": {
            "summary": "t",
            "description": None,
            "labels": [],
            "status": {
                # A localized name the fallback name-list can't know about.
                "name": "Fertig",
                "statusCategory": {"key": "done"},
            },
        },
    }
    adapter = _make_cloud_adapter(_search_handler_returning(issue))
    issues = await adapter.list_open_issues()
    assert issues[0].state == "closed"


async def test_status_category_indeterminate_is_open() -> None:
    issue = {
        "id": "1",
        "key": "AGT-9",
        "fields": {
            "summary": "t",
            "description": None,
            "labels": [],
            "status": {"name": "Done-ish", "statusCategory": {"key": "indeterminate"}},
        },
    }
    adapter = _make_cloud_adapter(_search_handler_returning(issue))
    issues = await adapter.list_open_issues()
    assert issues[0].state == "open"


# -- update_provenance ----------------------------------------------------------


def _provenance(**overrides: Any) -> IssueProvenance:
    fields: dict[str, Any] = {
        "rubric_version": "agents@1.0.0",
        "mode_id": "hallucination",
        "cluster_id": "cl-1",
        "representative_trace_id": "t-1",
        "run_id": "r-1",
        "member_trace_ids": ["t-1"],
    }
    fields.update(overrides)
    return IssueProvenance.model_validate(fields)


def _provenance_issue(prov: IssueProvenance) -> Issue:
    return Issue(
        id="10001",
        key="AGT-1",
        title="t",
        body=f"Flattened body.\n\n{prov.to_html_comment()}",
        labels=["docket"],
    )


async def test_cloud_update_provenance_patches_only_the_provenance_paragraph() -> None:
    """`Issue.body` on Cloud is a lossy text rendering; the live ADF document
    is patched instead, so formatting humans added survives."""
    old = _provenance()
    new = old.with_reported(cluster_id="cl-2", trace_ids=["t-2"])
    heading = {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [{"type": "text", "text": "RCA"}],
    }
    bullets = {
        "type": "bulletList",
        "content": [
            {
                "type": "listItem",
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "cache miss"}]}
                ],
            }
        ],
    }
    marked = {
        "type": "paragraph",
        "content": [{"type": "text", "text": "Owner: ", "marks": [{"type": "strong"}]}],
    }
    live_doc = {
        "type": "doc",
        "version": 1,
        "content": [
            heading,
            bullets,
            marked,
            {"type": "paragraph", "content": [{"type": "text", "text": old.to_html_comment()}]},
        ],
    }
    requests: list[tuple[str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/3/issue/10001"
        if request.method == "GET":
            requests.append(("GET", None))
            assert request.url.params.get("fields") == "description"
            return httpx.Response(200, json={"id": "10001", "fields": {"description": live_doc}})
        requests.append((request.method, json.loads(request.read().decode())))
        return httpx.Response(204)

    await _make_cloud_adapter(handler).update_provenance(_provenance_issue(old), new)
    assert [method for method, _ in requests] == ["GET", "PUT"]
    put = requests[1][1]
    assert put is not None
    content = put["fields"]["description"]["content"]
    assert content[:3] == [heading, bullets, marked]
    assert content[3] == {
        "type": "paragraph",
        "content": [{"type": "text", "text": new.to_html_comment()}],
    }


async def test_cloud_update_provenance_appends_a_paragraph_when_missing() -> None:
    new = _provenance()
    puts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": "10001", "fields": {"description": None}})
        puts.append(json.loads(request.read().decode()))
        return httpx.Response(204)

    await _make_cloud_adapter(handler).update_provenance(_provenance_issue(new), new)
    ((put,),) = [puts]
    assert put["fields"]["description"] == {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": new.to_html_comment()}]}
        ],
    }


@pytest.mark.parametrize(("failing_method", "match"), [("GET", "get_issue"), ("PUT", "update")])
async def test_cloud_update_provenance_raises_on_http_error(
    failing_method: str, match: str
) -> None:
    prov = _provenance()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == failing_method:
            return httpx.Response(403, text="forbidden")
        return httpx.Response(200, json={"id": "10001", "fields": {"description": None}})

    with pytest.raises(TrackerError, match=f"{match}.*403"):
        await _make_cloud_adapter(handler).update_provenance(_provenance_issue(prov), prov)


async def test_datacenter_update_provenance_rewrites_the_verbatim_description() -> None:
    old = _provenance()
    new = old.with_reported(cluster_id="cl-2", trace_ids=["t-2"])
    puts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            puts.append(json.loads(request.read().decode()))
            return httpx.Response(204)
        return httpx.Response(200, json={"id": "10001", "key": "AGT-1", "fields": {}})

    await _make_dc_adapter(handler).update_provenance(_provenance_issue(old), new)
    ((put,),) = [puts]
    assert put == {"fields": {"description": f"Flattened body.\n\n{new.to_html_comment()}"}}

"""Unit tests for `GitHubAdapter` against a mocked HTTP transport."""

import json
from typing import Any

import httpx
import pytest

from docket.adapters.tracker.github import (
    DEFAULT_GITHUB_API,
    GitHubAdapter,
    _parse_link_next,
)
from docket.errors import TrackerError
from docket.models.issue import IssueDraft, IssuePatch


def _make_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    owner: str = "docket",
    repo: str = "docket",
) -> GitHubAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=DEFAULT_GITHUB_API)
    return GitHubAdapter(
        owner=owner,
        repo=repo,
        token="gh-test",  # noqa: S106
        client=client,
    )


def _draft(**overrides: Any) -> IssueDraft:
    base: dict[str, Any] = {
        "cluster_id": "cl-1",
        "mode_id": "hallucination",
        "rubric_version": "agents@1.0.0",
        "run_id": "run-abc",
        "severity": "high",
        "representative_trace_id": "t-rep",
        "member_trace_ids": ["t-1", "t-2"],
        "title": "Agent hallucinates capitals",
        "body": "Body of the issue.\n\nSecond paragraph.",
        "labels": ["docket", "mode:hallucination", "rubric:agents@1.0.0"],
    }
    base.update(overrides)
    return IssueDraft(**base)


# -- credentials / headers --------------------------------------------------


def test_default_api_url_points_at_dot_com() -> None:
    assert DEFAULT_GITHUB_API == "https://api.github.com"


def test_construction_requires_token_or_client() -> None:
    with pytest.raises(TrackerError, match="GITHUB_TOKEN"):
        GitHubAdapter(owner="o", repo="r")


async def test_default_client_sets_bearer_and_required_headers() -> None:
    adapter = GitHubAdapter(owner="o", repo="r", token="ghp_xyz")  # noqa: S106
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert client.headers.get("Authorization") == "Bearer ghp_xyz"
    assert client.headers.get("Accept") == "application/vnd.github+json"
    assert client.headers.get("X-GitHub-Api-Version") == "2022-11-28"
    assert client.headers.get("User-Agent") == "docket"
    await adapter.close()


async def test_close_when_never_used_is_safe() -> None:
    adapter = GitHubAdapter(owner="o", repo="r", token="t")  # noqa: S106
    await adapter.close()


def test_repo_path_property() -> None:
    adapter = GitHubAdapter(owner="acme", repo="widgets", token="t")  # noqa: S106
    assert adapter.repo_path == "acme/widgets"


# -- list_open_issues -------------------------------------------------------


async def test_list_open_issues_passes_labels_csv_and_open_state() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json=[
                {
                    "id": 1001,
                    "number": 1,
                    "html_url": "https://github.com/docket/docket/issues/1",
                    "title": "Existing",
                    "body": "existing body",
                    "state": "open",
                    "labels": [{"name": "docket"}, {"name": "mode:hallucination"}],
                }
            ],
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues(
        filter={"labels": ["docket", "mode:hallucination"]},
    )
    assert captured["path"] == "/repos/docket/docket/issues"
    assert captured["params"]["state"] == "open"
    assert captured["params"]["labels"] == "docket,mode:hallucination"
    assert len(issues) == 1
    assert issues[0].id == "1"
    assert issues[0].key == "1"
    assert issues[0].url == "https://github.com/docket/docket/issues/1"
    assert issues[0].state == "open"
    assert sorted(issues[0].labels) == ["docket", "mode:hallucination"]


async def test_list_open_issues_skips_pull_request_items() -> None:
    """The GitHub Issues endpoint returns PRs as well; we filter them out."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": 1, "number": 1, "title": "issue", "body": "", "state": "open"},
                {
                    "id": 2,
                    "number": 2,
                    "title": "pr",
                    "body": "",
                    "state": "open",
                    "pull_request": {"url": "..."},
                },
            ],
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues()
    assert [i.key for i in issues] == ["1"]


async def test_list_open_issues_follows_link_next_pagination() -> None:
    page_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_count
        page_count += 1
        if page_count == 1:
            assert request.url.params.get("state") == "open"
            return httpx.Response(
                200,
                json=[{"id": 1, "number": 1, "title": "a", "body": "", "state": "open"}],
                headers={
                    "Link": (
                        '<https://api.github.com/repos/o/r/issues?page=2>; rel="next", '
                        '<https://api.github.com/repos/o/r/issues?page=5>; rel="last"'
                    )
                },
            )
        return httpx.Response(
            200,
            json=[{"id": 2, "number": 2, "title": "b", "body": "", "state": "open"}],
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues()
    assert page_count == 2
    assert [i.key for i in issues] == ["1", "2"]


async def test_list_open_issues_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad credentials")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 401"):
        await adapter.list_open_issues()


async def test_list_open_issues_raises_when_response_is_dict() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "not a list"})

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="returned non-list"):
        await adapter.list_open_issues()


async def test_list_open_issues_raises_on_non_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not json",
            headers={"content-type": "text/plain"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="non-JSON"):
        await adapter.list_open_issues()


# -- search_issues ----------------------------------------------------------


async def test_search_issues_scopes_query_to_repo_and_kind() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": 9,
                        "number": 9,
                        "title": "match",
                        "body": "",
                        "state": "open",
                        "labels": [],
                    }
                ]
            },
        )

    adapter = _make_adapter(handler)
    results = await adapter.search_issues("paris capital", k=5)
    assert captured["path"] == "/search/issues"
    q = captured["params"]["q"]
    assert "repo:docket/docket" in q
    assert "is:issue" in q
    assert "state:open" in q
    # The user query is quoted so qualifier-like tokens can't rewrite scope.
    assert '"paris capital"' in q
    assert captured["params"]["per_page"] == "5"
    assert len(results) == 1
    assert results[0].key == "9"


async def test_search_issues_quotes_query_and_escapes_embedded_quotes() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": []})

    adapter = _make_adapter(handler)
    await adapter.search_issues('repo:evil/evil "quoted" state:closed')
    q = captured["params"]["q"]
    # The whole user query lands inside one quoted term with inner quotes escaped.
    assert '"repo:evil/evil \\"quoted\\" state:closed"' in q
    assert q.startswith("repo:docket/docket is:issue state:open ")


async def test_search_issues_raises_when_items_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": "not-a-list"})

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="non-list `items`"):
        await adapter.search_issues("anything")


# -- create_issue -----------------------------------------------------------


async def test_create_issue_posts_markdown_body_directly() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201,
            json={
                "id": 42,
                "number": 42,
                "html_url": "https://github.com/docket/docket/issues/42",
                "title": "Agent hallucinates capitals",
                "body": "Body of the issue.\n\nSecond paragraph.",
                "state": "open",
                "labels": [
                    {"name": "docket"},
                    {"name": "mode:hallucination"},
                    {"name": "rubric:agents@1.0.0"},
                ],
            },
        )

    adapter = _make_adapter(handler)
    draft = _draft()
    issue = await adapter.create_issue(draft)

    assert captured["path"] == "/repos/docket/docket/issues"
    body = captured["body"]
    assert body["title"] == draft.title
    # GitHub takes markdown directly — no ADF, no wiki encoding.
    assert body["body"] == draft.body
    assert body["labels"] == draft.labels

    assert issue.key == "42"
    assert issue.url == "https://github.com/docket/docket/issues/42"
    assert issue.body == draft.body
    assert sorted(issue.labels) == sorted(draft.labels)


async def test_create_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="validation failed")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 422"):
        await adapter.create_issue(_draft())


async def test_create_issue_skips_priority_silently() -> None:
    """GitHub has no priority field; the severity context rides on labels."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201,
            json={"id": 1, "number": 1, "title": "t", "body": "b", "state": "open", "labels": []},
        )

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft(priority="P1"))
    assert "priority" not in captured["body"]


# -- retry behavior ----------------------------------------------------------


async def test_list_open_issues_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=[])

    adapter = _make_adapter(handler)
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
        return httpx.Response(
            201,
            json={"id": 42, "number": 42, "title": "t", "body": "b", "state": "open", "labels": []},
        )

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft())
    assert issue.key == "42"
    assert calls == 2


async def test_create_issue_connect_error_raises_tracker_error_without_retry() -> None:
    """Transport errors on a non-idempotent create must surface as TrackerError
    immediately — a blind retry could double-post."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed"):
        await adapter.create_issue(_draft())
    assert calls == 1


async def test_list_open_issues_page_cap_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Hitting the page cap must warn loudly (dedup may miss matches), not raise."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=[{"id": calls, "number": calls, "title": "t", "body": "", "state": "open"}],
            headers={"Link": '<https://api.github.com/repos/o/r/issues?page=99>; rel="next"'},
        )

    adapter = _make_adapter(handler)
    with caplog.at_level("WARNING", logger="docket.adapters.tracker.github"):
        issues = await adapter.list_open_issues()
    assert calls == 20  # _MAX_PAGES
    assert len(issues) == 20
    assert any("dedup may miss matches" in r.message for r in caplog.records)


# -- update_issue -----------------------------------------------------------


async def test_update_issue_patches_only_provided_fields() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "id": 42,
                "number": 42,
                "title": "Updated title",
                "body": "old body",
                "state": "open",
                "labels": [],
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.update_issue("42", IssuePatch(title="Updated title"))
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/repos/docket/docket/issues/42"
    assert captured["body"] == {"title": "Updated title"}
    assert issue.title == "Updated title"


async def test_update_issue_supports_state_transition_to_closed() -> None:
    """Unlike Jira/Linear, GitHub has a native open|closed flag."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "id": 42,
                "number": 42,
                "title": "t",
                "body": "b",
                "state": "closed",
                "labels": [],
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.update_issue("42", IssuePatch(state="closed"))
    assert captured["body"] == {"state": "closed"}
    assert issue.state == "closed"


async def test_update_issue_with_empty_patch_refetches() -> None:
    method_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method_seen.append(request.method)
        return httpx.Response(
            200,
            json={
                "id": 42,
                "number": 42,
                "title": "t",
                "body": "b",
                "state": "open",
                "labels": [],
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.update_issue("42", IssuePatch())
    assert method_seen == ["GET"]
    assert issue.key == "42"


async def test_update_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 404"):
        await adapter.update_issue("42", IssuePatch(title="x"))


# -- comment_on_issue -------------------------------------------------------


async def test_comment_on_issue_posts_markdown_body() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(201, json={"id": 999})

    adapter = _make_adapter(handler)
    await adapter.comment_on_issue("42", "New members:\n\n- `t-3`")
    assert captured["path"] == "/repos/docket/docket/issues/42/comments"
    assert captured["body"] == {"body": "New members:\n\n- `t-3`"}


async def test_comment_on_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 403"):
        await adapter.comment_on_issue("42", "hi")


# -- helpers ----------------------------------------------------------------


def test_parse_link_next_extracts_url() -> None:
    header = (
        '<https://api.github.com/repos/o/r/issues?page=2>; rel="next", '
        '<https://api.github.com/repos/o/r/issues?page=5>; rel="last"'
    )
    assert _parse_link_next(header) == "https://api.github.com/repos/o/r/issues?page=2"


def test_parse_link_next_returns_none_when_no_next() -> None:
    header = '<https://api.github.com/repos/o/r/issues?page=1>; rel="prev"'
    assert _parse_link_next(header) is None


def test_parse_link_next_returns_none_for_empty_header() -> None:
    assert _parse_link_next(None) is None
    assert _parse_link_next("") is None

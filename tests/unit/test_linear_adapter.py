"""Unit tests for `LinearAdapter` against a mocked HTTP transport."""

import json
from typing import Any

import httpx
import pytest

from agent_triage.adapters.tracker.linear import (
    DEFAULT_LINEAR_ENDPOINT,
    LinearAdapter,
)
from agent_triage.errors import TrackerError
from agent_triage.models.issue import IssueDraft, IssuePatch


def _make_adapter(
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
    *,
    team_id: str = "team-123",
) -> LinearAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://linear.test")
    return LinearAdapter(team_id=team_id, api_key="ln-test", client=client)  # noqa: S106


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
        "labels": ["agent-triage", "mode:hallucination", "rubric:agents@1.0.0"],
    }
    base.update(overrides)
    return IssueDraft(**base)


# -- credential / endpoint --------------------------------------------------


def test_default_endpoint_points_at_linear_cloud() -> None:
    assert DEFAULT_LINEAR_ENDPOINT == "https://api.linear.app/graphql"


def test_construction_requires_api_key_or_client() -> None:
    with pytest.raises(TrackerError, match="LINEAR_API_KEY"):
        LinearAdapter(team_id="t-1")


async def test_authorization_header_is_set_without_bearer_prefix() -> None:
    adapter = LinearAdapter(team_id="t-1", api_key="ln-key")  # noqa: S106
    client = adapter._get_client()  # type: ignore[attr-defined]  # noqa: SLF001
    assert client.headers.get("Authorization") == "ln-key"
    await adapter.close()


async def test_close_when_never_used_is_safe() -> None:
    adapter = LinearAdapter(team_id="t-1", api_key="ln-key")  # noqa: S106
    await adapter.close()


# -- list_open_issues -------------------------------------------------------


async def test_list_open_issues_filters_by_team_and_labels() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "id": "linear-issue-1",
                                "identifier": "AGT-1",
                                "url": "https://linear.app/agent-triage/issue/AGT-1",
                                "title": "Existing",
                                "description": "existing body",
                                "state": {"name": "Todo", "type": "unstarted"},
                                "labels": {
                                    "nodes": [
                                        {"name": "agent-triage"},
                                        {"name": "mode:hallucination"},
                                    ]
                                },
                            }
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues(
        filter={"labels": ["agent-triage", "mode:hallucination"]},
    )
    payload = captured["body"]
    variables = payload["variables"]
    f = variables["filter"]
    assert f["team"] == {"id": {"eq": "team-123"}}
    assert f["state"]["type"]["in"] == ["triage", "backlog", "unstarted", "started"]
    label_clauses = f["labels"]["and"]
    assert {"name": {"eq": "agent-triage"}} in label_clauses
    assert {"name": {"eq": "mode:hallucination"}} in label_clauses
    assert len(issues) == 1
    assert issues[0].key == "AGT-1"
    assert issues[0].url == "https://linear.app/agent-triage/issue/AGT-1"
    assert issues[0].body == "existing body"
    assert issues[0].labels == ["agent-triage", "mode:hallucination"]
    assert issues[0].state == "open"


async def test_dedup_finds_existing_issue_in_triage_state() -> None:
    """Regression (RB-6): Linear puts newly created issues in a Triage-type
    state by default. The list filter must include "triage" or the issue this
    tool created on the previous run is invisible to dedup and a duplicate is
    created every run. The handler honors the requested state filter the way
    Linear does: the stored issue is returned only if its state type was asked
    for."""
    from agent_triage.agent.subagents.poster import dedup_drafts
    from agent_triage.models.issue import IssueProvenance, make_labels

    mode_id = "hallucination"
    rubric_version = "agents@1.0.0"
    prov = IssueProvenance(
        rubric_version=rubric_version,
        mode_id=mode_id,
        cluster_id="cl-1",
        representative_trace_id="t-1",
        run_id="run-prior",
        member_trace_ids=["t-1", "t-2"],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        requested_types = (
            body["variables"]["filter"].get("state", {}).get("type", {}).get("in") or []
        )
        label_nodes = [{"name": n} for n in make_labels(mode_id, rubric_version)]
        nodes: list[dict[str, Any]] = []
        if "triage" in requested_types:
            nodes.append(
                {
                    "id": "linear-1",
                    "identifier": "AGT-1",
                    "url": "https://linear.app/agent-triage/issue/AGT-1",
                    "title": "Existing",
                    "description": f"body\n\n{prov.to_html_comment()}",
                    "state": {"name": "Triage", "type": "triage"},
                    "labels": {"nodes": label_nodes},
                }
            )
        return httpx.Response(200, json={"data": {"issues": {"nodes": nodes}}})

    adapter = _make_adapter(handler)
    draft = _draft(member_trace_ids=["t-1", "t-2"], labels=make_labels(mode_id, rubric_version))
    outcomes = await dedup_drafts([draft], tracker=adapter, auto_post_threshold="low")
    # The triage-state issue is visible → skipped, NOT a duplicate create.
    assert outcomes[0].action == "skipped"
    assert outcomes[0].existing_issue is not None
    assert outcomes[0].existing_issue.key == "AGT-1"


def _node(node_id: str, identifier: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "identifier": identifier,
        "url": f"https://linear.app/agent-triage/issue/{identifier}",
        "title": identifier,
        "description": "",
        "state": {"name": "Todo", "type": "unstarted"},
        "labels": {"nodes": []},
    }


async def test_list_open_issues_follows_end_cursor_pagination() -> None:
    cursors_seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        after = body["variables"].get("after")
        cursors_seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "issues": {
                            "nodes": [_node("n-1", "AGT-1")],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cur-2"},
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "nodes": [_node("n-2", "AGT-2")],
                        "pageInfo": {"hasNextPage": False, "endCursor": "cur-3"},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["agent-triage"]})
    assert [i.key for i in issues] == ["AGT-1", "AGT-2"]
    assert cursors_seen == [None, "cur-2"]


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
            json={
                "data": {
                    "issues": {
                        "nodes": [_node(f"n-{calls}", f"AGT-{calls}")],
                        "pageInfo": {"hasNextPage": True, "endCursor": f"cur-{calls}"},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    with caplog.at_level("WARNING", logger="agent_triage.adapters.tracker.linear"):
        issues = await adapter.list_open_issues()
    assert calls == 20  # _MAX_PAGES
    assert len(issues) == 20
    assert any("dedup may miss matches" in r.message for r in caplog.records)


async def test_list_open_issues_returns_empty_when_no_nodes() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"issues": {"nodes": []}}})

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["x"]})
    assert issues == []


async def test_list_open_issues_marks_completed_state_as_closed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "nodes": [
                            {
                                "id": "x",
                                "identifier": "AGT-9",
                                "title": "done one",
                                "description": "",
                                "state": {"name": "Done", "type": "completed"},
                                "labels": {"nodes": []},
                            }
                        ]
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues()
    assert issues[0].state == "closed"


async def test_canceled_state_is_closed_in_both_spellings() -> None:
    """Linear spells the state type "canceled"; accept "cancelled" too."""

    def handler(_request: httpx.Request) -> httpx.Response:
        nodes = [
            {
                "id": f"x-{i}",
                "identifier": f"AGT-{i}",
                "title": "t",
                "description": "",
                "state": {"name": "Canceled", "type": spelling},
                "labels": {"nodes": []},
            }
            for i, spelling in enumerate(["canceled", "cancelled"])
        ]
        return httpx.Response(200, json={"data": {"issues": {"nodes": nodes}}})

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues()
    assert [i.state for i in issues] == ["closed", "closed"]


async def test_list_open_issues_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 403"):
        await adapter.list_open_issues()


async def test_list_open_issues_raises_on_graphql_errors_payload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"errors": [{"message": "bad token"}]},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="GraphQL errors"):
        await adapter.list_open_issues()


async def test_list_open_issues_raises_on_non_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>oops</html>",
            headers={"content-type": "text/html"},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="non-JSON"):
        await adapter.list_open_issues()


async def test_list_open_issues_raises_when_data_object_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": "not-an-object"})

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="missing `data`"):
        await adapter.list_open_issues()


# -- search_issues ----------------------------------------------------------


async def test_search_issues_uses_search_issues_query() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={"data": {"searchIssues": {"nodes": []}}},
        )

    adapter = _make_adapter(handler)
    results = await adapter.search_issues("paris capital", k=3)
    assert results == []
    body = captured["body"]
    assert "searchIssues" in body["query"]
    assert body["variables"]["term"] == "paris capital"
    assert body["variables"]["first"] == 3


# -- create_issue -----------------------------------------------------------


async def test_create_issue_resolves_labels_and_returns_issue() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        calls.append(body)
        query = body["query"]
        if "team(id:" in query and "labels(first:" in query:
            # The label-cache fetch returns 2 known labels; the third will be created.
            return httpx.Response(
                200,
                json={
                    "data": {
                        "team": {
                            "labels": {
                                "nodes": [
                                    {"id": "lbl-1", "name": "agent-triage"},
                                    {"id": "lbl-2", "name": "mode:hallucination"},
                                ]
                            }
                        }
                    }
                },
            )
        if "issueLabelCreate" in query:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "issueLabelCreate": {
                            "success": True,
                            "issueLabel": {"id": "lbl-3", "name": "rubric:agents@1.0.0"},
                        }
                    }
                },
            )
        # The mutation under test.
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "linear-42",
                            "identifier": "AGT-42",
                            "url": "https://linear.app/agent-triage/issue/AGT-42",
                            "title": "Agent hallucinates capitals",
                            "description": "Body of the issue.\n\nSecond paragraph.",
                            "state": {"name": "Triage", "type": "triage"},
                            "labels": {
                                "nodes": [
                                    {"name": "agent-triage"},
                                    {"name": "mode:hallucination"},
                                    {"name": "rubric:agents@1.0.0"},
                                ]
                            },
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft())
    # 3 GraphQL calls: list labels, create missing label, create issue.
    assert len(calls) == 3
    create_call = calls[-1]
    assert create_call["variables"]["input"]["teamId"] == "team-123"
    assert create_call["variables"]["input"]["title"] == "Agent hallucinates capitals"
    assert sorted(create_call["variables"]["input"]["labelIds"]) == ["lbl-1", "lbl-2", "lbl-3"]
    assert issue.id == "linear-42"
    assert issue.key == "AGT-42"
    assert issue.url == "https://linear.app/agent-triage/issue/AGT-42"
    assert issue.title == "Agent hallucinates capitals"
    assert issue.body == "Body of the issue.\n\nSecond paragraph."
    assert sorted(issue.labels) == ["agent-triage", "mode:hallucination", "rubric:agents@1.0.0"]


async def test_create_issue_raises_when_mutation_unsuccessful() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        if "team(id:" in body["query"]:
            return httpx.Response(
                200,
                json={"data": {"team": {"labels": {"nodes": []}}}},
            )
        if "issueLabelCreate" in body["query"]:
            label_name = body["variables"]["input"]["name"]
            return httpx.Response(
                200,
                json={
                    "data": {
                        "issueLabelCreate": {
                            "success": True,
                            "issueLabel": {"id": f"lbl-{label_name}", "name": label_name},
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={"data": {"issueCreate": {"success": False, "issue": None}}},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="issueCreate did not succeed"):
        await adapter.create_issue(_draft())


async def test_create_issue_with_no_labels_skips_label_resolution() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "x",
                            "identifier": "AGT-1",
                            "title": "no labels",
                            "description": "",
                            "state": {"type": "unstarted"},
                            "labels": {"nodes": []},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft(labels=[]))
    assert len(calls) == 1
    assert calls[0]["variables"]["input"]["labelIds"] == []
    assert issue.labels == []


async def test_label_id_cache_avoids_refetch_across_calls() -> None:
    label_fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal label_fetches
        body = json.loads(request.read().decode())
        query = body["query"]
        if "team(id:" in query and "labels(first:" in query:
            label_fetches += 1
            return httpx.Response(
                200,
                json={
                    "data": {
                        "team": {
                            "labels": {
                                "nodes": [
                                    {"id": "lbl-1", "name": "agent-triage"},
                                    {"id": "lbl-2", "name": "mode:hallucination"},
                                    {"id": "lbl-3", "name": "rubric:agents@1.0.0"},
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "x",
                            "identifier": "AGT-1",
                            "title": "t",
                            "description": "b",
                            "state": {"type": "unstarted"},
                            "labels": {"nodes": []},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft())
    await adapter.create_issue(_draft(cluster_id="cl-2"))
    assert label_fetches == 1


# -- retry behavior ----------------------------------------------------------


def _issue_create_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": {
                "issueCreate": {
                    "success": True,
                    "issue": {
                        "id": "linear-1",
                        "identifier": "AGT-1",
                        "url": "https://linear.app/agent-triage/issue/AGT-1",
                        "title": "t",
                        "description": "b",
                        "state": {"name": "Triage", "type": "triage"},
                        "labels": {"nodes": []},
                    },
                }
            }
        },
    )


async def test_list_open_issues_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"data": {"issues": {"nodes": []}}})

    adapter = _make_adapter(handler)
    issues = await adapter.list_open_issues(filter={"labels": ["agent-triage"]})
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
        return _issue_create_response()

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft(labels=[]))
    assert issue.key == "AGT-1"
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
        await adapter.create_issue(_draft(labels=[]))
    assert calls == 1


# -- severity → priority -----------------------------------------------------


async def test_create_issue_maps_p_levels_to_linear_priority_ints() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return _issue_create_response()

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft(labels=[], priority="P1"))
    assert captured["body"]["variables"]["input"]["priority"] == 1


async def test_create_issue_ignores_unmappable_priority() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return _issue_create_response()

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft(labels=[], priority="Sev-1"))
    assert "priority" not in captured["body"]["variables"]["input"]


async def test_create_issue_omits_priority_when_draft_has_none() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return _issue_create_response()

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft(labels=[]))
    assert "priority" not in captured["body"]["variables"]["input"]


# -- update_issue -----------------------------------------------------------


async def test_update_issue_patches_only_provided_fields() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueUpdate": {
                        "success": True,
                        "issue": {
                            "id": "linear-42",
                            "identifier": "AGT-42",
                            "title": "New title",
                            "description": "old body",
                            "state": {"type": "unstarted"},
                            "labels": {"nodes": []},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.update_issue("linear-42", IssuePatch(title="New title"))
    body = captured["body"]
    assert body["variables"]["id"] == "linear-42"
    assert body["variables"]["input"] == {"title": "New title"}
    assert issue.title == "New title"


async def test_update_issue_state_change_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="state transitions are not supported"):
        await adapter.update_issue("linear-42", IssuePatch(state="closed"))


async def test_update_issue_with_empty_patch_refetches() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        calls.append(body["query"])
        return httpx.Response(
            200,
            json={
                "data": {
                    "issue": {
                        "id": "linear-42",
                        "identifier": "AGT-42",
                        "title": "current",
                        "description": "body",
                        "state": {"type": "unstarted"},
                        "labels": {"nodes": []},
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.update_issue("linear-42", IssuePatch())
    assert any("query GetIssue" in q for q in calls)
    assert issue.id == "linear-42"


async def test_update_issue_raises_when_mutation_unsuccessful() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"issueUpdate": {"success": False, "issue": None}}},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="issueUpdate did not succeed"):
        await adapter.update_issue("linear-42", IssuePatch(title="x"))


# -- comment_on_issue -------------------------------------------------------


async def test_comment_on_issue_posts_comment_mutation() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={"data": {"commentCreate": {"success": True, "comment": {"id": "c-1"}}}},
        )

    adapter = _make_adapter(handler)
    await adapter.comment_on_issue("linear-42", "new members:\n\n- t-3")
    body = captured["body"]
    assert body["variables"]["input"]["issueId"] == "linear-42"
    assert "t-3" in body["variables"]["input"]["body"]


async def test_comment_on_issue_raises_when_mutation_unsuccessful() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"commentCreate": {"success": False, "comment": None}}},
        )

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="commentCreate did not succeed"):
        await adapter.comment_on_issue("linear-42", "hi")


async def test_comment_on_issue_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    adapter = _make_adapter(handler)
    with pytest.raises(TrackerError, match="failed with 500"):
        await adapter.comment_on_issue("linear-42", "hi")


def _label_page(nodes: list[dict[str, str]], *, has_next: bool, cursor: str | None) -> Any:
    return {
        "data": {
            "team": {
                "labels": {
                    "nodes": nodes,
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                }
            }
        }
    }


async def test_label_cache_is_case_insensitive() -> None:
    """m-3: Linear label names are unique case-insensitively; match the same way."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        query = body["query"]
        if "labels(first:" in query:
            return httpx.Response(
                200,
                json=_label_page(
                    [{"id": "lbl-1", "name": "Agent-Triage"}], has_next=False, cursor=None
                ),
            )
        if "issueLabelCreate" in query:
            raise AssertionError("label should be matched case-insensitively, not re-created")
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "l-1",
                            "identifier": "AGT-1",
                            "url": "https://linear.app/x/AGT-1",
                            "title": "t",
                            "description": "b",
                            "state": {"name": "Triage", "type": "triage"},
                            "labels": {"nodes": [{"name": "Agent-Triage"}]},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft(labels=["agent-triage"]))
    assert issue.id == "l-1"


async def test_label_cache_paginates() -> None:
    pages = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pages
        body = json.loads(request.read().decode())
        query = body["query"]
        if "labels(first:" in query:
            pages += 1
            if body["variables"].get("after") is None:
                return httpx.Response(
                    200,
                    json=_label_page(
                        [{"id": "lbl-1", "name": "other-label"}], has_next=True, cursor="cur-1"
                    ),
                )
            return httpx.Response(
                200,
                json=_label_page(
                    [{"id": "lbl-2", "name": "agent-triage"}], has_next=False, cursor=None
                ),
            )
        if "issueLabelCreate" in query:
            raise AssertionError("label exists on page 2; must not be re-created")
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "l-2",
                            "identifier": "AGT-2",
                            "url": "https://linear.app/x/AGT-2",
                            "title": "t",
                            "description": "b",
                            "state": {"name": "Triage", "type": "triage"},
                            "labels": {"nodes": [{"name": "agent-triage"}]},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    await adapter.create_issue(_draft(labels=["agent-triage"]))
    assert pages == 2


async def test_label_create_race_falls_back_to_refetch() -> None:
    """A duplicate-name create failure (concurrent run) refetches instead of failing."""
    label_fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal label_fetches
        body = json.loads(request.read().decode())
        query = body["query"]
        if "labels(first:" in query:
            label_fetches += 1
            if label_fetches == 1:
                # First fetch: label not yet visible.
                return httpx.Response(200, json=_label_page([], has_next=False, cursor=None))
            # Refetch after the failed create: another run created it meanwhile.
            return httpx.Response(
                200,
                json=_label_page(
                    [{"id": "lbl-9", "name": "agent-triage"}], has_next=False, cursor=None
                ),
            )
        if "issueLabelCreate" in query:
            return httpx.Response(
                200,
                json={"data": {"issueLabelCreate": {"success": False}}},
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "l-9",
                            "identifier": "AGT-9",
                            "url": "https://linear.app/x/AGT-9",
                            "title": "t",
                            "description": "b",
                            "state": {"name": "Triage", "type": "triage"},
                            "labels": {"nodes": [{"name": "agent-triage"}]},
                        },
                    }
                }
            },
        )

    adapter = _make_adapter(handler)
    issue = await adapter.create_issue(_draft(labels=["agent-triage"]))
    assert issue.id == "l-9"
    assert label_fetches == 2

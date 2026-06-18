"""Linear tracker adapter (design §5.2).

Linear is GraphQL-only — there's no REST surface. The adapter speaks plain
HTTP via `httpx` against `https://api.linear.app/graphql` and shapes its
queries / mutations to the same five `Tracker` methods every other adapter
exposes.

Auth: Linear uses a personal API key passed in the `Authorization` header
*without* a `Bearer` prefix (this is what their docs call out). The key is
read from `LINEAR_API_KEY`.

Scope: every Linear issue belongs to a *team*. The adapter is constructed
with a required `team_id`; `list_open_issues` / `create_issue` are scoped to
that team. Labels are first-class workspace entities (not free strings), so
the adapter resolves label names → IDs at runtime and caches the mapping
across calls.

Markdown: Linear stores issue descriptions and comments as markdown directly
— no ADF conversion. The provenance HTML comment is preserved end-to-end
without any encoding tricks.

State: Linear workflows are workspace-specific. v1.0 of the adapter doesn't
transition state via `update_issue`; `IssuePatch(state=...)` raises so
callers don't think they closed something they didn't.
"""

import json
import logging
from typing import Any, cast

import httpx

from docket.adapters._retry import request_with_retry
from docket.adapters.base import Tracker
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssuePatch, IssueState
from docket.observability import redact

log = logging.getLogger(__name__)

DEFAULT_LINEAR_ENDPOINT = "https://api.linear.app/graphql"

_LIST_ISSUES_QUERY = """
query ListOpenIssues($filter: IssueFilter, $first: Int!, $after: String) {
  issues(filter: $filter, first: $first, after: $after) {
    nodes {
      id
      identifier
      url
      title
      description
      state { name type }
      labels { nodes { name } }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

_SEARCH_ISSUES_QUERY = """
query SearchIssues($term: String!, $first: Int!) {
  searchIssues(term: $term, first: $first) {
    nodes {
      id
      identifier
      url
      title
      description
      state { name type }
      labels { nodes { name } }
    }
  }
}
"""

_GET_ISSUE_QUERY = """
query GetIssue($id: String!) {
  issue(id: $id) {
    id
    identifier
    url
    title
    description
    state { name type }
    labels { nodes { name } }
  }
}
"""

_CREATE_ISSUE_MUTATION = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      url
      title
      description
      state { name type }
      labels { nodes { name } }
    }
  }
}
"""

_UPDATE_ISSUE_MUTATION = """
mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) {
    success
    issue {
      id
      identifier
      url
      title
      description
      state { name type }
      labels { nodes { name } }
    }
  }
}
"""

_COMMENT_CREATE_MUTATION = """
mutation CommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) {
    success
    comment { id }
  }
}
"""

_LIST_LABELS_QUERY = """
query TeamLabels($teamId: String!, $first: Int!, $after: String) {
  team(id: $teamId) {
    labels(first: $first, after: $after) {
      nodes { id name }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_CREATE_LABEL_MUTATION = """
mutation LabelCreate($input: IssueLabelCreateInput!) {
  issueLabelCreate(input: $input) {
    success
    issueLabel { id name }
  }
}
"""

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGES = 20

# P1..P4 → Linear priority ints (1=urgent, 2=high, 3=medium, 4=low).
# Unmappable values are ignored — Linear rejects arbitrary priorities.
_LINEAR_PRIORITY_INTS = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


class LinearAdapter(Tracker):
    def __init__(
        self,
        *,
        team_id: str,
        api_key: str | None = None,
        endpoint: str = DEFAULT_LINEAR_ENDPOINT,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key and client is None:
            raise TrackerError("Linear requires --linear-api-key or LINEAR_API_KEY.")
        self._team_id = team_id
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout = timeout
        self._client = client
        # Workspace labels are entities — cache name→ID once we've fetched them.
        self._label_id_by_name: dict[str, str] | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                # Linear's docs: Authorization: <api_key> with NO "Bearer" prefix.
                headers["Authorization"] = self._api_key
            self._client = httpx.AsyncClient(
                base_url=self._endpoint,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_open_issues(
        self,
        filter: dict[str, Any] | None = None,
    ) -> list[Issue]:
        labels = (filter or {}).get("labels") if filter else None
        graphql_filter: dict[str, Any] = {
            "team": {"id": {"eq": self._team_id}},
            # "triage" is included because Linear puts newly created issues in
            # a Triage-type state by default — omitting it would hide issues
            # this tool just created from the next run's dedup pass.
            "state": {"type": {"in": ["triage", "backlog", "unstarted", "started"]}},
        }
        if labels:
            graphql_filter["labels"] = {
                "and": [{"name": {"eq": name}} for name in labels],
            }
        out: list[Issue] = []
        after: str | None = None
        for _ in range(_MAX_PAGES):
            data = await self._graphql(
                _LIST_ISSUES_QUERY,
                {"filter": graphql_filter, "first": _DEFAULT_PAGE_SIZE, "after": after},
                idempotent=True,
            )
            nodes = _safe_path(data, "issues", "nodes") or []
            out.extend(self._issue_from_node(node) for node in nodes if isinstance(node, dict))
            page_info = _safe_path(data, "issues", "pageInfo") or {}
            end_cursor = page_info.get("endCursor") if isinstance(page_info, dict) else None
            has_next = bool(page_info.get("hasNextPage")) if isinstance(page_info, dict) else False
            if not has_next or not isinstance(end_cursor, str) or not end_cursor:
                return out
            after = end_cursor
        log.warning(
            "Linear list_open_issues hit the %d-page cap with more results pending; "
            "dedup may miss matches beyond %d issues.",
            _MAX_PAGES,
            len(out),
        )
        return out

    async def search_issues(self, query: str, k: int = 10) -> list[Issue]:
        data = await self._graphql(
            _SEARCH_ISSUES_QUERY,
            {"term": query, "first": k},
            idempotent=True,
        )
        nodes = _safe_path(data, "searchIssues", "nodes") or []
        return [self._issue_from_node(node) for node in nodes if isinstance(node, dict)]

    async def create_issue(self, draft: IssueDraft) -> Issue:
        label_ids = await self._resolve_label_ids(draft.labels)
        input_obj: dict[str, Any] = {
            "teamId": self._team_id,
            "title": draft.title,
            "description": draft.body,
            "labelIds": label_ids,
        }
        if draft.priority:
            priority_int = _LINEAR_PRIORITY_INTS.get(draft.priority)
            if priority_int is not None:
                input_obj["priority"] = priority_int
        data = await self._graphql(_CREATE_ISSUE_MUTATION, {"input": input_obj})
        payload = _safe_path(data, "issueCreate") or {}
        if not payload.get("success"):
            raise TrackerError(f"Linear issueCreate did not succeed: {redact(repr(payload))}")
        node = payload.get("issue") or {}
        if not isinstance(node, dict):
            raise TrackerError(f"Linear issueCreate returned no issue: {redact(repr(payload))}")
        return self._issue_from_node(node)

    async def update_issue(self, issue_id: str, patch: IssuePatch) -> Issue:
        if patch.state is not None:
            raise TrackerError(
                f"Linear state transitions are not supported yet (requested "
                f"state={patch.state!r} on {issue_id!r}); patch title/body/labels instead."
            )
        input_obj: dict[str, Any] = {}
        if patch.title is not None:
            input_obj["title"] = patch.title
        if patch.body is not None:
            input_obj["description"] = patch.body
        if patch.labels is not None:
            input_obj["labelIds"] = await self._resolve_label_ids(patch.labels)
        if not input_obj:
            # Refetch and return the current view.
            return await self._fetch_issue(issue_id)
        # Full-replace of the patched fields → safe to retry on 5xx/timeouts.
        data = await self._graphql(
            _UPDATE_ISSUE_MUTATION,
            {"id": issue_id, "input": input_obj},
            idempotent=True,
        )
        payload = _safe_path(data, "issueUpdate") or {}
        if not payload.get("success"):
            raise TrackerError(f"Linear issueUpdate did not succeed: {redact(repr(payload))}")
        node = payload.get("issue") or {}
        if not isinstance(node, dict):
            raise TrackerError(f"Linear issueUpdate returned no issue: {redact(repr(payload))}")
        return self._issue_from_node(node)

    async def comment_on_issue(self, issue_id: str, comment: str) -> None:
        data = await self._graphql(
            _COMMENT_CREATE_MUTATION,
            {"input": {"issueId": issue_id, "body": comment}},
        )
        payload = _safe_path(data, "commentCreate") or {}
        if not payload.get("success"):
            raise TrackerError(f"Linear commentCreate did not succeed: {redact(repr(payload))}")

    async def _fetch_issue(self, issue_id: str) -> Issue:
        data = await self._graphql(_GET_ISSUE_QUERY, {"id": issue_id}, idempotent=True)
        node = data.get("issue")
        if not isinstance(node, dict):
            raise TrackerError(f"Linear get issue {issue_id!r} returned no node")
        return self._issue_from_node(node)

    async def _resolve_label_ids(self, label_names: list[str]) -> list[str]:
        if not label_names:
            return []
        cache = await self._get_label_cache()
        out: list[str] = []
        for name in label_names:
            # Linear label names are unique case-insensitively; match the
            # same way so "Docket" vs "docket" can't fork labels.
            label_id = cache.get(name.lower())
            if label_id is None:
                label_id = await self._create_or_refetch_label(name)
                cache[name.lower()] = label_id
            out.append(label_id)
        return out

    async def _get_label_cache(self) -> dict[str, str]:
        if self._label_id_by_name is not None:
            return self._label_id_by_name
        self._label_id_by_name = await self._fetch_label_cache()
        return self._label_id_by_name

    async def _fetch_label_cache(self) -> dict[str, str]:
        """Fetch all team labels (cursor-paginated), keyed by lowercase name."""
        cache: dict[str, str] = {}
        after: str | None = None
        for _ in range(_MAX_PAGES):
            data = await self._graphql(
                _LIST_LABELS_QUERY,
                {"teamId": self._team_id, "first": _DEFAULT_PAGE_SIZE, "after": after},
                idempotent=True,
            )
            nodes = _safe_path(data, "team", "labels", "nodes") or []
            for node in nodes:
                if isinstance(node, dict):
                    name = node.get("name")
                    node_id = node.get("id")
                    if isinstance(name, str) and isinstance(node_id, str):
                        cache[name.lower()] = node_id
            page_info = _safe_path(data, "team", "labels", "pageInfo") or {}
            has_next = bool(page_info.get("hasNextPage")) if isinstance(page_info, dict) else False
            cursor = page_info.get("endCursor") if isinstance(page_info, dict) else None
            if not has_next or not isinstance(cursor, str):
                return cache
            after = cursor
        log.warning(
            "Linear label listing hit the %d-page cap; labels beyond the cap "
            "may be re-created instead of reused",
            _MAX_PAGES,
        )
        return cache

    async def _create_or_refetch_label(self, name: str) -> str:
        """Create a label, falling back to a cache refresh on failure.

        A concurrent run (or a label created since the cache was built) makes
        `issueLabelCreate` fail with a duplicate-name error; the label exists,
        so refresh the cache once and use it instead of failing the draft.
        """
        try:
            data = await self._graphql(
                _CREATE_LABEL_MUTATION,
                {"input": {"name": name, "teamId": self._team_id}},
            )
        except TrackerError:
            refreshed = await self._refetch_label(name)
            if refreshed is not None:
                return refreshed
            raise
        payload = _safe_path(data, "issueLabelCreate") or {}
        if not payload.get("success"):
            refreshed = await self._refetch_label(name)
            if refreshed is not None:
                return refreshed
            raise TrackerError(
                f"Linear issueLabelCreate failed for {name!r}: {redact(repr(payload))}"
            )
        node = payload.get("issueLabel") or {}
        label_id = node.get("id") if isinstance(node, dict) else None
        if not isinstance(label_id, str):
            raise TrackerError(f"Linear issueLabelCreate returned no id for {name!r}")
        return label_id

    async def _refetch_label(self, name: str) -> str | None:
        self._label_id_by_name = await self._fetch_label_cache()
        return self._label_id_by_name.get(name.lower())

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any],
        *,
        idempotent: bool = False,
    ) -> dict[str, Any]:
        response = await request_with_retry(
            self._get_client(),
            "POST",
            "",
            error_cls=TrackerError,
            idempotent=idempotent,
            json_body={"query": query, "variables": variables},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"Linear GraphQL failed with {response.status_code}: {redact(response.text)}"
            )
        try:
            parsed = response.json()
        except json.JSONDecodeError as e:
            raise TrackerError(f"Linear GraphQL returned non-JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise TrackerError(f"Linear GraphQL returned non-object: {type(parsed).__name__}")
        errors = parsed.get("errors")
        if errors:
            raise TrackerError(f"Linear GraphQL errors: {redact(repr(errors))}")
        data = parsed.get("data")
        if not isinstance(data, dict):
            raise TrackerError("Linear GraphQL response missing `data` object")
        return cast(dict[str, Any], data)

    def _issue_from_node(self, node: dict[str, Any]) -> Issue:
        identifier = str(node.get("identifier") or "")
        node_id = str(node.get("id") or "")
        url = node.get("url")
        if not isinstance(url, str):
            url = None
        labels_raw = _safe_path(node, "labels", "nodes") or []
        labels = [
            n["name"] for n in labels_raw if isinstance(n, dict) and isinstance(n.get("name"), str)
        ]
        state_type = _safe_path(node, "state", "type") or ""
        state: IssueState = (
            "closed"
            if str(state_type).lower()
            in (
                "completed",
                # Linear spells it "canceled"; accept both spellings.
                "canceled",
                "cancelled",
            )
            else "open"
        )
        return Issue(
            id=node_id,
            key=identifier or None,
            url=url,
            title=str(node.get("title") or ""),
            body=str(node.get("description") or ""),
            labels=labels,
            state=state,
        )


def _safe_path(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

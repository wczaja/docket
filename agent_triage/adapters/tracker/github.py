"""GitHub Issues tracker adapter (design §5.2).

Unlike Jira (REST + ADF body + Basic auth) and Linear (GraphQL + label
entities + workspace state machines), GitHub Issues is the simplest of the
three surfaces:

  - REST v3 against `https://api.github.com` with `Authorization: Bearer
    <PAT>` (classic or fine-grained — both work the same for issues).
  - Markdown bodies are stored verbatim, so the HTML provenance comment
    survives end-to-end without conversion.
  - Labels are free strings; GitHub creates any that don't exist.
  - State is a first-class `open|closed` value on the issue and can be
    transitioned via the standard `PATCH /issues/{number}` endpoint, so
    unlike Jira and Linear we DO support `IssuePatch(state=...)`.

Endpoints used:

  - GET  /repos/{owner}/{repo}/issues   (list, filtered by labels + state)
  - POST /repos/{owner}/{repo}/issues   (create)
  - PATCH /repos/{owner}/{repo}/issues/{number}    (update)
  - POST /repos/{owner}/{repo}/issues/{number}/comments    (comment)
  - GET  /search/issues   (free-text search across the repo)

Pagination follows the standard `Link: <url>; rel="next"` header.
"""

import json
import logging
import re
from typing import Any, cast

import httpx

from agent_triage.adapters._retry import request_with_retry
from agent_triage.adapters.base import Tracker
from agent_triage.errors import TrackerError
from agent_triage.models.issue import Issue, IssueDraft, IssuePatch, IssueState
from agent_triage.observability import redact

log = logging.getLogger(__name__)

DEFAULT_GITHUB_API = "https://api.github.com"

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGES = 20
_LINK_NEXT_RE = re.compile(r'<(?P<url>[^>]+)>;\s*rel="next"')


class GitHubAdapter(Tracker):
    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        token: str | None = None,
        api_url: str = DEFAULT_GITHUB_API,
        user_agent: str = "agent-triage",
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not token and client is None:
            raise TrackerError("GitHub tracker requires --github-token or GITHUB_TOKEN.")
        self._owner = owner
        self._repo = repo
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._user_agent = user_agent
        self._timeout = timeout
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": self._user_agent,
            }
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def repo_path(self) -> str:
        return f"{self._owner}/{self._repo}"

    async def _request(
        self,
        method: str,
        url: str,
        *,
        idempotent: bool,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        return await request_with_retry(
            self._get_client(),
            method,
            url,
            error_cls=TrackerError,
            idempotent=idempotent,
            json_body=json_body,
            params=params,
        )

    async def list_open_issues(
        self,
        filter: dict[str, Any] | None = None,
    ) -> list[Issue]:
        labels = (filter or {}).get("labels") if filter else None
        params: dict[str, str | int] = {
            "state": "open",
            "per_page": _DEFAULT_PAGE_SIZE,
        }
        if labels:
            params["labels"] = ",".join(labels)
        url: str | None = f"/repos/{self.repo_path}/issues"
        out: list[Issue] = []
        pages = 0
        while url and pages < _MAX_PAGES:
            response = await self._request(
                "GET", url, idempotent=True, params=params if pages == 0 else None
            )
            if response.status_code >= 400:
                raise TrackerError(
                    f"GitHub list_open_issues failed with {response.status_code}: "
                    f"{redact(response.text)}"
                )
            data = _parse_json_list(response, context="list_open_issues")
            for raw in data:
                if isinstance(raw, dict) and "pull_request" not in raw:
                    out.append(self._issue_from_github(raw))
            url = _parse_link_next(response.headers.get("Link"))
            pages += 1
        if url:
            log.warning(
                "GitHub list_open_issues hit the %d-page cap with more results pending; "
                "dedup may miss matches beyond %d issues.",
                _MAX_PAGES,
                len(out),
            )
        return out

    async def search_issues(self, query: str, k: int = 10) -> list[Issue]:
        # Scope the search to this repo + open issues. The user query is
        # quoted (embedded quotes escaped) so qualifier-like tokens inside it
        # cannot rewrite the search scope.
        quoted = '"' + query.replace('"', '\\"') + '"'
        q = f"repo:{self.repo_path} is:issue state:open {quoted}"
        response = await self._request(
            "GET",
            "/search/issues",
            idempotent=True,
            params={"q": q, "per_page": k},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"GitHub search_issues failed with {response.status_code}: {redact(response.text)}"
            )
        data = _parse_json_dict(response, context="search_issues")
        items = data.get("items") or []
        if not isinstance(items, list):
            raise TrackerError(f"GitHub search returned non-list `items`: {type(items).__name__}")
        return [self._issue_from_github(i) for i in items if isinstance(i, dict)]

    async def create_issue(self, draft: IssueDraft) -> Issue:
        # GitHub Issues has no priority field; severity context rides on the
        # labels, so `draft.priority` is intentionally skipped here.
        body = {
            "title": draft.title,
            "body": draft.body,
            "labels": list(draft.labels),
        }
        response = await self._request(
            "POST",
            f"/repos/{self.repo_path}/issues",
            idempotent=False,
            json_body=body,
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"GitHub create_issue failed with {response.status_code}: {redact(response.text)}"
            )
        return self._issue_from_github(_parse_json_dict(response, context="create_issue"))

    async def update_issue(self, issue_id: str, patch: IssuePatch) -> Issue:
        body: dict[str, Any] = {}
        if patch.title is not None:
            body["title"] = patch.title
        if patch.body is not None:
            body["body"] = patch.body
        if patch.labels is not None:
            body["labels"] = list(patch.labels)
        if patch.state is not None:
            # GitHub uses 'open' | 'closed' directly — no workflow mapping needed.
            body["state"] = patch.state
        if not body:
            return await self._fetch_issue(issue_id)
        # Full-replace of the patched fields → safe to retry on 5xx/timeouts.
        response = await self._request(
            "PATCH",
            f"/repos/{self.repo_path}/issues/{issue_id}",
            idempotent=True,
            json_body=body,
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"GitHub update_issue failed with {response.status_code}: {redact(response.text)}"
            )
        return self._issue_from_github(_parse_json_dict(response, context="update_issue"))

    async def comment_on_issue(self, issue_id: str, comment: str) -> None:
        response = await self._request(
            "POST",
            f"/repos/{self.repo_path}/issues/{issue_id}/comments",
            idempotent=False,
            json_body={"body": comment},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"GitHub comment_on_issue failed with {response.status_code}: "
                f"{redact(response.text)}"
            )

    async def _fetch_issue(self, issue_id: str) -> Issue:
        response = await self._request(
            "GET", f"/repos/{self.repo_path}/issues/{issue_id}", idempotent=True
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"GitHub get issue {issue_id!r} failed with {response.status_code}: "
                f"{redact(response.text)}"
            )
        return self._issue_from_github(_parse_json_dict(response, context="get_issue"))

    def _issue_from_github(self, raw: dict[str, Any]) -> Issue:
        # `number` is the per-repo identifier (`#42`); `id` is the global ID.
        number = raw.get("number")
        global_id = raw.get("id")
        key = str(number) if number is not None else None
        labels_raw = raw.get("labels") or []
        labels: list[str] = []
        for label in labels_raw:
            if isinstance(label, str):
                labels.append(label)
            elif isinstance(label, dict):
                name = label.get("name")
                if isinstance(name, str):
                    labels.append(name)
        state_raw = str(raw.get("state") or "open").lower()
        state: IssueState = "closed" if state_raw == "closed" else "open"
        return Issue(
            id=str(number) if number is not None else str(global_id or ""),
            key=key,
            url=raw.get("html_url") if isinstance(raw.get("html_url"), str) else None,
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
            labels=labels,
            state=state,
        )


def _parse_json_dict(response: httpx.Response, *, context: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as e:
        raise TrackerError(f"GitHub {context} returned non-JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise TrackerError(f"GitHub {context} returned non-object: {type(parsed).__name__}")
    return cast(dict[str, Any], parsed)


def _parse_json_list(response: httpx.Response, *, context: str) -> list[Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as e:
        raise TrackerError(f"GitHub {context} returned non-JSON: {e}") from e
    if not isinstance(parsed, list):
        raise TrackerError(f"GitHub {context} returned non-list: {type(parsed).__name__}")
    return parsed


def _parse_link_next(link_header: str | None) -> str | None:
    """Extract the `rel="next"` URL from a GitHub Link header."""
    if not link_header:
        return None
    match = _LINK_NEXT_RE.search(link_header)
    if not match:
        return None
    return match.group("url")

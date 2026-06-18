"""Jira tracker adapter (design §5.2).

Supports both Atlassian Cloud (REST API v3 with Atlassian Document Format
bodies) and self-hosted Data Center / Server (REST API v2 with plain text /
wiki markup bodies). The deployment mode is selected explicitly via the
`deployment` kwarg, or auto-detected from the host: `*.atlassian.net` →
`cloud`, anything else → `datacenter`.

Auth:

  - Cloud uses HTTP Basic with `(email, api_token)` per Atlassian's docs
    (https://developer.atlassian.com/cloud/jira/platform/basic-auth-for-rest-apis/).
  - Data Center uses a Personal Access Token (PAT) as `Authorization: Bearer
    <token>`. Older "username + password" Basic auth is intentionally not
    supported in v1.0 — modern DC supports PATs and they're safer.

Body encoding:

  - Cloud v3 requires ADF (Atlassian Document Format), a JSON document tree.
    We render the draft body as a flat sequence of ADF paragraphs — one per
    blank-line-separated chunk. Markdown styling (bullets, code fences,
    tables) does NOT survive this transform in v1.0; it lands as plain text
    inside paragraphs. The HTML provenance comment is one of the paragraphs.
    A proper md→ADF transform is a v1.1 follow-up.
  - Data Center v2 accepts plain text via `description` directly, which
    renders as wiki markup. We pass the body through unchanged.

Endpoints (Cloud / DC):

  - GET  /rest/api/3/search/jql        (JQL list / dedup — Cloud; Atlassian
    removed `GET /rest/api/3/search` from Cloud in 2025; the replacement
    paginates via `nextPageToken`, omitted on the last page)
  - GET  /rest/api/2/search            (JQL list / dedup — Data Center,
    classic `startAt`/`total` pagination)
  - POST /rest/api/{3|2}/issue         (create)
  - PUT  /rest/api/{3|2}/issue/{id}    (update)
  - POST /rest/api/{3|2}/issue/{id}/comment (comment)
"""

import base64
import json
import logging
from typing import Any, Literal, cast

import httpx

from docket.adapters._retry import request_with_retry
from docket.adapters.base import Tracker
from docket.errors import TrackerError
from docket.models.issue import Issue, IssueDraft, IssuePatch, IssueState
from docket.observability import redact

log = logging.getLogger(__name__)

Deployment = Literal["cloud", "datacenter"]

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGES = 20

# P1..P4 → Jira's default priority scheme names. Any other non-empty draft
# priority is passed through verbatim as the priority name.
_JIRA_PRIORITY_NAMES = {
    "P1": "Highest",
    "P2": "High",
    "P3": "Medium",
    "P4": "Low",
}


class JiraAdapter(Tracker):
    def __init__(
        self,
        host: str,
        *,
        project: str,
        email: str | None = None,
        api_token: str | None = None,
        pat: str | None = None,
        deployment: Deployment | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._project = project
        self._email = email
        self._api_token = api_token
        self._pat = pat
        self._deployment: Deployment = deployment or _detect_deployment(self._host)
        self._timeout = timeout
        self._client = client
        self._validate_auth()

    @property
    def deployment(self) -> Deployment:
        return self._deployment

    @property
    def api_version(self) -> str:
        return "3" if self._deployment == "cloud" else "2"

    def _validate_auth(self) -> None:
        if self._deployment == "cloud":
            if not (self._email and self._api_token):
                raise TrackerError(
                    "Jira Cloud requires both email and api_token "
                    "(set --jira-email + --jira-api-token, or JIRA_EMAIL + JIRA_API_TOKEN)."
                )
        elif not self._pat:
            raise TrackerError(
                "Jira Data Center requires a Personal Access Token (set --jira-pat or JIRA_PAT)."
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Accept": "application/json"}
            if self._deployment == "cloud":
                token = base64.b64encode(f"{self._email}:{self._api_token}".encode()).decode(
                    "ascii"
                )
                headers["Authorization"] = f"Basic {token}"
            else:
                headers["Authorization"] = f"Bearer {self._pat}"
            self._client = httpx.AsyncClient(
                base_url=self._host,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
        jql = self._build_jql(labels=labels, only_open=True)
        return await self._search(jql, max_results=None)

    async def search_issues(self, query: str, k: int = 10) -> list[Issue]:
        # Treat `query` as a free-text search; combine with project scope.
        jql = f'project = "{_escape_jql(self._project)}" AND text ~ "{_escape_jql(query)}"'
        return await self._search(jql, max_results=k)

    async def create_issue(self, draft: IssueDraft) -> Issue:
        fields: dict[str, Any] = {
            "project": {"key": self._project},
            "summary": draft.title,
            "issuetype": {"name": "Task"},
            "description": self._encode_body(draft.body),
            "labels": list(draft.labels),
        }
        if draft.priority:
            fields["priority"] = {"name": _JIRA_PRIORITY_NAMES.get(draft.priority, draft.priority)}
        response = await self._request(
            "POST",
            f"/rest/api/{self.api_version}/issue",
            idempotent=False,
            json_body={"fields": fields},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"Jira create_issue failed with {response.status_code}: {redact(response.text)}"
            )
        data = _parse_json(response, context="create_issue")
        issue_id = str(data.get("id") or "")
        issue_key = str(data.get("key") or "")
        if not issue_id or not issue_key:
            raise TrackerError(f"Jira create_issue returned no id/key: {redact(repr(data))}")
        return Issue(
            id=issue_id,
            key=issue_key,
            url=f"{self._host}/browse/{issue_key}",
            title=draft.title,
            body=draft.body,
            labels=list(draft.labels),
            state="open",
        )

    async def update_issue(self, issue_id: str, patch: IssuePatch) -> Issue:
        fields: dict[str, Any] = {}
        if patch.title is not None:
            fields["summary"] = patch.title
        if patch.body is not None:
            fields["description"] = self._encode_body(patch.body)
        if patch.labels is not None:
            fields["labels"] = list(patch.labels)
        if fields:
            # Full-replace update → safe to retry on 5xx/timeouts.
            response = await self._request(
                "PUT",
                f"/rest/api/{self.api_version}/issue/{issue_id}",
                idempotent=True,
                json_body={"fields": fields},
            )
            if response.status_code >= 400:
                raise TrackerError(
                    f"Jira update_issue failed with {response.status_code}: {redact(response.text)}"
                )
        if patch.state is not None:
            await self._transition_state(issue_id, patch.state)
        # Refetch to surface the canonical post-update view.
        return await self._get_issue(issue_id)

    async def comment_on_issue(self, issue_id: str, comment: str) -> None:
        response = await self._request(
            "POST",
            f"/rest/api/{self.api_version}/issue/{issue_id}/comment",
            idempotent=False,
            json_body={"body": self._encode_body(comment)},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"Jira comment_on_issue failed with {response.status_code}: {redact(response.text)}"
            )

    async def _search(self, jql: str, *, max_results: int | None) -> list[Issue]:
        if self._deployment == "cloud":
            return await self._search_cloud(jql, max_results=max_results)
        return await self._search_datacenter(jql, max_results=max_results)

    def _page_size(self, found: int, max_results: int | None) -> int:
        if max_results is None:
            return _DEFAULT_PAGE_SIZE
        return min(_DEFAULT_PAGE_SIZE, max_results - found)

    async def _search_cloud(self, jql: str, *, max_results: int | None) -> list[Issue]:
        """Cloud search via `GET /rest/api/3/search/jql` (nextPageToken pages).

        Atlassian removed `GET /rest/api/3/search` from Jira Cloud in 2025;
        the replacement omits `nextPageToken` on the last page.
        """
        out: list[Issue] = []
        next_page_token: str | None = None
        for _ in range(_MAX_PAGES):
            params: dict[str, str | int] = {
                "jql": jql,
                "maxResults": self._page_size(len(out), max_results),
                "fields": "summary,description,labels,status",
            }
            if next_page_token is not None:
                params["nextPageToken"] = next_page_token
            data = await self._search_page("/rest/api/3/search/jql", params)
            for raw in data.get("issues") or []:
                if isinstance(raw, dict):
                    out.append(self._issue_from_jira(raw))
                    if max_results is not None and len(out) >= max_results:
                        return out
            token = data.get("nextPageToken")
            if not isinstance(token, str) or not token:
                return out
            next_page_token = token
        log.warning(
            "Jira search hit the %d-page cap with more results pending; "
            "dedup may miss matches beyond %d issues.",
            _MAX_PAGES,
            len(out),
        )
        return out

    async def _search_datacenter(self, jql: str, *, max_results: int | None) -> list[Issue]:
        """Data Center search via classic `GET /rest/api/2/search` (startAt/total)."""
        out: list[Issue] = []
        start_at = 0
        for _ in range(_MAX_PAGES):
            params: dict[str, str | int] = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": self._page_size(len(out), max_results),
                "fields": "summary,description,labels,status",
            }
            data = await self._search_page(f"/rest/api/{self.api_version}/search", params)
            issues = data.get("issues") or []
            for raw in issues:
                if isinstance(raw, dict):
                    out.append(self._issue_from_jira(raw))
                    if max_results is not None and len(out) >= max_results:
                        return out
            total = data.get("total")
            if not isinstance(total, int) or start_at + len(issues) >= total:
                return out
            start_at += len(issues)
        log.warning(
            "Jira search hit the %d-page cap with more results pending; "
            "dedup may miss matches beyond %d issues.",
            _MAX_PAGES,
            len(out),
        )
        return out

    async def _search_page(self, path: str, params: dict[str, str | int]) -> dict[str, Any]:
        response = await self._request("GET", path, idempotent=True, params=dict(params))
        if response.status_code >= 400:
            raise TrackerError(
                f"Jira search failed with {response.status_code}: {redact(response.text)}"
            )
        data = _parse_json(response, context="search")
        issues = data.get("issues")
        if issues is not None and not isinstance(issues, list):
            raise TrackerError(f"Jira search returned non-list `issues`: {type(issues).__name__}")
        return data

    async def _get_issue(self, issue_id: str) -> Issue:
        response = await self._request(
            "GET",
            f"/rest/api/{self.api_version}/issue/{issue_id}",
            idempotent=True,
            params={"fields": "summary,description,labels,status"},
        )
        if response.status_code >= 400:
            raise TrackerError(
                f"Jira get_issue failed with {response.status_code}: {redact(response.text)}"
            )
        return self._issue_from_jira(_parse_json(response, context="get_issue"))

    async def _transition_state(self, issue_id: str, state: IssueState) -> None:
        # Jira workflows are project-specific; we don't try to match states by
        # name here. v1.0 only documents the "close on update" path as a
        # follow-up — for now, raise so callers don't think state changed.
        raise TrackerError(
            f"Jira state transitions are not supported yet (requested state={state!r} "
            f"on {issue_id!r}); patch title/body/labels instead."
        )

    def _build_jql(
        self,
        *,
        labels: list[str] | None,
        only_open: bool,
    ) -> str:
        parts = [f'project = "{_escape_jql(self._project)}"']
        if only_open:
            parts.append("resolution = Unresolved")
        if labels:
            for label in labels:
                parts.append(f'labels = "{_escape_jql(label)}"')
        return " AND ".join(parts)

    def _encode_body(self, markdown_body: str) -> Any:
        if self._deployment == "cloud":
            return _markdown_to_adf(markdown_body)
        return markdown_body

    def _issue_from_jira(self, raw: dict[str, Any]) -> Issue:
        issue_id = str(raw.get("id") or "")
        issue_key = str(raw.get("key") or "")
        fields = raw.get("fields") or {}
        if not isinstance(fields, dict):
            fields = {}
        title = str(fields.get("summary") or "")
        body = (
            _adf_to_text(fields.get("description"))
            if self._deployment == "cloud"
            else str(fields.get("description") or "")
        )
        labels_raw = fields.get("labels") or []
        labels = [str(label) for label in labels_raw if isinstance(label, str)]
        status_obj = fields.get("status") or {}
        status_name = ""
        category_key = ""
        if isinstance(status_obj, dict):
            status_name = str(status_obj.get("name") or "").lower()
            category_obj = status_obj.get("statusCategory")
            if isinstance(category_obj, dict):
                category_key = str(category_obj.get("key") or "").lower()
        # statusCategory.key is locale-independent ("new"/"indeterminate"/
        # "done"); prefer it so localized or custom workflow names still map
        # closed issues to closed. Name matching remains the fallback for
        # payloads that omit the category.
        if category_key:
            state: IssueState = "closed" if category_key == "done" else "open"
        else:
            state = "closed" if status_name in ("done", "closed", "resolved") else "open"
        return Issue(
            id=issue_id,
            key=issue_key,
            url=f"{self._host}/browse/{issue_key}" if issue_key else None,
            title=title,
            body=body,
            labels=labels,
            state=state,
        )


def _detect_deployment(host: str) -> Deployment:
    lowered = host.lower()
    if ".atlassian.net" in lowered or ".jira.com" in lowered:
        return "cloud"
    return "datacenter"


def _escape_jql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parse_json(response: httpx.Response, *, context: str) -> dict[str, Any]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as e:
        raise TrackerError(f"Jira {context} returned non-JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise TrackerError(f"Jira {context} returned non-object: {type(parsed).__name__}")
    return cast(dict[str, Any], parsed)


def _markdown_to_adf(body: str) -> dict[str, Any]:
    """Minimal markdown → ADF (Atlassian Document Format) converter.

    Splits the body on blank lines and emits one ADF paragraph per chunk.
    Inline markdown (bold, code, links) does NOT survive in v1.0 — chunks
    land as plain text. The HTML provenance comment becomes its own
    paragraph. v1.1 may swap in a real parser.
    """
    chunks = [c.strip() for c in body.split("\n\n") if c.strip()]
    content: list[dict[str, Any]] = []
    for chunk in chunks:
        content.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": chunk}],
            }
        )
    if not content:
        content.append({"type": "paragraph", "content": []})
    return {"type": "doc", "version": 1, "content": content}


def _adf_to_text(adf: Any) -> str:
    """Extract a best-effort plain-text rendering of an ADF document.

    Used for `Issue.body` round-trips out of Jira Cloud. Concatenates each
    paragraph's text nodes with double newlines so a provenance comment
    embedded in the body survives `IssueProvenance.parse_from_body`.
    """
    if adf is None:
        return ""
    if isinstance(adf, str):
        return adf
    if not isinstance(adf, dict):
        return ""
    paragraphs: list[str] = []
    for block in adf.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        text_parts: list[str] = []
        for node in block.get("content", []) or []:
            if isinstance(node, dict) and node.get("type") == "text":
                text = node.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            paragraphs.append("".join(text_parts))
    return "\n\n".join(paragraphs)

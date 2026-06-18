"""Adapter abstract base classes (design §5.1 + §5.2)."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from docket.models.classification import Annotation
from docket.models.issue import Issue, IssueDraft, IssuePatch
from docket.models.trace import OpenInferenceTrace


class TraceBackend(ABC):
    """Read + annotate interface implemented by each trace-backend adapter."""

    @abstractmethod
    async def list_traces(
        self,
        since: datetime,
        until: datetime | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return trace IDs in the `[since, until]` window.

        `filter` is backend-specific; adapters MAY ignore it or document the
        supported keys.
        """

    @abstractmethod
    async def get_trace(self, trace_id: str) -> OpenInferenceTrace:
        """Fetch a single trace by ID."""

    @abstractmethod
    async def annotate_trace(self, trace_id: str, annotation: Annotation) -> None:
        """Write `annotation` to the backend.

        Backends MUST upsert by `annotation.idempotency_key()` so that
        re-running the same `(run_id, rubric_version, mode_id)` against the
        same trace overwrites rather than duplicates.
        """

    @abstractmethod
    async def search_traces(self, query: str, k: int = 10) -> list[str]:
        """Semantic search where the backend supports it.

        Adapters without semantic search MUST raise `NotImplementedError`
        with a clear message rather than silently returning an empty list.
        """

    @abstractmethod
    async def mark_trace_processed(
        self,
        trace_id: str,
        *,
        run_id: str,
        rubric_version: str,
    ) -> None:
        """Write a sentinel annotation marking this trace as fully classified
        under `run_id`. Called once per trace after all modes finish (positive
        or negative). The pipeline reads these sentinels back on the next run
        to skip already-completed traces — backend annotations are the
        runtime's checkpoint per design §2.

        Backends MUST upsert (re-marking the same trace under the same run_id
        is a no-op). Adapters that can't write to the backend SHOULD raise
        a clear BackendError rather than silently swallowing.
        """

    @abstractmethod
    async def list_processed_trace_ids(
        self,
        *,
        run_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> set[str]:
        """Return the set of trace IDs that have a `mark_trace_processed`
        sentinel for `run_id` within the time window. The pipeline subtracts
        this set from `list_traces(since, until)` before classifying.

        Empty set is the correct return for a never-before-seen run_id;
        adapters MUST NOT raise just because no matching annotations exist.
        """

    async def close(self) -> None:  # noqa: B027  -- intentional no-op default
        """Release any held resources (HTTP clients, subprocesses, etc.).

        Default implementation is a no-op. Adapters that hold long-lived
        resources MUST override.
        """


class Tracker(ABC):
    """Read + write interface implemented by each issue-tracker adapter.

    Per design §5.2 the contract is five operations: list / search / create /
    update / comment. The pipeline's dedup loop calls `list_open_issues` with
    the standard label set (`docket`, `mode:<id>`, `rubric:<id>@<ver>`)
    to decide whether to create a new issue or comment on an existing one.
    """

    @abstractmethod
    async def list_open_issues(
        self,
        filter: dict[str, Any] | None = None,
    ) -> list[Issue]:
        """Return open issues, optionally filtered.

        Filter keys are tracker-specific but MUST include a `labels` key
        meaning "all of these labels are present"; this is the dedup loop's
        sole requirement on the backend.
        """

    @abstractmethod
    async def search_issues(self, query: str, k: int = 10) -> list[Issue]:
        """Free-text / structured search over issues.

        Trackers without a search surface MUST raise `NotImplementedError`
        with a clear message rather than silently returning an empty list.
        """

    @abstractmethod
    async def create_issue(self, draft: IssueDraft) -> Issue:
        """Create a new issue from a draft and return the tracker's view of it."""

    @abstractmethod
    async def update_issue(self, issue_id: str, patch: IssuePatch) -> Issue:
        """Apply a partial update to an existing issue."""

    @abstractmethod
    async def comment_on_issue(self, issue_id: str, comment: str) -> None:
        """Post a comment on an existing issue."""

    async def close(self) -> None:  # noqa: B027  -- intentional no-op default
        """Release any held resources. Default is a no-op."""

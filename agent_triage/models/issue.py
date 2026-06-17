"""Issue value types (draft + patch + provenance + tracker-side view).

Per design §5.2 every draft carries provenance in two places: an HTML-comment
block at the end of the body (machine-parseable, human-invisible) and tracker
labels (queryable without parsing). The provenance enables idempotent dedup
across runs — if the drafter sees an existing open issue with the same
`mode:<id>` + `rubric:<id>@<version>` labels, it comments instead of creating
a duplicate (Phase 8).
"""

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_triage.models.classification import Severity

PROVENANCE_TAG = "agent-triage:provenance"
# Match the whole comment block (everything up to the closing `-->`) rather
# than a brace-balanced JSON object: a `}` inside a JSON string value must not
# end the match early. The JSON payload is sliced out of the segment below.
_PROVENANCE_RE = re.compile(
    r"<!--\s*" + re.escape(PROVENANCE_TAG) + r"(?P<segment>.*?)-->",
    re.DOTALL,
)

# Keeps serialized provenance (and thus issue bodies) under tracker size
# limits; dedup overlap-matching still works on the first 100 members.
_PROVENANCE_MEMBER_CAP = 100

IssueState = Literal["open", "closed"]


class IssueProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    rubric_version: str
    mode_id: str
    cluster_id: str
    representative_trace_id: str
    run_id: str
    member_trace_ids: list[str] = Field(default_factory=list)

    def to_html_comment(self) -> str:
        data: dict[str, Any] = self.model_dump()
        members = data.get("member_trace_ids") or []
        if len(members) > _PROVENANCE_MEMBER_CAP:
            data["member_trace_ids"] = members[:_PROVENANCE_MEMBER_CAP]
            data["member_trace_ids_truncated"] = True
        payload = json.dumps(data, sort_keys=True)
        return f"<!-- {PROVENANCE_TAG} {payload} -->"

    @classmethod
    def parse_from_body(cls, body: str) -> "IssueProvenance | None":
        """Extract the embedded provenance block from an issue body, if present.

        Returns None when the body has no recognizable provenance comment.
        Used by the dedup loop to compare an existing tracker issue's cluster
        membership against the current cluster (design §5.2).

        The LAST provenance comment in the body wins: a drafted body that
        quotes an example block earlier must not shadow the real trailing one.
        """
        matches = list(_PROVENANCE_RE.finditer(body))
        if not matches:
            return None
        segment = matches[-1].group("segment")
        start = segment.find("{")
        end = segment.rfind("}")
        if start == -1 or end < start:
            return None
        try:
            payload = json.loads(segment[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return cls.model_validate(payload)
        except ValueError:
            return None


class IssueDraft(BaseModel):
    """A drafted issue ready to write to a tracker (or queue for review)."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    mode_id: str
    rubric_version: str
    run_id: str
    severity: Severity
    representative_trace_id: str
    member_trace_ids: list[str]
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    # Tracker priority mapped from the rubric's
    # `triage.default_severity_to_tracker` (e.g. "P1"); None when the rubric
    # defines no mapping. Adapters translate per-tracker (or skip).
    priority: str | None = None

    def to_markdown(self) -> str:
        """Render a human-readable markdown file for the local-file queue."""
        lines = [
            f"# {self.title}",
            "",
            f"**Severity**: {self.severity}",
            f"**Mode**: `{self.mode_id}`",
            f"**Rubric**: `{self.rubric_version}`",
            f"**Cluster**: `{self.cluster_id}`",
            f"**Representative trace**: `{self.representative_trace_id}`",
            f"**Member trace count**: {len(self.member_trace_ids)}",
            f"**Labels**: {', '.join(self.labels) if self.labels else '(none)'}",
            "",
            "## Description",
            "",
            self.body,
            "",
            "## Member traces",
            "",
            *[f"- `{tid}`" for tid in self.member_trace_ids],
        ]
        return "\n".join(lines)

    def to_json_record(self) -> dict[str, Any]:
        """Tracker-ready dict for the local-file queue."""
        return self.model_dump()


# GitHub caps label names at 50 chars; Jira rejects spaces. `_normalize_label`
# makes every built label safe for all three trackers, deterministically, so
# label creation and the dedup query always agree.
_MAX_LABEL_LEN = 50
_LABEL_HASH_LEN = 8


def _normalize_label(label: str) -> str:
    label = label.replace(" ", "-")
    if len(label) <= _MAX_LABEL_LEN:
        return label
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:_LABEL_HASH_LEN]
    return f"{label[: _MAX_LABEL_LEN - _LABEL_HASH_LEN - 1]}-{digest}"


def make_labels(mode_id: str, rubric_version: str) -> list[str]:
    """Standard label set per design §5.2, normalized to be tracker-safe."""
    return [
        _normalize_label(label)
        for label in (
            "agent-triage",
            f"mode:{mode_id}",
            f"rubric:{rubric_version}",
        )
    ]


class Issue(BaseModel):
    """Tracker-side view of an issue, returned by `Tracker.list_open_issues` etc.

    The shape is deliberately narrow — only what the dedup loop and the
    review surface need. Backend-specific fields (Jira's `fields`, GitHub's
    `number`, Linear's `identifier`) are not modeled here. Adapters MAY put
    them in `extra` when useful for tests / debugging, but the core triage
    pipeline never reads them.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    key: str | None = None
    url: str | None = None
    title: str
    body: str
    labels: list[str] = Field(default_factory=list)
    state: IssueState = "open"
    extra: dict[str, Any] = Field(default_factory=dict)


class IssuePatch(BaseModel):
    """Partial update for `Tracker.update_issue`. Fields left None are not patched."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    body: str | None = None
    labels: list[str] | None = None
    state: IssueState | None = None

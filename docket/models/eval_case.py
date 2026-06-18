"""Candidate eval-case model (design §1.1 item 5).

A triage run that finds a failure cluster has, by construction, everything a
regression eval needs: a failure mode, an expected verdict, and a
representative excerpt. `EvalCase` is the portable JSON shape those clusters
export to, for consumption by downstream eval suites (agentevals, DeepEval,
plain pytest fixtures). docket emits candidates; turning them into
executable evals is downstream work (design Appendix B).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docket.models.classification import Severity

EVAL_CASE_SCHEMA = "docket.dev/eval-case/v1"


class EvalCase(BaseModel):
    """One trace-derived candidate regression case, exported per cluster."""

    model_config = ConfigDict(frozen=True)

    schema_: str = Field(default=EVAL_CASE_SCHEMA, alias="schema")
    case_id: str
    mode_id: str
    severity: Severity
    expected: Literal["positive"] = "positive"
    rubric: str
    run_id: str
    representative_trace_id: str
    representative_excerpt: str | None = None
    member_trace_ids: list[str] = Field(default_factory=list)
    cluster_size: int
    created_at: datetime

    def to_json_record(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")

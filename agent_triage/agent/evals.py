"""Eval-case emission: export triage clusters as candidate regression cases.

Closes the triage → regression loop from design §1.1 item 5: each qualifying
cluster becomes one portable JSON file a downstream eval suite can consume.
Excerpts pass through `redact()` before they leave the runtime — eval cases
are designed to be committed to repositories.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from agent_triage.models.cluster import Cluster
from agent_triage.models.eval_case import EvalCase
from agent_triage.observability import redact
from agent_triage.rubric.spec import Rubric

log = logging.getLogger(__name__)


def emit_eval_cases(
    clusters: list[Cluster],
    *,
    rubric: Rubric,
    run_id: str,
    output_dir: Path,
) -> list[Path]:
    """Write one `EvalCase` JSON file per cluster into `output_dir`.

    File names are `{mode_id}--{cluster_id}.json` (both components are
    schema-constrained identifiers, not user data). Returns the written
    paths in cluster order. Re-running the same window overwrites the same
    files — emission is idempotent like the rest of the run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
    created_at = datetime.now(UTC)

    paths: list[Path] = []
    for cluster in clusters:
        case = EvalCase(
            case_id=f"{cluster.mode_id}--{cluster.cluster_id}",
            mode_id=cluster.mode_id,
            severity=cluster.severity,
            rubric=rubric_version,
            run_id=run_id,
            representative_trace_id=cluster.representative_trace_id,
            representative_excerpt=(
                redact(cluster.representative_excerpt) if cluster.representative_excerpt else None
            ),
            member_trace_ids=list(cluster.member_trace_ids),
            cluster_size=cluster.stats.size,
            created_at=created_at,
        )
        path = output_dir / f"{case.case_id}.json"
        path.write_text(json.dumps(case.to_json_record(), indent=2, sort_keys=True))
        paths.append(path)
    if paths:
        log.info("emitted %d candidate eval cases to %s", len(paths), output_dir)
    return paths

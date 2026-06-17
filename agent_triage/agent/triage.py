"""Top-level triage orchestrator wiring the four subagents.

Per design §4.1:
    1. Pull traces from the configured backend within the time window
    2. (Phase 5: filter step is a no-op; classify every trace)
    3. For each candidate:
       a. Run classifier -> set of (mode_id, positive, evidence)
       b. Annotate the trace in the backend with positives
    4. Cluster classified traces per mode_id
    5. For each cluster: draft an issue (local-file queue; no tracker yet)
    6. Emit summary report (report.md)

Phase 5 ships this as a deterministic pipeline (`run_triage_pipeline`). The
Deep Agents wrapper in `build_deep_agent` exposes the same operations as
LangChain tools so the harness from design §4.2 can drive them.

Deterministic `run_id` per design §4.2:

    run_id = sha256(f"{backend_id}|{rubric_id}@{rubric_version}|"
                    f"{window_start_iso}|{window_end_iso}").hexdigest()[:16]

so re-running with the same inputs upserts annotations rather than
duplicating.
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from agent_triage.adapters.base import TraceBackend, Tracker
from agent_triage.agent.report import render_report
from agent_triage.agent.review import ReviewOutcome
from agent_triage.agent.subagents.annotator import Annotator
from agent_triage.agent.subagents.classifier import Classifier, flatten_classifications
from agent_triage.agent.subagents.clusterer import cluster_per_mode
from agent_triage.agent.subagents.drafter import DEFAULT_QUEUE_DIR, draft_issues
from agent_triage.agent.subagents.poster import (
    AutoPostThreshold,
    DedupOutcome,
    dedup_drafts,
)
from agent_triage.errors import BackendError, BudgetExceededError
from agent_triage.llm import build_embedding_provider, build_provider
from agent_triage.llm.base import ModelProvider
from agent_triage.llm.embeddings import EmbeddingProvider
from agent_triage.models.classification import Classification
from agent_triage.models.cluster import Cluster
from agent_triage.models.issue import IssueDraft
from agent_triage.models.report import (
    FetchFailure,
    ModeStats,
    RunReport,
    TraceResult,
    TrackerFailure,
)
from agent_triage.models.trace import OpenInferenceTrace
from agent_triage.observability import redact
from agent_triage.rubric.spec import Rubric

log = logging.getLogger(__name__)


class TriageResult:
    """Bundle returned by `run_triage_pipeline` for the CLI / Deep Agents to format."""

    def __init__(
        self,
        *,
        run_report: RunReport,
        clusters: list[Cluster],
        drafts: list[IssueDraft],
        report_markdown: str,
        dedup_outcomes: list[DedupOutcome] | None = None,
        eval_case_paths: list[Path] | None = None,
    ) -> None:
        self.run_report = run_report
        self.clusters = clusters
        self.drafts = drafts
        self.report_markdown = report_markdown
        self.dedup_outcomes = dedup_outcomes or []
        self.eval_case_paths = eval_case_paths or []
        # Populated by the CLI's --review pass; absent for non-interactive runs.
        self.review_outcomes: list[ReviewOutcome] = []


def compute_run_id(
    *,
    backend_id: str,
    rubric_version: str,
    since: datetime,
    until: datetime,
) -> str:
    """Deterministic run_id per design §4.2."""
    h = hashlib.sha256()
    h.update(backend_id.encode("utf-8"))
    h.update(b"|")
    h.update(rubric_version.encode("utf-8"))
    h.update(b"|")
    h.update(since.isoformat().encode("utf-8"))
    h.update(b"|")
    h.update(until.isoformat().encode("utf-8"))
    return h.hexdigest()[:16]


async def run_triage_pipeline(
    *,
    backend: TraceBackend,
    rubric: Rubric,
    since: datetime,
    until: datetime,
    llm_provider: ModelProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    batch_size: int = 1,
    concurrency: int = 8,
    write_annotations: bool = False,
    run_id: str | None = None,
    backend_id: str = "phoenix",
    output_dir: Path | None = None,
    tracker: Tracker | None = None,
    auto_post_threshold: AutoPostThreshold = "never",
    sample_count: int | None = None,
    sample_strategy: str = "uniform",
    checkpoint: bool = False,
    max_traces: int | None = None,
    max_estimated_cost_usd: float | None = None,
    emit_evals_dir: Path | None = None,
) -> TriageResult:
    """Run the full Phase 5 pipeline: fetch -> classify -> annotate ->
    cluster -> draft -> report.

    Defaults are deliberate:
      - `write_annotations=False` for safe-by-default behavior; pass True
        to enable backend writeback.
      - `run_id` defaults to the deterministic value from `compute_run_id`
        so re-runs with the same inputs upsert.

    Phase 11 additions:
      - `sample_count` + `sample_strategy`: cap per-run work at N traces,
        sampled from the window. Seeded by `run_id` so sampling is
        reproducible across re-runs.
      - `checkpoint`: write sentinel annotations after each trace and
        skip already-processed traces on resume. Requires backend write
        access; implies write_annotations behavior for the sentinel
        regardless of mode outcomes.
      - `max_traces`: hard budget cap (design §4.4 / §8.1 decision 5).
        If the post-checkpoint, post-sampling candidate count still
        exceeds the cap, the run aborts with `BudgetExceededError` —
        silent truncation is forbidden; the operator partitions
        explicitly (narrower window or `--sample`).
      - `max_estimated_cost_usd`: dollar-denominated budget gate (design
        §8.1 decision 5). When set, the run's LLM cost is estimated up
        front (after listing/sampling, before any fetch) and the run
        aborts with `BudgetExceededError` if the estimate exceeds it.
      - `emit_evals_dir`: when set, each cluster is also exported as a
        candidate eval case JSON file (design §1.1 item 5).
    """
    rubric_version = f"{rubric.metadata.name}@{rubric.metadata.version}"
    final_run_id = run_id or compute_run_id(
        backend_id=backend_id,
        rubric_version=rubric_version,
        since=since,
        until=until,
    )
    if llm_provider is None:
        from agent_triage.llm import DEFAULT_PROVIDER_URI

        llm_provider = build_provider(DEFAULT_PROVIDER_URI)
    if embedding_provider is None:
        from agent_triage.llm import DEFAULT_EMBEDDING_URI

        embedding_provider = build_embedding_provider(DEFAULT_EMBEDDING_URI)
    # Eagerly validate both providers' credentials so a missing API key
    # aborts at startup, before any backend I/O (design §4.4: credential
    # failure must not produce a partial run).
    llm_provider.preflight()
    embedding_provider.preflight()

    started_at = datetime.now(UTC)
    log.info(
        "listing traces from backend=%s since=%s until=%s",
        backend_id,
        since.isoformat(),
        until.isoformat(),
    )
    trace_ids = await backend.list_traces(since, until)
    log.info("listed %d trace ids", len(trace_ids))

    if checkpoint and trace_ids:
        already_done = await backend.list_processed_trace_ids(
            run_id=final_run_id, since=since, until=until
        )
        if already_done:
            before = len(trace_ids)
            trace_ids = [tid for tid in trace_ids if tid not in already_done]
            log.info(
                "checkpoint: skipping %d already-processed traces (run_id=%s); %d remain",
                before - len(trace_ids),
                final_run_id,
                len(trace_ids),
            )

    if sample_count is not None and sample_count < len(trace_ids):
        from agent_triage.sampling import Strategy, sample_trace_ids

        strategy: Strategy = sample_strategy  # type: ignore[assignment]
        trace_ids = sample_trace_ids(
            trace_ids,
            n=sample_count,
            strategy=strategy,
            seed=final_run_id,
        )
        log.info(
            "sampled %d traces (strategy=%s, seed=%s)",
            len(trace_ids),
            sample_strategy,
            final_run_id,
        )

    # Budget cap (design §4.4): abort rather than silently truncate. The
    # check runs after checkpoint-skip and sampling, both of which are
    # explicit operator partitioning, so it gates what would actually be
    # fetched and classified.
    if max_traces is not None and len(trace_ids) > max_traces:
        raise BudgetExceededError(
            f"{len(trace_ids)} candidate traces exceed max_traces_per_run={max_traces}. "
            "Narrow the time window, pass --sample N, or raise max_traces_per_run "
            "in agent-triage.yaml. Refusing to truncate silently."
        )

    # Dollar budget gate (design §8.1 decision 5): same estimate as
    # `--dry-run`, computed before any fetch/classify work.
    if max_estimated_cost_usd is not None and trace_ids:
        from agent_triage.cost import estimate_cost

        try:
            estimate = estimate_cost(
                trace_count=len(trace_ids),
                rubric=rubric,
                model=llm_provider.model,
                batch_size=batch_size,
            )
        except ValueError as e:
            raise BudgetExceededError(
                f"max_estimated_cost_usd={max_estimated_cost_usd} is set but the "
                f"run cost could not be estimated: {e}"
            ) from e
        if estimate.estimated_usd > max_estimated_cost_usd:
            raise BudgetExceededError(
                f"estimated LLM cost ${estimate.estimated_usd:.4f} exceeds "
                f"max_estimated_cost_usd={max_estimated_cost_usd}. Narrow the "
                "time window, pass --sample N, or raise max_estimated_cost_usd "
                "in agent-triage.yaml. Refusing to run."
            )

    # 2. Fetch and classify (concurrency from --concurrency).
    classifier = Classifier(llm_provider, batch_size=batch_size, concurrency=concurrency)
    traces: list[tuple[str, OpenInferenceTrace]] = []
    fetch_step = max(1, len(trace_ids) // 10) if trace_ids else 1
    fetch_errors: list[tuple[str, str]] = []
    for idx, tid in enumerate(trace_ids, start=1):
        try:
            traces.append((tid, await backend.get_trace(tid)))
        except Exception as e:  # noqa: BLE001 -- adapter errors vary by backend
            reason = redact(str(e))
            fetch_errors.append((tid, reason))
            log.warning("skipping trace %s after fetch failure: %s", tid, reason)
        if idx == 1 or idx == len(trace_ids) or idx % fetch_step == 0:
            log.info(
                "fetched %d/%d traces (skipped: %d)",
                len(traces),
                len(trace_ids),
                len(fetch_errors),
            )
    if fetch_errors:
        log.warning(
            "fetch phase completed with %d/%d traces skipped due to backend errors",
            len(fetch_errors),
            len(trace_ids),
        )
    if trace_ids and not traces:
        raise BackendError(f"no traces could be fetched from backend (all {len(trace_ids)} failed)")

    log.info(
        "classifying %d traces against %d modes (concurrency=%d)",
        len(traces),
        len(rubric.modes),
        concurrency,
    )
    classify_step = max(1, len(traces) // 10) if traces else 1

    async def _classify_progress(_trace_id: str, done: int, total: int) -> None:
        if done == 1 or done == total or done % classify_step == 0:
            log.info("classified %d/%d traces", done, total)

    classifications_by_trace = await classifier.classify_all(
        traces, rubric, on_progress=_classify_progress
    )
    all_classifications = flatten_classifications(classifications_by_trace)
    positive_count = sum(1 for c in all_classifications if c.error is None and c.positive)
    error_count = sum(1 for c in all_classifications if c.error is not None)
    log.info(
        "classification complete: %d positive, %d negative, %d errors",
        positive_count,
        len(all_classifications) - positive_count - error_count,
        error_count,
    )

    # 3. Annotate.
    annotations_written = 0
    if write_annotations:
        log.info("writing %d positive annotations to backend", positive_count)
        annotator = Annotator(backend, run_id=final_run_id)
        annotations_written = await annotator.annotate_positive(all_classifications, rubric)
        log.info("annotations written: %d", annotations_written)

    # 3a. Checkpoint sentinels. Written only AFTER the annotate stage so an
    # annotation-stage abort never checkpoints traces whose annotations were
    # not written (that would be a forbidden partial-write resume, design
    # §4.4). Traces whose every classification errored (`unprocessed`) are
    # excluded so they get retried on the next resume.
    if checkpoint and classifications_by_trace:
        checkpointable = [
            tid
            for tid, items in classifications_by_trace.items()
            if not items or any(c.error is None for c in items)
        ]
        unprocessed_count = len(classifications_by_trace) - len(checkpointable)
        if unprocessed_count:
            log.warning(
                "%d trace(s) had every mode error and were left un-checkpointed "
                "for retry on resume",
                unprocessed_count,
            )
        log.info(
            "writing %d sentinel annotations for resumability",
            len(checkpointable),
        )
        sem = asyncio.Semaphore(concurrency)
        sentinel_failures = 0

        async def _mark_one(trace_id: str) -> None:
            nonlocal sentinel_failures
            async with sem:
                try:
                    await backend.mark_trace_processed(
                        trace_id, run_id=final_run_id, rubric_version=rubric_version
                    )
                except Exception as e:  # noqa: BLE001 -- adapter errors vary by backend
                    sentinel_failures += 1
                    log.warning(
                        "sentinel write failed for trace %s (will reclassify on resume): %s",
                        trace_id,
                        e,
                    )

        await asyncio.gather(*(_mark_one(tid) for tid in checkpointable))
        if sentinel_failures:
            log.warning(
                "%d/%d sentinel writes failed; those traces will be reclassified on resume",
                sentinel_failures,
                len(checkpointable),
            )

    # 4. Cluster.
    log.info("clustering positive classifications")
    clusters = await cluster_per_mode(
        all_classifications,
        rubric=rubric,
        embedding_provider=embedding_provider,
        traces_by_id=dict(traces),
    )
    log.info("produced %d clusters", len(clusters))

    # 5. Draft.
    if clusters:
        log.info("drafting issues for %d clusters", len(clusters))
    drafts = await draft_issues(
        clusters,
        rubric=rubric,
        llm_provider=llm_provider,
        run_id=final_run_id,
        output_dir=output_dir if output_dir is not None else DEFAULT_QUEUE_DIR,
    )
    log.info("drafted %d issues", len(drafts))

    # 5a. Optional eval-case emission (design §1.1 item 5): one portable
    # JSON candidate regression case per cluster.
    eval_case_paths: list[Path] = []
    if emit_evals_dir is not None and clusters:
        from agent_triage.agent.evals import emit_eval_cases

        eval_case_paths = emit_eval_cases(
            clusters,
            rubric=rubric,
            run_id=final_run_id,
            output_dir=emit_evals_dir,
        )

    # 5b. Dedup against tracker (Phase 8): for drafts matching an existing
    # open issue, comment if new members appeared; for new issues, auto-post
    # when severity meets `auto_post_threshold` and skip otherwise (the CLI's
    # `--review` mode handles `needs_create` outcomes interactively).
    dedup_outcomes: list[DedupOutcome] = []
    if tracker is not None and drafts:
        log.info("deduplicating %d drafts against tracker", len(drafts))
        dedup_outcomes = await dedup_drafts(
            drafts,
            tracker=tracker,
            auto_post_threshold=auto_post_threshold,
        )
        log.info("tracker dedup complete (%d outcomes)", len(dedup_outcomes))

    # Tracker write failures are contained per draft (design §4.4): surface
    # them in the run report instead of aborting. `failure_reason` is already
    # redacted by the poster.
    tracker_failures = [
        TrackerFailure(
            cluster_id=o.draft.cluster_id,
            mode_id=o.draft.mode_id,
            reason=o.failure_reason or "tracker error",
        )
        for o in dedup_outcomes
        if o.action == "failed"
    ]
    if tracker_failures:
        log.warning(
            "%d/%d drafts failed tracker writeback; they remain in the local queue",
            len(tracker_failures),
            len(dedup_outcomes),
        )

    # 6. Report. `trace_results` covers only traces that were actually
    # fetched; never-fetched traces are surfaced separately as
    # `fetch_failures` (design §4.4: log, skip, report at end).
    finished_at = datetime.now(UTC)
    fetched_ids = {tid for tid, _ in traces}
    run_report = RunReport(
        run_id=final_run_id,
        rubric_name=rubric.metadata.name,
        rubric_version=rubric.metadata.version,
        since=since,
        until=until,
        started_at=started_at,
        finished_at=finished_at,
        trace_count=len(trace_ids),
        trace_results=[
            TraceResult(trace_id=tid, classifications=classifications_by_trace.get(tid, []))
            for tid in trace_ids
            if tid in fetched_ids
        ],
        mode_stats=_compute_mode_stats(rubric, classifications_by_trace),
        annotations_written=annotations_written,
        tracker_failures=tracker_failures,
        fetch_failures=[FetchFailure(trace_id=tid, reason=reason) for tid, reason in fetch_errors],
    )
    report_md = render_report(
        run_report=run_report,
        clusters=clusters,
        drafts=drafts,
        dedup_outcomes=dedup_outcomes or None,
        queue_dir=output_dir,
    )

    return TriageResult(
        run_report=run_report,
        clusters=clusters,
        drafts=drafts,
        report_markdown=report_md,
        dedup_outcomes=dedup_outcomes,
        eval_case_paths=eval_case_paths,
    )


def _compute_mode_stats(
    rubric: Rubric,
    classifications_by_trace: dict[str, list[Classification]],
) -> list[ModeStats]:
    by_mode: dict[str, ModeStats] = {
        mode.id: ModeStats(mode_id=mode.id, severity=mode.severity) for mode in rubric.modes
    }
    for items in classifications_by_trace.values():
        for c in items:
            stats = by_mode.get(c.mode_id)
            if stats is None:
                continue
            if c.error is not None:
                stats.error_count += 1
            elif c.positive:
                stats.positive_count += 1
            else:
                stats.negative_count += 1
    return [by_mode[m.id] for m in rubric.modes]

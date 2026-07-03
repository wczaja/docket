"""docket CLI."""

import asyncio
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click

from docket import __version__
from docket.agent.triage import TriageResult, run_triage_pipeline
from docket.config import Config
from docket.errors import BudgetExceededError, ConfigError, DocketError
from docket.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_embedding_provider,
    build_provider,
)
from docket.rubric.loader import load_rubric
from docket.rubric.registry import is_builtin_uri
from docket.rubric.spec import Rubric
from docket.rubric.validator import validate_rubric_yaml
from docket.runtime import build_backend, build_tracker, resolve_backend_id
from docket.self_test import run_self_test

DEFAULT_MAX_TRACES_PER_RUN = 1000


def _configure_logging(*, quiet: bool, verbose: int) -> None:
    """Wire stderr logging for the `docket` namespace only.

    Scoping to the namespace (rather than the root logger) avoids stomping on
    consumers of the library who configure their own logging.
    """
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger = logging.getLogger("docket")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="docket")
def main() -> None:
    """docket: triage runtime for LLM agent traces."""


@main.command()
@click.argument("source", type=str)
def validate(source: str) -> None:
    """Validate a rubric file or URI against the v1 schema.

    SOURCE may be a filesystem path, a `file://` URI, or an
    `docket.dev/builtin/<name>/<version>` URI.
    """
    resolved = _normalize_cli_source(source)
    try:
        validate_rubric_yaml(resolved)
        rubric = load_rubric(resolved)
    except DocketError as e:
        click.echo(f"INVALID: {e}", err=True)
        sys.exit(1)
    click.echo(f"OK: {source} ({rubric.metadata.name} v{rubric.metadata.version})")


DEFAULT_DEMO_RUBRIC = "docket.dev/builtin/agents/v1"
_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@main.command()
@click.option(
    "--live",
    is_flag=True,
    help="Judge with a real LLM provider instead of the scripted demo judge. "
    "Needs the provider's API key; everything else stays local and free.",
)
@click.option(
    "--provider",
    "provider_uri",
    default=None,
    help="provider:model URI for --live (default: the standard provider). "
    "Only meaningful together with --live.",
)
@click.option(
    "--embedding",
    "embedding_uri",
    default=None,
    help="provider:model URI for clustering embeddings (e.g. "
    "'openai:text-embedding-3-small'). Default: free deterministic demo "
    "embeddings — no key needed, even with --live.",
)
@click.option(
    "--rubric",
    "rubric_source",
    type=str,
    default=None,
    help=f"Rubric to classify against (default: {DEFAULT_DEMO_RUBRIC}). "
    "Point at your own YAML to see taxonomy-as-code in action; custom "
    "llm_judge modes need --live to be judged by a real model.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docket-demo"),
    show_default=True,
    help="Directory for the queued issue drafts and report.md.",
)
@click.option(
    "--to-phoenix",
    "phoenix_url",
    default=None,
    metavar="URL",
    help="Don't run the pipeline; instead ingest the demo traces into a "
    "running Phoenix at URL (e.g. http://localhost:6006) and print the "
    "`docket run` command that triages them there.",
)
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (repeatable).")
@click.option("--quiet", is_flag=True, help="Suppress progress logs; print only the results.")
def demo(  # noqa: PLR0913 -- CLI options form one logical unit
    live: bool,
    provider_uri: str | None,
    embedding_uri: str | None,
    rubric_source: str | None,
    out_dir: Path,
    phoenix_url: str | None,
    verbose: int,
    quiet: bool,
) -> None:
    """Run the real triage pipeline on bundled synthetic traces. No setup.

    Classifies 60 synthetic agent traces (20 clean + 40 seeded failures)
    against the builtin `agents/v1` failure-mode rubric, clusters the
    positives, and drafts issues into local files — the exact pipeline
    `docket run` executes against a real backend, minus the credentials:
    no API keys, no Docker, no instrumented app.

    By default the LLM-judge modes run under a clearly-labeled scripted
    judge (deterministic, free). Pass --live to judge with a real model
    using one API key. The deterministic detectors (regex, tool_call,
    metric_threshold) run for real either way.
    """
    _configure_logging(quiet=quiet, verbose=verbose)
    if phoenix_url is not None:
        _demo_seed_phoenix(phoenix_url)
        return

    from docket.demo import (
        DEMO_BACKEND_ID,
        DemoBackend,
        DemoEmbeddingProvider,
        DemoJudgeProvider,
        demo_summary,
    )
    from docket.llm import DEFAULT_PROVIDER_URI
    from docket.llm.base import ModelProvider

    try:
        source = rubric_source if rubric_source is not None else DEFAULT_DEMO_RUBRIC
        rubric = load_rubric(_normalize_cli_source(source))
        llm_provider: ModelProvider
        if live:
            llm_provider = build_provider(provider_uri or DEFAULT_PROVIDER_URI)
        elif provider_uri is not None:
            raise ConfigError(
                "--provider only applies with --live; the default demo judge is scripted."
            )
        else:
            llm_provider = DemoJudgeProvider()
        embedding_provider = (
            build_embedding_provider(embedding_uri)
            if embedding_uri is not None
            else DemoEmbeddingProvider()
        )
    except DocketError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    summary = demo_summary()
    judge_label = (
        f"live model `{llm_provider.model}`"
        if live
        else "scripted demo judge (deterministic, free — pass --live for a real model)"
    )
    click.echo(
        f"docket demo: {summary['total']} synthetic traces "
        f"({summary['clean']} clean, {summary['seeded_failures']} seeded failures), "
        f"rubric `{rubric.metadata.name}`, judge: {judge_label}\n"
    )

    backend = DemoBackend()
    until = datetime.now(UTC)
    since = until - timedelta(hours=1)

    async def _demo_run() -> TriageResult:
        return await run_triage_pipeline(
            backend=backend,
            rubric=rubric,
            since=since,
            until=until,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            backend_id=DEMO_BACKEND_ID,
            output_dir=out_dir,
            max_traces=DEFAULT_MAX_TRACES_PER_RUN,
        )

    try:
        result = asyncio.run(_demo_run())
    except DocketError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    click.echo(result.report_markdown)
    if result.drafts:
        top = min(result.drafts, key=lambda d: _SEVERITY_RANK.get(d.severity, 9))
        click.echo("\n---\n\n# Sample drafted issue\n")
        click.echo(top.to_markdown())
    click.echo(_demo_epilogue(result, out_dir=out_dir, live=live, llm_provider=llm_provider))


def _demo_epilogue(
    result: TriageResult,
    *,
    out_dir: Path,
    live: bool,
    llm_provider: Any,
) -> str:
    from docket.demo import DemoJudgeProvider

    lines = ["", "---", "", f"Drafts and report.md written to `{out_dir}/`."]
    halluc = next(
        (ms for ms in result.run_report.mode_stats if ms.mode_id == "hallucination"),
        None,
    )
    if (
        halluc is not None
        and halluc.positive_count
        and not any(c.mode_id == "hallucination" for c in result.clusters)
    ):
        lines.append(
            f"Note: {halluc.positive_count} `hallucination` positives formed no "
            "cluster — each falsehood is distinct, and groups smaller than the "
            "rubric's `min_cluster_size: 3` are dropped, not drafted. That is "
            "the issue-spam guard doing its job."
        )
    if isinstance(llm_provider, DemoJudgeProvider) and llm_provider.unknown_judge_calls:
        lines.append(
            f"WARNING: {llm_provider.unknown_judge_calls} llm_judge evaluations hit "
            "modes the scripted judge doesn't know; they scored negative. Pass "
            "--live to judge custom modes with a real model."
        )
    lines += [
        "",
        "Where to go from here:",
        "  1. The taxonomy is code. Copy a rubric (rubrics/ in the repo, or "
        "start from the builtin), add a mode, and re-run:",
        "       docket demo --rubric ./my-rubric.yaml --live",
    ]
    if not live:
        lines.append(
            "  2. Same traces, real judge (one API key):\n"
            "       ANTHROPIC_API_KEY=... docket demo --live"
        )
    lines.append(
        "  3. Point it at a real backend: start Phoenix (`docker compose up "
        "phoenix` from the repo, or `docker run -p 6006:6006 -p 4317:4317 "
        "arizephoenix/phoenix:latest`), seed it with `docket demo "
        "--to-phoenix http://localhost:6006`, and run the printed command. "
        "`docket init` scaffolds the config — see docs/quickstart.md."
    )
    return "\n".join(lines)


def _demo_seed_phoenix(phoenix_url: str) -> None:
    from docket.demo import ingest_to_phoenix

    click.echo(f"Ingesting demo traces into Phoenix at {phoenix_url} ...")
    ingested, failures = asyncio.run(ingest_to_phoenix(phoenix_url))
    for failure in failures:
        click.echo(f"  FAIL {failure}", err=True)
    if ingested == 0:
        click.echo(
            f"ERROR: nothing ingested — is Phoenix running at {phoenix_url}? "
            "Start one with: docker compose up phoenix (or docker run -p 6006:6006 "
            "-p 4317:4317 arizephoenix/phoenix:latest)",
            err=True,
        )
        sys.exit(1)
    click.echo(f"Ingested {ingested} traces. Triage them with:\n")
    click.echo(
        f"  docket run --backend phoenix --phoenix-url {phoenix_url} \\\n"
        f"      --rubric {DEFAULT_DEMO_RUBRIC} --since 1h"
    )
    if failures:
        sys.exit(1)


_INIT_BACKENDS: dict[str, tuple[str, str]] = {
    # backend -> (adapter command, env block template)
    "phoenix": (
        "docket-adapter-phoenix",
        "    PHOENIX_URL: {phoenix_url}",
    ),
    "langfuse": (
        "docket-adapter-langfuse",
        "    LANGFUSE_HOST: {langfuse_host}\n"
        "    LANGFUSE_PUBLIC_KEY: ${{LANGFUSE_PUBLIC_KEY}}\n"
        "    LANGFUSE_SECRET_KEY: ${{LANGFUSE_SECRET_KEY}}",
    ),
    "langsmith": (
        "docket-adapter-langsmith",
        "    LANGSMITH_API_KEY: ${{LANGSMITH_API_KEY}}\n    LANGSMITH_PROJECT: {langsmith_project}",
    ),
}


@main.command()
@click.option(
    "--path",
    "config_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("docket.yaml"),
    show_default=True,
    help="Where to write the config file.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init(config_out: Path, force: bool) -> None:
    """Interactively scaffold a docket.yaml (backend, tracker, rubric).

    Every prompt has a sensible default, so pressing Enter through the
    whole flow yields a working local-Phoenix, no-tracker, builtin-rubric
    config. Secrets are referenced as ${ENV_VAR}, never written to disk.
    """
    if config_out.exists() and not force:
        click.echo(
            f"ERROR: {config_out} already exists; pass --force to overwrite.",
            err=True,
        )
        sys.exit(1)

    backend = click.prompt(
        "Trace backend",
        type=click.Choice(["phoenix", "langfuse", "langsmith"]),
        default="phoenix",
    )
    env_values: dict[str, str] = {}
    if backend == "phoenix":
        env_values["phoenix_url"] = click.prompt("Phoenix URL", default="http://localhost:6006")
    elif backend == "langfuse":
        env_values["langfuse_host"] = click.prompt(
            "Langfuse host", default="https://cloud.langfuse.com"
        )
    else:
        env_values["langsmith_project"] = click.prompt("LangSmith project", default="default")
    backend_command, backend_env_template = _INIT_BACKENDS[backend]
    backend_env = backend_env_template.format(**env_values)

    tracker = click.prompt(
        "Issue tracker (none = queue drafts locally)",
        type=click.Choice(["none", "github", "jira", "linear"]),
        default="none",
    )
    tracker_block = ""
    if tracker == "github":
        owner = click.prompt("GitHub owner (user or org)")
        repo = click.prompt("GitHub repo for issues")
        tracker_block = (
            "tracker:\n"
            "  type: mcp\n"
            "  command: docket-adapter-github\n"
            "  env:\n"
            "    GITHUB_TOKEN: ${GITHUB_TOKEN}\n"
            f"    GITHUB_OWNER: {owner}\n"
            f"    GITHUB_REPO: {repo}\n"
        )
    elif tracker == "jira":
        host = click.prompt("Jira host (e.g. https://example.atlassian.net)")
        project = click.prompt("Jira project key (e.g. AGT)")
        tracker_block = (
            "tracker:\n"
            "  type: mcp\n"
            "  command: docket-adapter-jira\n"
            "  env:\n"
            f"    JIRA_HOST: {host}\n"
            f"    JIRA_PROJECT: {project}\n"
            "    JIRA_EMAIL: ${JIRA_EMAIL}\n"
            "    JIRA_API_TOKEN: ${JIRA_API_TOKEN}\n"
        )
    elif tracker == "linear":
        team_id = click.prompt("Linear team ID (the UUID, not the team key)")
        tracker_block = (
            "tracker:\n"
            "  type: mcp\n"
            "  command: docket-adapter-linear\n"
            "  env:\n"
            "    LINEAR_API_KEY: ${LINEAR_API_KEY}\n"
            f"    LINEAR_TEAM_ID: {team_id}\n"
        )

    rubric = click.prompt(
        "Rubric (builtin URI or path to your YAML)",
        default=DEFAULT_DEMO_RUBRIC,
    )
    try:
        load_rubric(_normalize_cli_source(rubric))
    except DocketError as e:
        click.echo(f"ERROR: rubric {rubric!r} does not load: {e}", err=True)
        sys.exit(1)

    auto_post = click.prompt(
        "Auto-post threshold (never = human reviews every draft)",
        type=click.Choice(["never", "critical", "high", "medium", "low"]),
        default="never",
    )

    tracker_section = (
        tracker_block
        if tracker_block
        else "# No tracker configured: drafts queue locally for `docket queue` /\n"
        "# `docket run --review`. Add one later — see docs/configuration.md.\n"
    )
    content = (
        "# docket.yaml — generated by `docket init`.\n"
        "# Reference for every field: docs/configuration.md\n"
        "# Secrets stay in the environment; ${VAR} references are resolved at load.\n"
        "\n"
        "trace_backend:\n"
        "  type: mcp\n"
        f"  command: {backend_command}\n"
        "  env:\n"
        f"{backend_env}\n"
        "\n"
        f"{tracker_section}"
        "\n"
        f"rubric: {rubric}\n"
        "\n"
        "# Budget guardrails: the run aborts (loudly) past either ceiling.\n"
        "max_traces_per_run: 1000\n"
        "# max_estimated_cost_usd: 5.0\n"
        "\n"
        f"auto_post_threshold: {auto_post}\n"
    )
    config_out.write_text(content)
    click.echo(f"\nWrote {config_out}.")

    required_env = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", content)))
    if required_env:
        click.echo("Export these before running: " + ", ".join(required_env))
    click.echo(
        "The classifier needs ANTHROPIC_API_KEY (or OPENAI_API_KEY with "
        "--provider openai); clustering defaults to OpenAI embeddings — "
        "single-key setups pass --clustering mode-only or --embedding local:... "
        '(pip install "docket-runtime[local-embeddings]").'
    )
    click.echo("\nNext:\n")
    click.echo(f"  docket run --config {config_out} --since 1h --dry-run   # price the window")
    click.echo(f"  docket run --config {config_out} --since 1h             # real run")


@main.command()
@click.option(
    "--rubric",
    "rubric_source",
    type=str,
    default=None,
    help="Rubric source (path, file:// URI, or docket.dev/builtin/...). "
    "Overrides the rubric field in the config.",
)
@click.option(
    "--since",
    default="1h",
    help="Start of window: a duration back from now (e.g. '1h', '24h', '7d') "
    "or an absolute ISO-8601 timestamp (e.g. '2026-06-01T00:00:00Z', "
    "'2026-06-01'; naive timestamps are assumed UTC). Absolute timestamps "
    "keep the default run_id stable across invocations, enabling resume.",
)
@click.option(
    "--until",
    default=None,
    help="End of window (defaults to now). Same formats as --since: a "
    "duration back from now or an absolute ISO-8601 timestamp.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default="docket.yaml",
    help="Path to docket.yaml. May be absent if --backend + --phoenix-url are given.",
)
@click.option(
    "--backend",
    type=click.Choice(["phoenix", "langfuse", "langsmith", "demo"]),
    default=None,
    help="Override the trace backend. Supports: phoenix, langfuse, langsmith, "
    "and demo (bundled synthetic traces — see `docket demo` for the "
    "zero-credential wrapper).",
)
@click.option(
    "--phoenix-url",
    default=None,
    help="Phoenix base URL (e.g. http://localhost:6006). Overrides PHOENIX_URL in config.",
)
@click.option(
    "--phoenix-api-key",
    default=None,
    help="Phoenix API key. Overrides PHOENIX_API_KEY in config.",
)
@click.option(
    "--langfuse-host",
    default=None,
    help="Langfuse host URL (e.g. http://localhost:3000). Overrides LANGFUSE_HOST in config.",
)
@click.option(
    "--langfuse-public-key",
    default=None,
    help="Langfuse public key. Overrides LANGFUSE_PUBLIC_KEY in config.",
)
@click.option(
    "--langfuse-secret-key",
    default=None,
    help="Langfuse secret key. Overrides LANGFUSE_SECRET_KEY in config.",
)
@click.option(
    "--langsmith-api-key",
    default=None,
    help="LangSmith API key. Overrides LANGSMITH_API_KEY in config.",
)
@click.option(
    "--langsmith-endpoint",
    default=None,
    help="LangSmith API endpoint (defaults to https://api.smith.langchain.com). "
    "Overrides LANGSMITH_ENDPOINT in config.",
)
@click.option(
    "--langsmith-project",
    default=None,
    help="LangSmith project (session) name to filter runs against. "
    "Overrides LANGSMITH_PROJECT in config.",
)
@click.option(
    "--tracker",
    type=click.Choice(["jira", "linear", "github"]),
    default=None,
    help="Issue tracker for dedup + posting. Supports: jira, linear, github.",
)
@click.option(
    "--jira-host",
    default=None,
    help="Jira host URL (e.g. https://example.atlassian.net). Overrides JIRA_HOST in config.",
)
@click.option(
    "--jira-project",
    default=None,
    help="Jira project key (e.g. AGT). Overrides JIRA_PROJECT in config.",
)
@click.option(
    "--jira-email",
    default=None,
    help="Atlassian account email for Cloud Basic auth. Overrides JIRA_EMAIL in config.",
)
@click.option(
    "--jira-api-token",
    default=None,
    help="Atlassian Cloud API token. Overrides JIRA_API_TOKEN in config.",
)
@click.option(
    "--jira-pat",
    default=None,
    help="Jira Data Center Personal Access Token. Overrides JIRA_PAT in config.",
)
@click.option(
    "--jira-deployment",
    type=click.Choice(["cloud", "datacenter"]),
    default=None,
    help="Jira deployment type. Default: auto-detect from hostname.",
)
@click.option(
    "--linear-api-key",
    default=None,
    help="Linear personal API key. Overrides LINEAR_API_KEY in config.",
)
@click.option(
    "--linear-team",
    "linear_team_id",
    default=None,
    help="Linear team ID (the UUID, not the team name/key). Overrides LINEAR_TEAM_ID in config.",
)
@click.option(
    "--linear-endpoint",
    default=None,
    help="Linear GraphQL endpoint. Default: https://api.linear.app/graphql.",
)
@click.option(
    "--github-token",
    default=None,
    help="GitHub personal access token (classic or fine-grained). Overrides "
    "GITHUB_TOKEN in config.",
)
@click.option(
    "--github-owner",
    default=None,
    help="GitHub repository owner (user or organization). Overrides GITHUB_OWNER in config.",
)
@click.option(
    "--github-repo",
    default=None,
    help="GitHub repository name. Overrides GITHUB_REPO in config.",
)
@click.option(
    "--github-api-url",
    default=None,
    help="GitHub API base URL (set for GitHub Enterprise Server). Default: https://api.github.com.",
)
@click.option(
    "--annotate/--no-annotate",
    default=False,
    help="Write annotations back to the backend. Default: read-only.",
)
@click.option(
    "--batch",
    "batch_size",
    default=1,
    type=click.IntRange(1, 32),
    help="Traces per llm_judge LLM call (budget mode). >1 batches multiple "
    "traces into one structured-output call, cutting cost roughly "
    "proportionally at some accuracy risk for long traces. Default: 1.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM provider for llm_judge modes that don't set their own `model:`",
)
@click.option(
    "--model",
    default=None,
    help="Override the provider's default model.",
)
@click.option(
    "--concurrency",
    default=8,
    type=click.IntRange(1, 64),
    help="Max traces classified in parallel. Default: 8. Lower this (e.g. 1-2) "
    "if your LLM provider tier has a tight requests-per-minute limit; the "
    "classifier issues one structured-output call per (trace, mode) pair.",
)
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help="Override the deterministic run_id (defaults to sha256(backend|rubric|since|until)[:16]).",
)
@click.option(
    "--queue-dir",
    "queue_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory for drafted issues. Default: ~/.docket/queued-issues/.",
)
@click.option(
    "--instrument-to",
    "instrument_to",
    default=None,
    help="OpenInference instrumentation endpoint. When set, the triage "
    "agent emits OTLP spans for its own run to this URL (typically "
    "http://localhost:6006 for a local Phoenix).",
)
@click.option(
    "--auto-post-threshold",
    type=click.Choice(["critical", "high", "medium", "low", "never"]),
    default=None,
    help="Severity at or above which to auto-post new issues to the tracker. "
    "`never` (default unless set in config) keeps every needs-create draft "
    "in the local queue for `--review` or manual inspection.",
)
@click.option(
    "--review/--no-review",
    default=False,
    help="After the pipeline runs, walk each `needs_create` draft through "
    "`$EDITOR` + accept/reject + post. Overrides --auto-post-threshold for "
    "drafts the operator approves.",
)
@click.option(
    "--agent/--no-agent",
    default=False,
    help="Drive the workflow through the Deep Agents harness. Default is the "
    "deterministic pipeline (recommended for batch / CI runs). Agent mode "
    "ignores --tracker, --review, --sample, --checkpoint, and "
    "--auto-post-threshold.",
)
@click.option(
    "--sample",
    "sample_count",
    type=click.IntRange(1, 1_000_000),
    default=None,
    help="Cap the run at N traces, sampled from the window. Sampling is seeded "
    "by the run_id so re-runs with the same inputs sample identically. "
    "Recommended for production windows holding 10k+ traces.",
)
@click.option(
    "--strategy",
    "sample_strategy",
    type=click.Choice(["uniform", "stratified", "errors-only"]),
    default="uniform",
    help="Sampling strategy (v1.1 implements uniform; stratified and "
    "errors-only fall back to uniform with a warning pending Phase 12).",
)
@click.option(
    "--checkpoint/--no-checkpoint",
    default=False,
    help="Write per-trace sentinel annotations after classification, and on "
    "resume skip traces already marked for this run_id. Requires backend "
    "write access; safe to use alongside --annotate. Recommended for "
    "sub-hourly cron runs where resumability across transient failures "
    "matters.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Estimate LLM cost for the run and exit without executing it. "
    "Uses the same windowing, sampling, and rubric resolution as a real "
    "run so the estimate reflects what you'd actually spend.",
)
@click.option(
    "--max-traces",
    "max_traces",
    type=click.IntRange(1, 10_000_000),
    default=None,
    help="Hard budget cap on candidate traces per run. The run aborts (no "
    "silent truncation) if the windowed count after sampling still exceeds "
    "the cap. Default: max_traces_per_run from config (1000 when unset).",
)
@click.option(
    "--emit-evals",
    "emit_evals_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Also export each cluster as a candidate eval case (portable JSON) "
    "into this directory, for downstream regression suites.",
)
@click.option(
    "--embedding",
    "embedding_uri",
    default=None,
    help="Embedding provider for clustering, as 'provider:model' "
    "(e.g. 'openai:text-embedding-3-small', 'voyage:voyage-3.5-lite', "
    "'local:BAAI/bge-small-en-v1.5'). Default: OpenAI. voyage needs "
    "VOYAGE_API_KEY; local needs no key (pip install "
    '"docket-runtime[local-embeddings]").',
)
@click.option(
    "--clustering",
    type=click.Choice(["embedding", "mode-only"]),
    default="embedding",
    show_default=True,
    help="Clustering strategy. 'embedding': per-mode HDBSCAN over excerpt "
    "embeddings (needs an embedding provider). 'mode-only': one cluster per "
    "firing mode, no embeddings — the single-API-key operating mode; lossy "
    "(sub-patterns within a mode are not separated).",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Emit DEBUG-level logs (use -vv for even more detail).",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress informational progress output; only warnings and errors.",
)
def run(  # noqa: PLR0913 -- CLI options form one logical unit
    rubric_source: str | None,
    since: str,
    until: str | None,
    config_path: Path,
    backend: str | None,
    phoenix_url: str | None,
    phoenix_api_key: str | None,
    langfuse_host: str | None,
    langfuse_public_key: str | None,
    langfuse_secret_key: str | None,
    langsmith_api_key: str | None,
    langsmith_endpoint: str | None,
    langsmith_project: str | None,
    tracker: str | None,
    jira_host: str | None,
    jira_project: str | None,
    jira_email: str | None,
    jira_api_token: str | None,
    jira_pat: str | None,
    jira_deployment: str | None,
    linear_api_key: str | None,
    linear_team_id: str | None,
    linear_endpoint: str | None,
    github_token: str | None,
    github_owner: str | None,
    github_repo: str | None,
    github_api_url: str | None,
    annotate: bool,
    batch_size: int,
    provider: str,
    model: str | None,
    concurrency: int,
    run_id: str | None,
    queue_dir: Path | None,
    instrument_to: str | None,
    auto_post_threshold: str | None,
    review: bool,
    agent: bool,
    sample_count: int | None,
    sample_strategy: str,
    checkpoint: bool,
    dry_run: bool,
    max_traces: int | None,
    emit_evals_dir: Path | None,
    embedding_uri: str | None,
    clustering: str,
    verbose: int,
    quiet: bool,
) -> None:
    """Run the full triage pipeline against the configured backend.

    Phases:
      1. Fetch traces in [since, until]
      2. Classify each trace against every mode in the rubric
      3. (Optional) annotate positive classifications back to the backend
      4. Cluster classified-positive traces per mode
      5. Draft one issue per cluster (queued to local file by default)
      6. Emit a markdown report

    Default is read-only -- pass --annotate to enable backend writeback.
    """
    _configure_logging(quiet=quiet, verbose=verbose)
    try:
        config = _maybe_load_config(config_path)
        rubric = _resolve_rubric(rubric_source, config)
        adapter = build_backend(
            backend_name=backend,
            config=config,
            phoenix_url=phoenix_url,
            phoenix_api_key=phoenix_api_key,
            langfuse_host=langfuse_host,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
        )
        tracker_adapter = build_tracker(
            tracker_name=tracker,
            config=config,
            jira_host=jira_host,
            jira_project=jira_project,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            jira_pat=jira_pat,
            jira_deployment=jira_deployment,
            linear_api_key=linear_api_key,
            linear_team_id=linear_team_id,
            linear_endpoint=linear_endpoint,
            github_token=github_token,
            github_owner=github_owner,
            github_repo=github_repo,
            github_api_url=github_api_url,
        )
        resolved_threshold = _resolve_auto_post_threshold(auto_post_threshold, config)
        since_dt, until_dt = _resolve_window(since, until)
        default_model = (
            model
            or {
                "anthropic": DEFAULT_ANTHROPIC_MODEL,
                "openai": DEFAULT_OPENAI_MODEL,
            }[provider]
        )
        llm_provider = build_provider(f"{provider}:{default_model}")
        embedding_provider = (
            build_embedding_provider(embedding_uri) if embedding_uri is not None else None
        )
        # Budget cap (design §8.1 decision 5): CLI flag > config > 1000.
        resolved_max_traces = max_traces
        if resolved_max_traces is None:
            resolved_max_traces = config.max_traces_per_run if config is not None else 1000
    except DocketError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    if agent:
        ignored = [
            ("--tracker", tracker is not None),
            ("--review", review),
            ("--sample", sample_count is not None),
            ("--checkpoint", checkpoint),
            ("--auto-post-threshold", auto_post_threshold is not None),
            ("--clustering", clustering != "embedding"),
        ]
        for flag, was_set in ignored:
            if was_set:
                click.echo(
                    f"WARNING: {flag} is ignored in agent mode (--agent); "
                    "use the deterministic pipeline (drop --agent) for it to take effect.",
                    err=True,
                )

    max_estimated_cost_usd = config.max_estimated_cost_usd if config is not None else None

    if dry_run:
        from docket.cost import estimate_cost

        async def _dry_run_count() -> int:
            try:
                ids = await adapter.list_traces(since_dt, until_dt)
            finally:
                await adapter.close()
            if sample_count is not None and sample_count < len(ids):
                return sample_count
            return len(ids)

        try:
            n = asyncio.run(_dry_run_count())
        except DocketError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        try:
            estimate = estimate_cost(
                trace_count=n,
                rubric=rubric,
                model=default_model,
                batch_size=batch_size,
            )
        except ValueError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        click.echo(estimate.render())
        if max_estimated_cost_usd is not None and estimate.estimated_usd > max_estimated_cost_usd:
            err = BudgetExceededError(
                f"estimated LLM cost ${estimate.estimated_usd:.4f} exceeds "
                f"max_estimated_cost_usd={max_estimated_cost_usd}. Narrow the time "
                "window, pass --sample N, or raise max_estimated_cost_usd in "
                "docket.yaml. Refusing to run."
            )
            click.echo(f"ERROR: {err}", err=True)
            sys.exit(1)
        sys.exit(0)

    async def _run() -> TriageResult | str:
        try:
            if agent:
                return await _run_via_deep_agent(
                    backend_adapter=adapter,
                    rubric=rubric,
                    llm_provider=llm_provider,
                    since_dt=since_dt,
                    until_dt=until_dt,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    write_annotations=annotate,
                    run_id=run_id,
                    queue_dir=queue_dir,
                    embedding_provider=embedding_provider,
                    backend_id=resolve_backend_id(backend, config),
                )
            pipeline_result = await run_triage_pipeline(
                backend=adapter,
                rubric=rubric,
                since=since_dt,
                until=until_dt,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                batch_size=batch_size,
                concurrency=concurrency,
                write_annotations=annotate,
                run_id=run_id,
                backend_id=resolve_backend_id(backend, config),
                output_dir=queue_dir,
                tracker=tracker_adapter,
                auto_post_threshold=resolved_threshold,
                sample_count=sample_count,
                sample_strategy=sample_strategy,
                checkpoint=checkpoint,
                max_traces=resolved_max_traces,
                max_estimated_cost_usd=max_estimated_cost_usd,
                emit_evals_dir=emit_evals_dir,
                clustering=clustering,
            )
            if review and tracker_adapter is None:
                click.echo(
                    "WARNING: --review does nothing without a tracker; pass "
                    "--tracker (or set `tracker:` in the config) to post "
                    "reviewed drafts.",
                    err=True,
                )
            if review and tracker_adapter is not None:
                from docket.agent.review import review_and_post

                review_outcomes = await review_and_post(
                    pipeline_result.dedup_outcomes,
                    tracker=tracker_adapter,
                )
                pipeline_result.review_outcomes = review_outcomes
            return pipeline_result
        finally:
            await adapter.close()
            if tracker_adapter is not None:
                await tracker_adapter.close()

    try:
        if instrument_to:
            from docket.observability import configure_instrumentation

            with configure_instrumentation(endpoint=instrument_to):
                result = asyncio.run(_run())
        else:
            result = asyncio.run(_run())
    except DocketError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    if isinstance(result, TriageResult):
        click.echo(result.report_markdown)
        if result.review_outcomes:
            # The report was rendered before the interactive review pass;
            # append what actually got posted so the printed output is honest.
            from docket.agent.review import summarize_review_outcomes

            click.echo(summarize_review_outcomes(result.review_outcomes))
        if result.eval_case_paths:
            click.echo(
                f"\nEmitted {len(result.eval_case_paths)} candidate eval case(s) "
                f"to {result.eval_case_paths[0].parent}"
            )
    else:
        click.echo(result)


async def _run_via_deep_agent(
    *,
    backend_adapter: Any,
    rubric: Rubric,
    llm_provider: Any,
    since_dt: datetime,
    until_dt: datetime,
    batch_size: int,
    concurrency: int,
    write_annotations: bool,
    run_id: str | None,
    queue_dir: Path | None,
    embedding_provider: Any = None,
    backend_id: str = "phoenix",
) -> str:
    from langchain_core.messages import HumanMessage

    from docket.agent.deep_agent import (
        build_triage_agent,
        extract_report_markdown,
    )
    from docket.llm import DEFAULT_EMBEDDING_URI

    if embedding_provider is None:
        embedding_provider = build_embedding_provider(DEFAULT_EMBEDDING_URI)
    # Credential preflight (design §4.4): abort before any backend I/O.
    llm_provider.preflight()
    embedding_provider.preflight()
    deep_agent, _state = build_triage_agent(
        backend=backend_adapter,
        rubric=rubric,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        since=since_dt,
        until=until_dt,
        run_id=run_id,
        output_dir=queue_dir,
        write_annotations=write_annotations,
        batch_size=batch_size,
        concurrency=concurrency,
        backend_id=backend_id,
    )
    instruction = (
        f"Triage traces between {since_dt.isoformat()} and {until_dt.isoformat()}. "
        f"Run the full workflow (list_traces -> classify_traces -> "
        f"{'annotate_classifications -> ' if write_annotations else ''}"
        f"cluster_classifications -> draft_issues_tool -> write_report). "
        f"Stop after write_report."
    )
    final_state = await deep_agent.ainvoke({"messages": [HumanMessage(content=instruction)]})
    return extract_report_markdown(final_state) or "(deep agent run produced no /report.md)"


@main.command("self-test")
@click.argument("source", type=str)
@click.option(
    "--batch",
    "batch_size",
    default=1,
    type=click.IntRange(1, 32),
    help="Traces per llm_judge LLM call (budget mode). >1 batches multiple "
    "traces into one structured-output call, cutting cost roughly "
    "proportionally at some accuracy risk for long traces. Default: 1.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM provider for llm_judge modes that don't set their own `model:`",
)
@click.option(
    "--model",
    default=None,
    help="Override the provider's default model",
)
def self_test_cmd(source: str, batch_size: int, provider: str, model: str | None) -> None:
    """Exercise a rubric's examples against the live detectors.

    Phase 2 self-test only runs `llm_judge` examples; other detector types are
    reported as skipped.
    """
    resolved = _normalize_cli_source(source)
    try:
        rubric = load_rubric(resolved)
    except DocketError as e:
        click.echo(f"INVALID: {e}", err=True)
        sys.exit(1)

    default_model = (
        model
        or {
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "openai": DEFAULT_OPENAI_MODEL,
        }[provider]
    )
    default_provider = build_provider(f"{provider}:{default_model}")

    results = asyncio.run(run_self_test(rubric, default_provider, batch_size=batch_size))
    failures = 0
    skips = 0
    for r in results:
        if r.skipped:
            marker = "SKIP"
            skips += 1
        elif r.passed:
            marker = "PASS"
        else:
            marker = "FAIL"
            failures += 1
        label = f"{r.mode_id}[{r.example_index}]" if r.example_index >= 0 else r.mode_id
        click.echo(f"{marker} {label}: {r.message}")
    click.echo(
        f"\nSummary: {len(results) - failures - skips} passed, {failures} failed, {skips} skipped"
    )
    if failures:
        sys.exit(1)


@main.group()
def queue() -> None:
    """Inspect and replay locally queued issue drafts.

    Drafts land in the queue when no tracker is configured, when their
    severity is below `auto_post_threshold`, or when a tracker write
    failed mid-run. `queue post` is the replay half of that contract.
    """


_QUEUE_DIR_OPTION = click.option(
    "--queue-dir",
    "queue_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Queue directory. Default: ~/.docket/queued-issues/.",
)


@queue.command("list")
@_QUEUE_DIR_OPTION
def queue_list(queue_dir: Path | None) -> None:
    """List queued drafts (cluster, severity, mode, title)."""
    from docket.queue_store import list_queued_drafts

    queued = list_queued_drafts(queue_dir)
    if not queued:
        click.echo("Queue is empty.")
        return
    for q in queued:
        d = q.draft
        click.echo(f"{d.cluster_id}  [{d.severity}]  {d.mode_id}  {d.title}")
    click.echo(f"\n{len(queued)} draft(s) queued.")


@queue.command("post")
@_QUEUE_DIR_OPTION
@click.option(
    "--cluster",
    "cluster_ids",
    multiple=True,
    help="Only post drafts with these cluster IDs (repeatable). Default: all.",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Post without per-draft confirmation prompts.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default="docket.yaml",
    help="Path to docket.yaml (used for tracker settings).",
)
@click.option(
    "--tracker",
    type=click.Choice(["jira", "linear", "github"]),
    default=None,
    help="Issue tracker to post to. Defaults to the tracker in the config.",
)
@click.option("--jira-host", default=None)
@click.option("--jira-project", default=None)
@click.option("--jira-email", default=None)
@click.option("--jira-api-token", default=None)
@click.option("--jira-pat", default=None)
@click.option("--jira-deployment", type=click.Choice(["cloud", "datacenter"]), default=None)
@click.option("--linear-api-key", default=None)
@click.option("--linear-team", "linear_team_id", default=None)
@click.option("--linear-endpoint", default=None)
@click.option("--github-token", default=None)
@click.option("--github-owner", default=None)
@click.option("--github-repo", default=None)
@click.option("--github-api-url", default=None)
def queue_post(  # noqa: PLR0913 -- tracker options form one logical unit
    queue_dir: Path | None,
    cluster_ids: tuple[str, ...],
    yes: bool,
    config_path: Path,
    tracker: str | None,
    jira_host: str | None,
    jira_project: str | None,
    jira_email: str | None,
    jira_api_token: str | None,
    jira_pat: str | None,
    jira_deployment: str | None,
    linear_api_key: str | None,
    linear_team_id: str | None,
    linear_endpoint: str | None,
    github_token: str | None,
    github_owner: str | None,
    github_repo: str | None,
    github_api_url: str | None,
) -> None:
    """Post queued drafts to the tracker, then retire them to posted/.

    Each successfully posted draft's files move into the queue's `posted/`
    subdirectory so a replay can never double-post. Failures leave the
    draft in place and continue with the next one.
    """
    from docket.errors import TrackerError
    from docket.queue_store import list_queued_drafts, mark_posted

    _configure_logging(quiet=False, verbose=0)
    try:
        config = _maybe_load_config(config_path)
        tracker_adapter = build_tracker(
            tracker_name=tracker,
            config=config,
            jira_host=jira_host,
            jira_project=jira_project,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            jira_pat=jira_pat,
            jira_deployment=jira_deployment,
            linear_api_key=linear_api_key,
            linear_team_id=linear_team_id,
            linear_endpoint=linear_endpoint,
            github_token=github_token,
            github_owner=github_owner,
            github_repo=github_repo,
            github_api_url=github_api_url,
        )
    except DocketError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    if tracker_adapter is None:
        click.echo(
            "ERROR: No tracker configured. Pass --tracker (plus its credentials) "
            "or set `tracker:` in the config.",
            err=True,
        )
        sys.exit(1)

    queued = list_queued_drafts(queue_dir)
    if cluster_ids:
        wanted = set(cluster_ids)
        queued = [q for q in queued if q.draft.cluster_id in wanted]
    if not queued:
        click.echo("Nothing to post.")
        asyncio.run(tracker_adapter.close())
        return

    async def _post_all() -> tuple[int, int]:
        posted = 0
        failed = 0
        try:
            for q in queued:
                d = q.draft
                if not yes and not click.confirm(
                    f"Post {d.cluster_id} [{d.severity}] {d.title!r}?", default=False
                ):
                    click.echo(f"skipped {d.cluster_id}")
                    continue
                try:
                    issue = await tracker_adapter.create_issue(d)
                except TrackerError as e:
                    failed += 1
                    click.echo(f"FAILED {d.cluster_id}: {e}", err=True)
                    continue
                mark_posted(q, issue_url=issue.url)
                posted += 1
                click.echo(f"posted {d.cluster_id} -> {issue.url or issue.id}")
        finally:
            await tracker_adapter.close()
        return posted, failed

    posted, failed = asyncio.run(_post_all())
    click.echo(f"\n{posted} posted, {failed} failed, {len(queued) - posted - failed} skipped.")
    if failed:
        sys.exit(1)


@queue.command("clear")
@_QUEUE_DIR_OPTION
@click.option("--yes", is_flag=True, default=False, help="Clear without confirmation.")
def queue_clear(queue_dir: Path | None, yes: bool) -> None:
    """Delete all queued (non-posted) drafts."""
    from docket.queue_store import clear_queue, list_queued_drafts

    count = len(list_queued_drafts(queue_dir))
    if count == 0:
        click.echo("Queue is empty.")
        return
    if not yes and not click.confirm(f"Delete {count} queued draft(s)?", default=False):
        click.echo("Aborted.")
        return
    removed = clear_queue(queue_dir)
    click.echo(f"Removed {removed} draft(s).")


def _normalize_cli_source(source: str) -> Path | str:
    """Normalize a CLI source string into something the rubric layer accepts.

    Builtin URIs and `file://` URIs pass through; everything else is treated
    as a path so file-not-found is surfaced cleanly.
    """
    if is_builtin_uri(source) or source.startswith("file://"):
        return source
    return Path(source)


def _maybe_load_config(config_path: Path) -> Config | None:
    if not config_path.exists():
        return None
    return Config.from_yaml(config_path)


def _resolve_rubric(rubric_source: str | None, config: Config | None) -> Rubric:
    source: str | None = rubric_source
    if source is None and config is not None:
        source = config.rubric
    if not source:
        raise ConfigError("No rubric specified. Pass --rubric or set `rubric:` in the config.")
    return load_rubric(_normalize_cli_source(source))


_DURATION_RE = re.compile(r"^(\d+)([smhd])$")
_DURATION_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def _parse_duration(s: str) -> timedelta:
    match = _DURATION_RE.match(s)
    if not match:
        raise click.UsageError(
            f"Expected a duration like '1h', '30m', '7d' or an ISO-8601 "
            f"timestamp like '2026-06-01T00:00:00Z' (got {s!r})."
        )
    num = int(match.group(1))
    unit = match.group(2)
    return timedelta(**{_DURATION_UNITS[unit]: num})


def _parse_window_point(s: str, *, now: datetime) -> datetime:
    """Parse one --since/--until value: absolute ISO-8601 first, then a
    duration anchored to `now`. A trailing 'Z' means UTC; naive timestamps
    are assumed UTC."""
    iso = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return now - _parse_duration(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _resolve_auto_post_threshold(
    cli_value: str | None,
    config: Config | None,
) -> Any:
    """CLI flag overrides `auto_post_threshold` from the config; default 'never'."""
    if cli_value is not None:
        return cli_value
    if config is not None:
        return config.auto_post_threshold
    return "never"


def _resolve_window(since: str, until: str | None) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    since_dt = _parse_window_point(since, now=now)
    until_dt = now if until is None else _parse_window_point(until, now=now)
    if until_dt <= since_dt:
        raise click.UsageError(
            f"--until ({until_dt.isoformat()}) must be after --since ({since_dt.isoformat()})."
        )
    return since_dt, until_dt


# ----------------------------------------------------------------------------
# Daemon mode: scheduled triage over tiling time windows (serve command +
# its shared-options / per-tick resolution infrastructure).
# ----------------------------------------------------------------------------

_SHARED_PIPELINE_OPTIONS: list[Any] = [
    click.option(
        "--rubric",
        "rubric_source",
        type=str,
        default=None,
        help="Rubric source (path, file:// URI, or docket.dev/builtin/...). "
        "Overrides the rubric field in the config.",
    ),
    click.option(
        "--config",
        "config_path",
        type=click.Path(exists=False, dir_okay=False, path_type=Path),
        default="docket.yaml",
        help="Path to docket.yaml. May be absent if --backend + --phoenix-url are given.",
    ),
    click.option(
        "--backend",
        type=click.Choice(["phoenix", "langfuse", "langsmith"]),
        default=None,
        help="Override the trace backend. Supports: phoenix, langfuse, langsmith.",
    ),
    click.option(
        "--phoenix-url",
        default=None,
        help="Phoenix base URL (e.g. http://localhost:6006). Overrides PHOENIX_URL in config.",
    ),
    click.option(
        "--phoenix-api-key",
        default=None,
        help="Phoenix API key. Overrides PHOENIX_API_KEY in config.",
    ),
    click.option(
        "--langfuse-host",
        default=None,
        help="Langfuse host URL (e.g. http://localhost:3000). Overrides LANGFUSE_HOST in config.",
    ),
    click.option(
        "--langfuse-public-key",
        default=None,
        help="Langfuse public key. Overrides LANGFUSE_PUBLIC_KEY in config.",
    ),
    click.option(
        "--langfuse-secret-key",
        default=None,
        help="Langfuse secret key. Overrides LANGFUSE_SECRET_KEY in config.",
    ),
    click.option(
        "--langsmith-api-key",
        default=None,
        help="LangSmith API key. Overrides LANGSMITH_API_KEY in config.",
    ),
    click.option(
        "--langsmith-endpoint",
        default=None,
        help="LangSmith API endpoint (defaults to https://api.smith.langchain.com). "
        "Overrides LANGSMITH_ENDPOINT in config.",
    ),
    click.option(
        "--langsmith-project",
        default=None,
        help="LangSmith project (session) name to filter runs against. "
        "Overrides LANGSMITH_PROJECT in config.",
    ),
    click.option(
        "--tracker",
        type=click.Choice(["jira", "linear", "github"]),
        default=None,
        help="Issue tracker for dedup + posting. Supports: jira, linear, github.",
    ),
    click.option(
        "--jira-host",
        default=None,
        help="Jira host URL (e.g. https://example.atlassian.net). Overrides JIRA_HOST in config.",
    ),
    click.option(
        "--jira-project",
        default=None,
        help="Jira project key (e.g. AGT). Overrides JIRA_PROJECT in config.",
    ),
    click.option(
        "--jira-email",
        default=None,
        help="Atlassian account email for Cloud Basic auth. Overrides JIRA_EMAIL in config.",
    ),
    click.option(
        "--jira-api-token",
        default=None,
        help="Atlassian Cloud API token. Overrides JIRA_API_TOKEN in config.",
    ),
    click.option(
        "--jira-pat",
        default=None,
        help="Jira Data Center Personal Access Token. Overrides JIRA_PAT in config.",
    ),
    click.option(
        "--jira-deployment",
        type=click.Choice(["cloud", "datacenter"]),
        default=None,
        help="Jira deployment type. Default: auto-detect from hostname.",
    ),
    click.option(
        "--linear-api-key",
        default=None,
        help="Linear personal API key. Overrides LINEAR_API_KEY in config.",
    ),
    click.option(
        "--linear-team",
        "linear_team_id",
        default=None,
        help="Linear team ID (the UUID, not the team name/key). Overrides "
        "LINEAR_TEAM_ID in config.",
    ),
    click.option(
        "--linear-endpoint",
        default=None,
        help="Linear GraphQL endpoint. Default: https://api.linear.app/graphql.",
    ),
    click.option(
        "--github-token",
        default=None,
        help="GitHub personal access token (classic or fine-grained). Overrides "
        "GITHUB_TOKEN in config.",
    ),
    click.option(
        "--github-owner",
        default=None,
        help="GitHub repository owner (user or organization). Overrides GITHUB_OWNER in config.",
    ),
    click.option(
        "--github-repo",
        default=None,
        help="GitHub repository name. Overrides GITHUB_REPO in config.",
    ),
    click.option(
        "--github-api-url",
        default=None,
        help="GitHub API base URL (set for GitHub Enterprise Server). Default: https://api.github.com.",
    ),
    click.option(
        "--annotate/--no-annotate",
        default=False,
        help="Write annotations back to the backend. Default: read-only.",
    ),
    click.option("--batch", "batch_size", default=1, type=click.IntRange(1, 32)),
    click.option(
        "--provider",
        type=click.Choice(["anthropic", "openai"]),
        default="anthropic",
        help="LLM provider for llm_judge modes that don't set their own `model:`",
    ),
    click.option(
        "--model",
        default=None,
        help="Override the provider's default model.",
    ),
    click.option(
        "--concurrency",
        default=8,
        type=click.IntRange(1, 64),
        help="Max traces classified in parallel. Default: 8. Lower this (e.g. 1-2) "
        "if your LLM provider tier has a tight requests-per-minute limit; the "
        "classifier issues one structured-output call per (trace, mode) pair.",
    ),
    click.option(
        "--queue-dir",
        "queue_dir",
        type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
        default=None,
        help="Directory for drafted issues. Default: ~/.docket/queued-issues/.",
    ),
    click.option(
        "--instrument-to",
        "instrument_to",
        default=None,
        help="OpenInference instrumentation endpoint. When set, the triage "
        "agent emits OTLP spans for its own run to this URL (typically "
        "http://localhost:6006 for a local Phoenix).",
    ),
    click.option(
        "--auto-post-threshold",
        type=click.Choice(["critical", "high", "medium", "low", "never"]),
        default=None,
        help="Severity at or above which to auto-post new issues to the tracker. "
        "`never` (default unless set in config) keeps every needs-create draft "
        "in the local queue for `--review` or manual inspection.",
    ),
    click.option(
        "--sample",
        "sample_count",
        type=click.IntRange(1, 1_000_000),
        default=None,
        help="Cap the run at N traces, sampled from the window. Sampling is seeded "
        "by the run_id so re-runs with the same inputs sample identically. "
        "Recommended for production windows holding 10k+ traces.",
    ),
    click.option(
        "--strategy",
        "sample_strategy",
        type=click.Choice(["uniform", "stratified", "errors-only"]),
        default="uniform",
        help="Sampling strategy. uniform: seeded random over the listing. "
        "errors-only: only traces whose root run/span ended in error (the "
        "filter is pushed down to the backend listing). stratified: equal "
        "allocation across the strata of --stratify-by, so rare strata "
        "(errors, small tenants, tail latencies) get seen.",
    ),
    click.option(
        "--max-traces",
        "max_traces",
        type=click.IntRange(1, 10_000_000),
        default=None,
        help="Override max_traces_per_run from config for this run. The run "
        "aborts before any trace fetch if the post-sample, post-checkpoint "
        "workload exceeds the cap; silent truncation is forbidden. Default: "
        "the config value, or 1000 without a config file.",
    ),
    click.option(
        "--checkpoint/--no-checkpoint",
        default=False,
        help="Write per-trace sentinel annotations after classification, and on "
        "resume skip traces already marked for this run_id. Requires backend "
        "write access; safe to use alongside --annotate. Recommended for "
        "sub-hourly cron runs where resumability across transient failures "
        "matters.",
    ),
    click.option(
        "--embedding",
        "embedding_uri",
        default=None,
        help="Embedding provider for clustering, as 'provider:model' "
        "(e.g. 'openai:text-embedding-3-small', 'voyage:voyage-3.5-lite', "
        "'local:BAAI/bge-small-en-v1.5'). Default: OpenAI.",
    ),
    click.option(
        "--clustering",
        type=click.Choice(["embedding", "mode-only"]),
        default="embedding",
        show_default=True,
        help="Clustering strategy: 'embedding' (per-mode HDBSCAN, needs an "
        "embedding provider) or 'mode-only' (one cluster per firing mode, "
        "no embeddings needed; lossy).",
    ),
    click.option(
        "-v",
        "--verbose",
        count=True,
        help="Emit DEBUG-level logs (use -vv for even more detail).",
    ),
    click.option(
        "-q",
        "--quiet",
        is_flag=True,
        default=False,
        help="Suppress informational progress output; only warnings and errors.",
    ),
]


def _shared_pipeline_options(f: Any) -> Any:
    """Apply the option set shared by `run` and `serve` (display order preserved)."""
    for option in reversed(_SHARED_PIPELINE_OPTIONS):
        f = option(f)
    return f


@dataclass
class _ResolvedInvocation:
    """Everything `run` resolves once — and `serve` resolves per tick."""

    config: Config | None
    rubric: Rubric
    backend: Any
    tracker: Any
    backend_id: str
    auto_post_threshold: Any
    max_traces: int
    max_cost: float | None
    default_model: str
    llm_provider: Any


def _resolve_invocation(
    *,
    rubric_source: str | None,
    config_path: Path,
    backend: str | None,
    phoenix_url: str | None,
    phoenix_api_key: str | None,
    langfuse_host: str | None,
    langfuse_public_key: str | None,
    langfuse_secret_key: str | None,
    langsmith_api_key: str | None,
    langsmith_endpoint: str | None,
    langsmith_project: str | None,
    tracker: str | None,
    jira_host: str | None,
    jira_project: str | None,
    jira_email: str | None,
    jira_api_token: str | None,
    jira_pat: str | None,
    jira_deployment: str | None,
    linear_api_key: str | None,
    linear_team_id: str | None,
    linear_endpoint: str | None,
    github_token: str | None,
    github_owner: str | None,
    github_repo: str | None,
    github_api_url: str | None,
    auto_post_threshold: str | None,
    max_traces: int | None,
    provider: str,
    model: str | None,
) -> _ResolvedInvocation:
    """Resolve config, rubric, adapters, and provider for one pipeline invocation.

    Raises `DocketError` subclasses on config/credential problems —
    before any I/O, per design §4.4.
    """
    config = _maybe_load_config(config_path)
    rubric = _resolve_rubric(rubric_source, config)
    backend_adapter = build_backend(
        backend_name=backend,
        config=config,
        phoenix_url=phoenix_url,
        phoenix_api_key=phoenix_api_key,
        langfuse_host=langfuse_host,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
    )
    tracker_adapter = build_tracker(
        tracker_name=tracker,
        config=config,
        jira_host=jira_host,
        jira_project=jira_project,
        jira_email=jira_email,
        jira_api_token=jira_api_token,
        jira_pat=jira_pat,
        jira_deployment=jira_deployment,
        linear_api_key=linear_api_key,
        linear_team_id=linear_team_id,
        linear_endpoint=linear_endpoint,
        github_token=github_token,
        github_owner=github_owner,
        github_repo=github_repo,
        github_api_url=github_api_url,
    )
    default_model = (
        model
        or {
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "openai": DEFAULT_OPENAI_MODEL,
        }[provider]
    )
    return _ResolvedInvocation(
        config=config,
        rubric=rubric,
        backend=backend_adapter,
        tracker=tracker_adapter,
        backend_id=resolve_backend_id(backend, config),
        auto_post_threshold=_resolve_auto_post_threshold(auto_post_threshold, config),
        max_traces=_resolve_max_traces(max_traces, config),
        max_cost=_resolve_max_cost(config),
        default_model=default_model,
        llm_provider=build_provider(f"{provider}:{default_model}"),
    )


async def _serve_tick(
    inv: _ResolvedInvocation,
    *,
    since_dt: datetime,
    until_dt: datetime,
    annotate: bool,
    batch_size: int,
    concurrency: int,
    queue_dir: Path | None,
    sample_count: int | None,
    sample_strategy: str,
    checkpoint: bool,
    embedding_uri: str | None = None,
    clustering: str = "embedding",
) -> TriageResult:
    """One serve tick: the deterministic pipeline over [since_dt, until_dt].

    Adapters are owned by the tick — constructed by the caller immediately
    before, closed here — so a long-lived daemon never holds stale
    connections across the sleep.
    """
    try:
        return await run_triage_pipeline(
            backend=inv.backend,
            rubric=inv.rubric,
            since=since_dt,
            until=until_dt,
            llm_provider=inv.llm_provider,
            embedding_provider=(
                build_embedding_provider(embedding_uri) if embedding_uri is not None else None
            ),
            batch_size=batch_size,
            concurrency=concurrency,
            write_annotations=annotate,
            run_id=None,
            backend_id=inv.backend_id,
            output_dir=queue_dir,
            tracker=inv.tracker,
            auto_post_threshold=inv.auto_post_threshold,
            sample_count=sample_count,
            sample_strategy=sample_strategy,
            checkpoint=checkpoint,
            max_traces=inv.max_traces,
            max_estimated_cost_usd=inv.max_cost,
            clustering=clustering,
        )
    finally:
        await inv.backend.close()
        if inv.tracker is not None:
            await inv.tracker.close()


@main.command()
@_shared_pipeline_options
@click.option(
    "--interval",
    default="1h",
    help="Cadence between pipeline runs (e.g. '30m', '1h', '24h'). Each tick "
    "processes the window since the last successful tick, so consecutive "
    "windows tile exactly — no gaps, no overlap.",
)
@click.option(
    "--max-ticks",
    type=click.IntRange(1),
    default=None,
    help="Exit after N ticks. Useful under a supervising scheduler and in "
    "smoke tests; default is to run until interrupted.",
)
def serve(  # noqa: PLR0913 -- CLI options form one logical unit
    rubric_source: str | None,
    config_path: Path,
    backend: str | None,
    phoenix_url: str | None,
    phoenix_api_key: str | None,
    langfuse_host: str | None,
    langfuse_public_key: str | None,
    langfuse_secret_key: str | None,
    langsmith_api_key: str | None,
    langsmith_endpoint: str | None,
    langsmith_project: str | None,
    tracker: str | None,
    jira_host: str | None,
    jira_project: str | None,
    jira_email: str | None,
    jira_api_token: str | None,
    jira_pat: str | None,
    jira_deployment: str | None,
    linear_api_key: str | None,
    linear_team_id: str | None,
    linear_endpoint: str | None,
    github_token: str | None,
    github_owner: str | None,
    github_repo: str | None,
    github_api_url: str | None,
    annotate: bool,
    batch_size: int,
    provider: str,
    model: str | None,
    concurrency: int,
    queue_dir: Path | None,
    instrument_to: str | None,
    auto_post_threshold: str | None,
    sample_count: int | None,
    sample_strategy: str,
    max_traces: int | None,
    checkpoint: bool,
    embedding_uri: str | None,
    clustering: str,
    interval: str,
    max_ticks: int | None,
    verbose: int,
    quiet: bool,
) -> None:
    """Run the triage pipeline on a fixed cadence (daemon mode).

    Tick i processes the window [last successful tick's end, now]; the first
    tick processes the trailing --interval. A failed tick logs the error and
    does NOT advance the window, so the next tick retries the union — no
    traces are silently dropped. Config and credential errors exit
    immediately (they don't fix themselves); stop the daemon with Ctrl-C or
    SIGTERM.

    Interactive flags (--review, --dry-run) and --agent mode are
    deliberately not available here: serve is a production surface and the
    deterministic pipeline is the production execution model.
    """
    _configure_logging(quiet=quiet, verbose=verbose)
    interval_td = _parse_duration(interval)
    log = logging.getLogger("docket.serve")

    def _resolve() -> _ResolvedInvocation:
        return _resolve_invocation(
            rubric_source=rubric_source,
            config_path=config_path,
            backend=backend,
            phoenix_url=phoenix_url,
            phoenix_api_key=phoenix_api_key,
            langfuse_host=langfuse_host,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
            tracker=tracker,
            jira_host=jira_host,
            jira_project=jira_project,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            jira_pat=jira_pat,
            jira_deployment=jira_deployment,
            linear_api_key=linear_api_key,
            linear_team_id=linear_team_id,
            linear_endpoint=linear_endpoint,
            github_token=github_token,
            github_owner=github_owner,
            github_repo=github_repo,
            github_api_url=github_api_url,
            auto_post_threshold=auto_post_threshold,
            max_traces=max_traces,
            provider=provider,
            model=model,
        )

    def _loop() -> None:
        last_until = datetime.now(UTC) - interval_td
        ticks_completed = 0
        while True:
            until_dt = datetime.now(UTC)
            try:
                inv = _resolve()
            except DocketError as e:
                # Config/credential errors are permanent; fail fast rather
                # than burning ticks (design §4.4).
                click.echo(f"ERROR: {e}", err=True)
                sys.exit(1)
            log.info(
                "serve tick %d: window [%s, %s]",
                ticks_completed + 1,
                last_until.isoformat(),
                until_dt.isoformat(),
            )
            try:
                result = asyncio.run(
                    _serve_tick(
                        inv,
                        since_dt=last_until,
                        until_dt=until_dt,
                        annotate=annotate,
                        batch_size=batch_size,
                        concurrency=concurrency,
                        queue_dir=queue_dir,
                        sample_count=sample_count,
                        sample_strategy=sample_strategy,
                        checkpoint=checkpoint,
                        embedding_uri=embedding_uri,
                        clustering=clustering,
                    )
                )
            except DocketError as e:
                log.error(
                    "serve tick failed: %s -- window [%s, %s] will be retried on the next tick",
                    e,
                    last_until.isoformat(),
                    until_dt.isoformat(),
                )
            else:
                click.echo(result.report_markdown)
                last_until = until_dt
            ticks_completed += 1
            if max_ticks is not None and ticks_completed >= max_ticks:
                log.info("serve: reached --max-ticks %d, exiting", max_ticks)
                return
            next_fire = until_dt + interval_td
            delay = (next_fire - datetime.now(UTC)).total_seconds()
            if delay > 0:
                time.sleep(delay)

    try:
        if instrument_to:
            from docket.observability import configure_instrumentation

            with configure_instrumentation(endpoint=instrument_to):
                _loop()
        else:
            _loop()
    except KeyboardInterrupt:
        click.echo("serve: interrupted, exiting cleanly.", err=True)


def _resolve_max_traces(cli_value: int | None, config: Config | None) -> int:
    """`--max-traces` overrides `max_traces_per_run` from the config; the cap
    defaults to 1000 even without a config file (design §8.1 decision 5)."""
    if cli_value is not None:
        return cli_value
    if config is not None:
        return config.max_traces_per_run
    return DEFAULT_MAX_TRACES_PER_RUN


def _resolve_max_cost(config: Config | None) -> float | None:
    """`max_estimated_cost_usd` is config-only; None = no dollar gating."""
    if config is not None:
        return config.max_estimated_cost_usd
    return None

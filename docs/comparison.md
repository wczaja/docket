# How docket compares

Automatic failure detection over LLM agent traces got crowded in
2025–26: LangSmith shipped Insights and then Engine, Galileo (now part
of Cisco) shipped its Insights Engine and renamed it Signals, Braintrust
shipped Topics, Latitude relaunched around agent issues. If you're
evaluating docket, you should know exactly where it sits in that field —
and where it deliberately doesn't compete.

> **Honesty header.** Everything below is stated at the capability
> level, verified against official docs and changelogs **as of July
> 2026**, with dates on the claims most likely to drift. These products
> move monthly; if a row is stale or unfair, please
> [file an issue](https://github.com/wczaja/docket/issues). Each section
> says what the product does *better* than docket, because every one of
> them does more than docket somewhere — docket is a thin triage
> runtime, not a platform.

## Three questions sort this space

1. **Where must your traces live** for the analysis to run?
2. **Where does the failure taxonomy live**, and who can version it?
3. **Where do the findings go** when a failure recurs?

docket's answers: *wherever they already are* (it reads Phoenix,
Langfuse, and LangSmith in place); *in a YAML file in your repo*,
reviewed and versioned like code; *into your tracker* (Jira, Linear,
GitHub Issues) as deduplicated drafts with a human in the loop by
default. As of mid-2026, no product below shares more than one of those
answers.

| | Analyzes traces in your existing backend | Taxonomy as versioned files in your repo | Files issues to Jira / Linear / GitHub Issues | Open source |
|---|---|---|---|---|
| **docket** | ✅ Phoenix, Langfuse, LangSmith | ✅ YAML, composable, semver'd | ✅ deduplicated drafts, human-in-the-loop default | ✅ Apache-2.0 |
| LangSmith Insights + Engine | ❌ LangSmith-resident traces | ❌ saved in-platform config | ❌ in-platform issue dashboard; Engine opens GitHub *PRs* | ❌ commercial (OSS SDKs) |
| Galileo Signals (Cisco/Splunk) | ❌ Galileo-resident traces | ❌ built-in failure-mode catalog | ❌ Slack/email/webhook alerts | ❌ commercial (OSS SDK) |
| Latitude | ❌ OTLP ingested into Latitude | ❌ auto-discovered, lives in platform DB | ❌ Slack/email/webhook | ✅ LGPL-3.0 |
| Braintrust Topics | ❌ Braintrust-resident logs | ❌ in-platform facets | ❌ Slack/webhook | ❌ commercial |
| Arize Phoenix (OSS) | — it *is* a backend docket reads | ❌ no failure clustering in OSS | ❌ no alerting in OSS | ✅ ELv2 |
| Eval frameworks (DeepEval, agentevals, …) | — dataset/CI-oriented | ❌ | ❌ | ✅ libraries |

The pattern behind the first column: **ingestion is OTel-neutral
everywhere, analysis is platform-bound everywhere.** Every platform
accepts OTLP in; each analyzes only traces resident in its own store.
docket inverts that — it's the analysis that travels to the data.

---

## LangSmith — Insights Agent + LangSmith Engine

The deepest automation in the field. **Insights** (GA October 2025)
clusters traces into an auto-generated hierarchical categorization with
frequency, error, latency, and cost breakdowns; you can pin the
top-level category names, and discovered categories persist across
scheduled reports. **Engine** (June 2026) closes the loop entirely
in-platform: recurring failures are detected on a 6-hour scan, root
causes diagnosed, an issue dashboard tracks lifecycle
(resolve/ignore/reopen-on-regression), an evaluator is deployed to catch
regressions — and it can open a pull request in your connected repo with
a proposed fix.

**Structural differences from docket:** analysis runs only over traces
in LangSmith (the SDK path for external data *uploads* it into a
LangSmith project); categories live in platform config, not in your
repo; findings land in LangSmith's own dashboard and GitHub *PRs* — not
in Jira/Linear/GitHub Issues where your team's queue lives. Commercial
(Plus/Enterprise for Insights; Engine priced in compute units;
self-hosting is an Enterprise add-on).

**Choose LangSmith's built-ins when** you're all-in on LangSmith,
want maximum automation over configuration, and your triage output can
live inside LangSmith. **docket reads LangSmith as one of its
backends** — running docket over LangSmith traces to file Jira tickets
from a git-versioned taxonomy is a supported, first-class path.

## Galileo — Insights Engine → Signals (now Cisco / Splunk)

Galileo's Insights Engine (June 2025, GA July 2025 with a free tier,
upgraded and renamed **Signals** in January 2026) classifies agent
failures against a curated, research-informed failure-mode catalog,
attaches root-cause links to exact traces, suggests remediations, and
can generate an optimized metric from an observed failure. Luna-2, its
fine-tuned small-model evaluators, make always-on evaluation cheap at
enterprise scale. Cisco announced the acquisition April 2026 (closed
May 2026); it's being positioned under Splunk observability as "Splunk
Agent Observability, powered by Galileo," including an on-prem offering.

**Structural differences:** traces must be logged into Galileo; the
failure-mode catalog is built-in (adaptive, but not a file you own,
review, or version); findings surface in-platform with Slack/webhook
alerts — no tracker integration (its MCP server lets *your* agent pull
signals out, which is a bridge, not a queue).

**Choose Galileo/Splunk when** you want a managed, enterprise
observability suite with failure detection included and Splunk is
already your operational home. **docket's angle:** your taxonomy and
triage policy stay yours, in git, portable across whichever platform
consolidation lands you on next.

## Latitude — the closest open-source neighbor

Latitude relaunched in May 2026 as "Sentry for agents": open source
(LGPL-3.0 + commercial licensing), self-hostable, OTLP-native. Failed
traces are grouped into tracked **Issues** with a real lifecycle
(New / Escalating / Resolved / Regressed / Ignored); ~10 built-in
"flaggers" annotate known failure categories; from any issue you can
one-click generate an evaluation aligned to human annotations, and its
GEPA integration (the reflective prompt-evolution algorithm,
arXiv:2507.19457) auto-optimizes prompts and eval scripts against those
annotations. That closed loop — issue → aligned eval → optimized
prompt — is genuinely ahead of anything docket does.

**Structural differences:** Latitude is a *platform* — you ingest
traces into it and its issue taxonomy is discovered automatically and
stored in its database. docket is a *runtime* — no storage, no UI, no
ingestion; the taxonomy is a reviewable YAML artifact you write, and
findings go to the tracker your team already triages in. Philosophies
more than feature deltas: auto-discovered taxonomy vs. declared
taxonomy; own-platform lifecycle vs. your-tracker lifecycle.

**Choose Latitude when** you want a full self-hostable monitoring
product with a UI and automatic issue discovery, and you're happy
adopting its store as the home for triage. **Choose docket when** the
taxonomy itself is an asset you want under code review, and triage must
land in Jira/Linear/GitHub next to everything else your team ships.

## Arize Phoenix — complement, not competitor

Phoenix is the open-source (ELv2) observability backend — storage,
tracing UI, datasets, experiments, and a strong client-side evals
library. It has no automatic failure clustering and no alerting in the
OSS product (those live in Arize AX, the commercial sibling). None of
that is a knock: **Phoenix is docket's default backend and the
5-minute-quickstart pairing.** If you're choosing between them, you're
holding the map upside down — docket assumes you run something like
Phoenix.

## Traceloop / OpenLLMetry (now ServiceNow)

OpenLLMetry is the Apache-2.0 OTel instrumentation layer — genuinely
backend-neutral on the *write* side (exports to 20+ backends), and a
good way to get traces into Phoenix or Langfuse for docket to read. The
commercial platform adds LLM-judge monitors and threshold alerts on
ingested spans; no issue clustering or tracker filing that we could
verify. ServiceNow acquired Traceloop in March 2026 — roadmap
uncertainty applies.

## Eval frameworks — DeepEval, Braintrust, agentevals

A different layer of the stack: they score outputs, primarily
pre-production or in CI. DeepEval (Apache-2.0) with its Confident AI
cloud; Braintrust (commercial; its **Topics** feature — GA June 2026 —
does real embedding-based failure clustering, but over logs in
Braintrust); `agentevals` (MIT) is a trajectory-scoring library with no
production loop at all. None reads your existing backend in place,
maintains taxonomies as files, or files tracker issues. docket borrows
their spirit at a different point in the lifecycle — its `--emit-evals`
flag exports confirmed failure clusters as candidate regression cases
*for* these frameworks.

## Also in the space (one-liners, as of mid-2026)

- **Patronus Percival** — agent-trace debugger with a fixed 20+ mode
  taxonomy (TRAIL); platform-bound, commercial.
- **Raindrop** — "Sentry for AI": implicit-signal issue detection over
  production conversations; commercial SaaS.
- **W&B Weave** — monitors with automatic error-signal categorization
  (May 2026); platform-bound.
- **Datadog LLM Observability** — trace cluster map + anomaly insights;
  the big-APM analogue.
- **Langfuse** — docket backend; no native failure clustering as of
  mid-2026 (evals and manual analysis instead).

## What docket deliberately is not

No trace storage, no dashboards, no web UI, no eval execution at scale,
no prompt optimization. Those are exactly the things the platforms
above are good at — docket is designed to sit on top of one of them and
stay small: read traces where they are, classify against the taxonomy
in your repo, cluster, and draft deduplicated issues where your team
already works, with a human approving by default. If that thin slice is
the part you're missing, that's the fit.

*Claims verified against official documentation, changelogs, and
release notes, July 2026. Corrections welcome — file an issue with a
source link.*

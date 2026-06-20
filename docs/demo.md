# Recording a demo

A short, reproducible end-to-end demo of `docket` for a README clip, a
`Show HN` post, or a subreddit. It runs entirely on the project's existing
acceptance mechanism — the same 60-trace fixture and ingest script the gated
integration tests use — so the run is deterministic and you can rehearse it as
many times as you like.

The driver script is [`scripts/demo.sh`](../scripts/demo.sh). It steps through
four beats, one per ENTER press, so you can narrate (or cut) between them.

---

## What it shows

| # | Beat | On screen | ~time |
| - | ---- | --------- | ----- |
| 1 | **Seed** | `ingest_acceptance_traces.py` prints a 60-line `OK …` manifest; optional cut to the Phoenix UI at `localhost:6006` | 0:06–0:18 |
| 2 | **Triage (read-only)** | `docket run` streams progress (`listed 60 trace ids` → `classified 60/60` → `produced 5 clusters` → `drafted 5 issues`), then renders the **Frequency by mode** + **Clusters** report | 0:18–0:42 |
| 3 | **Post to GitHub** | Same command `+ --tracker github --auto-post-threshold high` → cut to the repo's Issues tab: **4 issues** appear, labeled `docket`, `mode:*`, `rubric:agents-builtin@1.0.0` | 0:42–1:05 |
| 4 | **Re-run = no-op** | Re-run the identical command → `## Tracker dedup` shows `action=skipped` on every row; zero new issues | 1:05–1:22 |

The numbers are deterministic: the fixture seeds five modes — `hallucination`
(critical), `infinite-loop`, `premature-termination`, `unsafe-tool-call`
(high), `refusal-leakage` (medium), eight traces each. With the built-in
rubric's `min_cluster_size: 3` that yields **5 clusters / 5 drafts**.
`--auto-post-threshold high` posts the **four** at high or critical severity
and leaves `refusal-leakage` (medium) in the local queue — a one-line point
about severity gating. (`bad-handoff` is in the rubric but unseeded, so it
shows zero positives.)

---

## Prerequisites

Docker Desktop, Python 3.11+, an Anthropic key (the `llm_judge` detectors), and
an OpenAI key (clustering embeddings). A run over this fixture costs a few cents.

```bash
# 1. Install
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"
docket --version

# 2. Local trace backend (ships in docker-compose.yml)
docker compose up -d phoenix      # UI + OTLP at http://localhost:6006
docker compose ps                 # wait for "healthy"

# 3. A throwaway GitHub repo for the drafts (e.g. create `docket-demo`)
#    plus a fine-grained PAT scoped to it with Issues: Read and write.

# 4. Credentials (in the terminal you'll record — clear scrollback first)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GITHUB_TOKEN="github_pat_..."
export GITHUB_OWNER="your-gh-username"
export GITHUB_REPO="docket-demo"
```

> Do a full dry run off-camera first: it warms Docker, confirms `--since 1h`
> catches the freshly-ingested traces, and confirms the 4-issue post. Then
> [close the issues](#cleanup) so the recorded run posts into an empty tab.

For the backend/tracker mechanics behind each step, see
[`docs/local-phoenix.md`](local-phoenix.md) and the full
[end-to-end testing guide](e2e-testing.md).

---

## Run it

```bash
./scripts/demo.sh
```

Press ENTER to advance between the four beats. Overrides via env vars:
`PHOENIX_URL`, `DOCKET_DEMO_RUBRIC`, `DOCKET_DEMO_CONCURRENCY` (default 8, to
shorten the API wait), `DOCKET_DEMO_SINCE` (default `1h`).

If the read-only run reports `Pulled 0 traces`, the `--since` window didn't
overlap the ingest — widen it: `DOCKET_DEMO_SINCE=24h ./scripts/demo.sh`.

---

## Recording (macOS)

- **Polished:** [Screen Studio](https://screen.studio) — auto-zoom on the
  cursor and automatic keystroke captions; the format travels well on social.
- **Free:** QuickTime Player (`Cmd-Shift-5` to record a region) → trim and
  speed-ramp in iMovie. Bump the terminal font to ~18–20pt so it survives
  downscaling.
- **Terminal-only companion:** [asciinema](https://asciinema.org)
  (`asciinema rec`) captures the session as selectable text — great to embed in
  the README, though it can't show the GitHub tab.

Tips:

- **Speed-ramp the waits.** Each `docket run` re-classifies 60 traces
  (~15–30s of API calls). Cut or 3–6× the "thinking" stretches in the editor;
  keep the table and issue reveals at full speed. That's how ~3 real minutes
  becomes ~75s.
- **The GitHub cut is the payoff.** After beat 3, switch to a pre-opened, empty
  Issues tab and refresh on camera so the four issues pop in; hover one to show
  the labels.
- **Design for muted autoplay.** Social video autoplays silent — lean on the
  script's on-screen comments and add a few short text overlays
  ("read-only by default", "4 issues filed, deduped", "re-run → 0 new").

---

## Posting

- Target ~75 seconds; hard cap 2:00. Shorter over-performs.
- **Reddit:** upload the mp4 *natively* to the subreddit (native video
  outreaches an external link). Fits: r/LLMOps, r/MachineLearning weekend
  threads, r/Python, r/devops. Lead with the problem, not the tool.
- **Hacker News:** `Show HN: docket – triage LLM agent traces into tracker
  issues`. Host the video (or the asciinema cast) and put a short "why I built
  this" as the first comment.

---

## Cleanup

```bash
# Close the demo issues between takes / when done (requires the gh CLI):
gh issue list --repo "$GITHUB_OWNER/$GITHUB_REPO" --label docket --state open \
  --json number --jq '.[].number' \
  | xargs -I{} gh issue close --repo "$GITHUB_OWNER/$GITHUB_REPO" {}

docker compose down          # add -v to wipe the Phoenix volume
# Revoke the demo PAT when you're finished.
```

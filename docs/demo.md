# Recording a demo

A short, reproducible end-to-end demo of `docket` for a README clip, a
`Show HN` post, or a subreddit. It runs on the project's existing acceptance
mechanism — the same 60-trace fixture and ingest script the gated integration
tests use — so the run is deterministic and you can rehearse it as many times
as you like.

The demo uses **LangSmith** as the trace backend and **GitHub Issues** as the
tracker (the combination the [end-to-end testing guide](e2e-testing.md) is
built around). The driver script is [`scripts/demo.sh`](../scripts/demo.sh); it
steps through two beats, one per ENTER press, so you can narrate (or cut)
between them.

> **Prefer a fully-local backend?** The demo uses LangSmith because it needs no
> Docker. To run locally instead, bring up Phoenix with
> `docker compose up -d phoenix`, seed with
> `python scripts/ingest_acceptance_traces.py`, and pass
> `--backend phoenix --phoenix-url http://localhost:6006` to `docket run`. See
> [`local-phoenix.md`](local-phoenix.md).

---

## What it shows

| # | Beat | On screen | ~time |
| - | ---- | --------- | ----- |
| 1 | **Seed** | `ingest_acceptance_traces_langsmith.py` prints a 60-line `OK …` manifest; optional cut to the LangSmith project showing the traces | 0:06–0:18 |
| 2 | **Triage + post** | `docket run … --tracker github --auto-post-threshold high` streams progress (`classified 60/60` → `produced 5 clusters` → `drafted 5 issues`) and renders the **Frequency by mode** + **Clusters** report; then cut to the repo's Issues tab: **4 issues** appear, labeled `docket`, `mode:*`, `rubric:agents-builtin@1.0.0` | 0:18–0:55 |

End on the payoff (~55s). *Optional beat 3 — re-run the identical command to show
idempotent dedup (`## Tracker dedup` all `action=skipped`, zero new issues). Off
by default (`DOCKET_DEMO_RERUN=1` to include it): a "nothing happens" no-op is a
weak closer for a social clip, and the idempotency point lands better as a line
in the post (see [Posting](#posting)). Keep it for a longer or docs cut.*

The numbers are deterministic: the fixture seeds five modes — `hallucination`
(critical), `infinite-loop`, `premature-termination`, `unsafe-tool-call`
(high), `refusal-leakage` (medium), eight traces each. With the built-in
rubric's `min_cluster_size: 3` that yields **5 clusters / 5 drafts**.
Triage is **read-only by default**; `--auto-post-threshold high` is the explicit
opt-in, and even then severity gates what lands in the tracker — it posts the
**four** clusters at high or critical severity and leaves `refusal-leakage`
(medium) in the local queue. (`bad-handoff` is in the rubric but unseeded, so it
shows zero positives.)

---

## Prerequisites

Python 3.11+, a free LangSmith account, an Anthropic key (the `llm_judge`
detectors), and an OpenAI key (clustering embeddings). A run over this fixture
costs a few cents. No Docker required.

```bash
# 1. Install
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"
docket --version

# 2. LangSmith: sign in at https://smith.langchain.com, create an API key
#    (Settings → API Keys), and a project named `docket-demo` (or let it be
#    created on first ingest).

# 3. A throwaway GitHub repo for the drafts (e.g. create `docket-demo`)
#    plus a fine-grained PAT scoped to it with Issues: Read and write.

# 4. Credentials (in the terminal you'll record — clear scrollback first)
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT="docket-demo"
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GITHUB_TOKEN="github_pat_..."
export GITHUB_OWNER="your-gh-username"
export GITHUB_REPO="docket-demo"
```

> Do a full dry run off-camera first: it confirms `--since 1h` catches the
> freshly-ingested traces (LangSmith indexes asynchronously, so wait a few
> seconds after the ingest) and confirms the 4-issue post. Then
> [close the issues](#cleanup) so the recorded run posts into an empty tab.

For the backend/tracker mechanics behind each step, and the analogous flows for
Phoenix / Langfuse and Jira / Linear, see the
[end-to-end testing guide](e2e-testing.md).

---

## Run it

```bash
./scripts/demo.sh
```

Press ENTER to advance between the two beats. Overrides via env vars:
`DOCKET_DEMO_RERUN=1` (add the optional idempotency beat), `LANGSMITH_PROJECT`,
`DOCKET_DEMO_RUBRIC`, `DOCKET_DEMO_CONCURRENCY` (default 8, to shorten the API
wait), `DOCKET_DEMO_SINCE` (default `1h`).

If the triage run reports `Pulled 0 traces`, the `--since` window didn't
overlap the ingest (or LangSmith is still indexing) — wait a few seconds and
widen it: `DOCKET_DEMO_SINCE=24h ./scripts/demo.sh`.

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

- **Speed-ramp the waits.** The triage run re-classifies 60 traces (~15–30s of
  API calls; the optional re-run adds another). Cut or 3–6× the "thinking"
  stretches in the editor; keep the table and issue reveals at full speed.
  That's how a minute-plus of real time becomes ~50s.
- **The GitHub cut is the payoff.** After beat 2, switch to a pre-opened, empty
  Issues tab and refresh on camera so the four issues pop in; hover one to show
  the labels.
- **Design for muted autoplay.** Social video autoplays silent — lean on the
  script's on-screen comments and add a few short text overlays
  ("read-only by default", "4 issues filed, deduped").

---

## Producing the assets (GIF + MP4)

Two artifacts, two pipelines. The terminal-only **GIF** (asciinema → agg) is the
low-friction README/HN asset; the **MP4** with the live GitHub Issues cut is the
higher-polish Reddit/Twitter asset. Both work muted — no voiceover.

### Shared prep

- Big terminal font (~18–20pt), minimal prompt (`export PS1='$ '`), clear
  scrollback (`Cmd-K`). Keep the window compact (≈ 92×28) so text stays legible
  after downscaling.
- Export the env vars (see [Prerequisites](#prerequisites)); they're passed via
  env, so no secrets render on screen.
- Rehearse once off-camera, then [close the issues](#cleanup) so the recorded
  run posts into an empty tracker.

### A. Terminal GIF — asciinema → agg (README + HN)

No browser needed: the run's `## Tracker dedup` table prints each created issue
with its number and URL, so the payoff is already on screen.

```bash
brew install asciinema agg gifsicle      # one-time

# 1. Record. Sized small for a crisp GIF. A fresh shell starts; run the demo,
#    step through the beats with ENTER, then `exit` to stop the recording.
asciinema rec docs/demo.cast --cols 92 --rows 28 --idle-time-limit 2
#    ./scripts/demo.sh
#    exit

# 2. Preview; re-record if a take is fumbled.
asciinema play docs/demo.cast

# 3. Render the GIF. --speed and --idle-time-limit cut the API-wait dead time.
agg docs/demo.cast docs/demo.gif \
  --font-size 20 --line-height 1.4 --speed 1.4 --idle-time-limit 1 --theme asciinema

# 4. Shrink for the README (aim < ~8 MB).
gifsicle -O3 --lossy=80 --colors 128 docs/demo.gif -o docs/demo.gif
```

Embed in the README — GitHub autoplays and loops GIFs inline:

```markdown
![docket demo](docs/demo.gif)
```

Useful `agg` flags: `--theme` (`asciinema`, `dracula`, `monokai`,
`solarized-dark`, …), `--font-size`, `--speed`, `--idle-time-limit` (cap idle
gaps, seconds), `--fps-cap` (default 30; drop to 24 for a smaller file),
`--last-frame-duration` (hold the final summary card). Full list: `agg --help`
or <https://docs.asciinema.org/manual/agg/>. Bonus HN asset:
`asciinema upload docs/demo.cast` gives a shareable asciinema.org link with
selectable text.

### B. MP4 with the GitHub payoff (Reddit + Twitter/X)

The browser "money shot" can't be a GIF (huge + color-banded), so this path is a
screen recording.

```bash
brew install ffmpeg      # for compression / conversion
```

1. Pre-open an **empty** GitHub Issues tab (after cleanup).
2. Record with **QuickTime** (`Cmd-Shift-5` → record a region around the
   terminal) or **Screen Studio** (auto-zoom + keystroke captions). Run
   `./scripts/demo.sh`; after beat 2 posts, `Cmd-Tab` to the browser, refresh so
   the 4 issues appear, hover one to show the labels. Stop.
3. Edit in **iMovie** (or Screen Studio):
   - Trim dead air; **speed-ramp the API-wait stretches 3–6×**; keep the report
     table and the issue reveal at full speed.
   - Add muted-first text overlays ("read-only by default", "4 issues filed,
     deduped").
4. Export H.264 MP4, 1080p. To compress a raw `.mov`:

```bash
ffmpeg -i raw.mov -vf "scale=1280:-2" -c:v libx264 -crf 23 -preset slow \
  -pix_fmt yuv420p -movflags +faststart docs/demo.mp4
```

Upload the MP4 *natively* to Reddit (don't link out).

### Conversions

```bash
# GIF -> MP4 (e.g. an MP4 of the terminal-only cut):
ffmpeg -i docs/demo.gif -movflags faststart -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" docs/demo.mp4

# MP4 -> high-quality GIF (two-pass palette, if you screen-recorded instead):
ffmpeg -i docs/demo.mp4 -vf "fps=15,scale=900:-1:flags=lanczos,palettegen" -y /tmp/pal.png
ffmpeg -i docs/demo.mp4 -i /tmp/pal.png \
  -filter_complex "fps=15,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse" -y docs/demo.gif
```

---

## Posting

- Target ~60 seconds; hard cap 2:00. Shorter over-performs.
- **Pre-empt the obvious objection.** "Won't an auto-filing agent spam my
  tracker?" Answer it in the post or a pinned comment: re-runs are idempotent —
  docket dedups by labels + embedded provenance, so the same window posts
  nothing twice (safe on a cron). That one line does the job the cut beat-3
  re-run would have; show it on camera only for a longer cut
  (`DOCKET_DEMO_RERUN=1`).
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

# The LangSmith project can be deleted from its Settings tab if you don't want
# to keep the synthetic traces. Revoke the demo GitHub PAT when you're finished.
```

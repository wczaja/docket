# Triage as CI: scheduled runs without a daemon

docket is a one-shot CLI by design — there is no resident daemon to
operate in v1.0. The intended production pattern is a scheduler invoking
`docket run` on a window slightly larger than the cadence (overlap
is safe: re-runs are idempotent via the deterministic `run_id` and
backend annotation upserts).

## GitHub Actions (recommended starting point)

A complete workflow lives at
[`examples/github-actions/triage.yml`](../examples/github-actions/triage.yml).
Copy it into `.github/workflows/` of any repository, set the secrets, and
you have hourly triage with the run report in the job summary:

```yaml
name: docket
on:
  schedule:
    - cron: "5 * * * *"   # hourly at :05
  workflow_dispatch:

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install docket
      - name: Run triage
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   # embeddings (or use --embedding-provider voyage)
          GITHUB_TOKEN: ${{ secrets.TRIAGE_GITHUB_TOKEN }} # PAT with Issues write on the target repo
        run: |
          docket run \
            --backend langsmith \
            --langsmith-api-key "${{ secrets.LANGSMITH_API_KEY }}" \
            --langsmith-project prod-agents \
            --tracker github \
            --github-owner my-org \
            --github-repo agent-issues \
            --rubric docket.dev/builtin/agents/v1 \
            --since 2h \
            --queue-dir ./triage-queue \
            | tee -a "$GITHUB_STEP_SUMMARY"
      - name: Upload queued drafts for human review
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: queued-drafts
          path: ./triage-queue
          if-no-files-found: ignore
```

Notes:

- **Window vs. cadence.** Run hourly with `--since 2h`. The overlap means
  a missed/failed run never leaves a gap; idempotent annotations and
  label-based issue dedup absorb the double coverage.
- **Human-in-the-loop in CI.** Leave `auto_post_threshold` at `never` at
  first: new-issue drafts land in the queue artifact for review
  (`docket queue list` / `docket queue post` locally), while
  comments on *existing* matched issues are posted directly (additive — a
  human is already on the issue). Once the rubric's false-positive rate
  is calibrated, ratchet down with `--auto-post-threshold high`.
- **Cost control.** `max_traces_per_run` (config) aborts runs over the
  cap rather than silently truncating; use `--sample N` for high-volume
  windows and `--dry-run` to preview cost.
- **Eval regression loop.** Add `--emit-evals ./eval-cases` and upload
  the directory as an artifact; each qualifying cluster exports a
  portable JSON candidate regression case for your eval suite.

## cron

```cron
5 * * * *  DOCKET_CONFIG=/etc/docket.yaml \
           docket run --config /etc/docket.yaml --since 2h \
           >> /var/log/docket.log 2>&1
```

Keep secrets in the environment (the config file supports `${VAR}`
interpolation and refuses to run if a referenced variable is unset).

## Airflow / Argo / anything else

Any scheduler that can run a container works the same way: the runtime is
stateless, so there is nothing to coordinate between runs beyond the
backend and tracker themselves. For high-volume deployments, see the
sharding plan in `docs/design.md` §7 Phase 13.

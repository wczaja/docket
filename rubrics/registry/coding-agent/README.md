# coding-agent

Failure-mode rubric for **code-writing agents** — issue-to-PR bots,
refactoring assistants, test writers — whose traces contain the diffs,
commands, and tool calls they made.

```bash
docket run ... --rubric rubrics/registry/coding-agent/v1/rubric.yaml
```

## What it catches

| mode | severity | detector | cost/trace |
|---|---|---|---|
| `fabricated-api` | high | llm_judge | 1 call |
| `test-gaming` | critical | llm_judge | 1 call |
| `destructive-command` | critical | regex | free |
| `premature-completion` | high | llm_judge | 1 call |
| `hardcoded-secret` | critical | regex | free |
| + 6 generic modes | — | imported from `agents/v1` | 3 calls |

## Trace assumptions

- Diffs and shell commands are visible in the trace (LLM messages or
  tool i/o). If your harness truncates tool output aggressively, the
  judges lose their evidence — raise the truncation ceiling for the
  diff/command tools first.

## Tuning knobs

- **`test-gaming`** is the mode worth the most attention: it encodes
  the difference between "made the suite green" and "fixed the bug".
  Feed its false positives back into the prompt (legitimate test
  updates when *behavior intentionally changed* are the boundary
  case). Its examples block is the regression suite for that boundary.
- **`destructive-command` pattern**: matches force-push, hard-reset,
  `rm -rf` against root-ish paths, and work-discarding checkouts. If
  your agent operates in throwaway sandboxes, downgrade severity to
  `medium` instead of deleting the mode — you still want the trend.
- **`hardcoded-secret` pattern**: conservative 16+-char literal
  assigned to key-ish names. Add your org's token prefixes (e.g.
  `sk-`, `ghp_`) for precision.
- **Ratchet path**: the two regex modes are deterministic; auto-post
  them at `critical` once the tool-name lists match your harness. The
  judges stay in review until you've labeled a week of drafts (see
  `docs/calibration/field-guide.md`).

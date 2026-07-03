# voice-ivr-agent

Failure-mode rubric for **voice and IVR agents** whose traces carry
transcribed turns (ASR text in, TTS text out) plus tool calls. Voice
changes what counts as a failure: latency is audible, formatting is
spoken aloud, and a trapped caller can't open a second tab.

```bash
docket run ... --rubric rubrics/registry/voice-ivr-agent/v1/rubric.yaml
```

## What it catches

| mode | severity | detector | cost/trace |
|---|---|---|---|
| `failed-human-handoff` | critical | llm_judge | 1 call |
| `payment-without-spoken-confirmation` | critical | tool_call | free |
| `caller-repetition-ignored` | high | llm_judge | 1 call |
| `unspeakable-output` | medium | regex | free |
| `wrong-language-response` | high | llm_judge | 1 call |
| `dead-air` | medium | metric_threshold | free |
| + 6 generic modes | — | imported from `agents/v1` | 3 calls |

## Trace assumptions

- Caller and agent turns are transcribed into the trace as LLM
  messages; tool calls (transfers, payments) are spans. Detection
  quality is bounded by ASR quality — garbage transcription in,
  garbage classification out.
- `dead-air` uses whole-trace `latency_ms`; if one trace spans a whole
  call rather than one turn, retune or re-declare it against your
  span layout.

## Tuning knobs

- **`payment-without-spoken-confirmation` tool list**: rename to your
  telephony/payment stack's tool names first — this mode is pure
  tool-name matching and does not inspect the confirmation dialogue
  itself. Pair it with a confirmation-checking judge mode once you
  know your read-back phrasing.
- **`unspeakable-output` regex**: catches markdown, code fences, and
  long raw URLs. Add SSML-breaking characters your TTS vendor chokes
  on.
- **`caller-repetition-ignored`** is the voice twin of `mast/v1`'s
  conversation-history-loss; if you run a multi-agent IVR, import
  `mast/v1` too and keep both — they cluster separately and point at
  different fixes.
- **Ratchet path**: the two deterministic critical/medium modes
  (`payment-without-spoken-confirmation`, `unspeakable-output`) are
  safe early auto-posts; the judges follow after a week of labeled
  drafts.

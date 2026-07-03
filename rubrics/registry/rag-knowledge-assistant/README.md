# rag-knowledge-assistant

Failure-mode rubric for **retrieval-grounded Q&A assistants** (help
centers, internal wikis, policy libraries). Composes the `rag/v1`
builtin (off-corpus answers, missing citations, stale retrieval,
context overflow) with the four failure modes that need your corpus in
view to define.

```bash
docket run ... --rubric rubrics/registry/rag-knowledge-assistant/v1/rubric.yaml
```

## What it catches

| mode | severity | detector | cost/trace |
|---|---|---|---|
| `fabricated-citation` | critical | llm_judge | 1 call |
| `false-unanswerable` | high | llm_judge | 1 call |
| `query-question-mismatch` | high | llm_judge | 1 call |
| `conflict-silently-resolved` | medium | llm_judge | 1 call |
| + 4 retrieval modes | — | imported from `rag/v1` | 4 calls |

All eight modes are judge-based — retrieval quality is inherently
semantic — so this is the **most judge-call-intensive registry rubric**
(~8 calls/trace). Use `--sample` and `--dry-run` before pointing it at
a big window, or start with a subset via your own rubric that imports
this one and re-declares unwanted mode ids with cheaper detection.

## Trace assumptions

- Retrieved documents appear in the trace as retriever spans (standard
  OpenInference `retrieval.documents` attributes).
- The retrieval query is visible (retriever span input or a search
  tool's parameters) — required by `query-question-mismatch`.

## Tuning knobs

- **`false-unanswerable`** is the recall-side complement of `rag/v1`'s
  `off-corpus-answer` precision check. If your users complain the bot
  "never knows anything", this is the mode to watch first.
- **`conflict-silently-resolved`** assumes versioned docs coexist in
  one index. If you already filter by version at retrieval time,
  re-declare the mode id with a cheaper `regex` no-op or drop its
  severity to `low`.
- **Ratchet path**: `fabricated-citation` earns auto-posting first —
  false provenance is indefensible and the judge's false-positive rate
  is the lowest of the four (the citation either is in the retrieval
  set or it isn't).

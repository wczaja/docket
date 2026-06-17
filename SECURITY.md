# Security Policy

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately via GitHub's [private vulnerability reporting](https://github.com/wczaja/agent-triage/security/advisories/new)
(preferred), or email **william.czaja@gmail.com** with subject
`[agent-triage security]`. You'll get an acknowledgment within a few
days; fixes for confirmed vulnerabilities are prioritized ahead of all
feature work and credited to the reporter unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---|---|
| 1.x (latest minor) | yes |
| < 1.0 | no |

## What the runtime promises

agent-triage handles three kinds of sensitive material: provider/backend
credentials, trace contents (which may contain end-user data), and
drafted issue text. The guarantees, each enforced by tests:

- **Credentials never traverse logs, annotations, drafted issues, or the
  agent's virtual filesystem.** API keys and tokens are read from
  environment variables or flags, held in memory only by the adapter or
  provider that needs them, and sent only as auth headers. Missing or
  invalid credentials abort the run at startup, naming the missing
  *variable* — never echoing a value.
- **PII redaction before exfiltration points.**
  `agent_triage.observability.redact()` scrubs emails, phone numbers,
  SSNs, and account-number shapes from trace text before it is logged
  and before it is sent to an LLM judge. Deterministic detectors
  (`regex`, `tool_call`, `metric_threshold`) run in-process and see the
  unredacted trace; nothing leaves the process for them.
- **Read-only by default.** The pipeline writes nothing to your
  observability backend unless you pass `--annotate`, posts nothing to
  your tracker unless severity meets your explicit `auto_post_threshold`
  or you approve drafts in `--review`, and writes local files only under
  `~/.agent-triage/` (or your configured queue directory).
- **No network in the validation path.** `agent-triage validate` and
  rubric import resolution (`file://`, packaged builtins) perform no
  network I/O in v1.0.

## What it does not promise

- Redaction is regex-based defense-in-depth, **not** a PII inventory or
  a compliance control. If your traces contain regulated data, scrub at
  the instrumentation layer before traces reach your backend; treat
  agent-triage's redaction as a backstop.
- The LLM judge sends (redacted) trace excerpts to your configured model
  provider. If that's unacceptable for your data, restrict rubrics to
  deterministic detection types.
- Drafted issues quote trace evidence by design. Review drafts before
  posting to trackers visible beyond your trust boundary — that's why
  human-in-the-loop is the default.

## Dependency posture

Runtime dependencies are MIT/BSD/Apache-2.0 only (no copyleft); the set
is curated and additions require maintainer review.

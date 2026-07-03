#!/usr/bin/env python3
"""Render the README hero: an animated terminal SVG of a real `docket demo` run.

Runs `docket demo` in a subprocess, curates the interesting lines, and
emits a self-contained animated SVG (CSS keyframes only — no scripts, no
external assets, animates inside GitHub's <img> sandbox).

Regenerate after output-shaping changes so the hero never drifts from
reality:

    python scripts/render_demo_svg.py                # writes docs/assets/demo.svg
    python scripts/render_demo_svg.py --out other.svg

The recording is *curated but real*: every rendered line (after the
typed command) is matched from actual demo output; nothing is invented.
Trace-id suffixes vary per regeneration — that's fine, they're real.
"""

import argparse
import html
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo.svg"

# --- palette (GitHub-dark-friendly, self-contained) --------------------------
BG = "#0d1117"
CHROME = "#161b22"
BORDER = "#30363d"
FG = "#e6edf3"
DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
YELLOW = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"

FONT = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
LINE_H = 21
FONT_SIZE = 13.5
PAD_X = 22
PAD_TOP = 56
PAD_BOTTOM = 22
WIDTH = 860

_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2} ")


@dataclass
class Pick:
    """One curated output line: match `pattern` in the demo output, render
    with `color`; `gap` inserts a blank line before it."""

    pattern: str
    color: str = FG
    gap: bool = False
    max_len: int = 96


# Ordered curation of the run. Patterns are matched (regex, first hit wins,
# consumed in order) against timestamp-stripped output lines.
PICKS: list[Pick] = [
    Pick(r"^docket demo: 60 synthetic traces", DIM),
    Pick(r"INFO listed \d+ trace ids", DIM),
    Pick(r"INFO classifying \d+ traces against \d+ modes", DIM),
    Pick(r"INFO classified 60/60 traces$", DIM),
    Pick(r"INFO classification complete: \d+ positive", BLUE),
    Pick(r"INFO clustering positive classifications", DIM),
    Pick(r"INFO produced \d+ clusters", BLUE),
    Pick(r"INFO drafted \d+ issues", BLUE),
    Pick(r"^# docket run", FG, gap=True),
    Pick(r"\*\*Traces processed\*\*", FG),
    Pick(r"\*\*Clusters formed\*\*", FG),
    Pick(r"\*\*Issues drafted\*\*", FG),
    Pick(r"^## Frequency by mode", PURPLE, gap=True),
    Pick(r"^\| mode \| severity", DIM),
    Pick(r"^\| `hallucination` \| critical", RED),
    Pick(r"^\| `infinite-loop` \| high", YELLOW),
    Pick(r"^\| `premature-termination` \| high", YELLOW),
    Pick(r"^\| `unsafe-tool-call` \| high", YELLOW),
    Pick(r"^\| `refusal-leakage` \| medium", FG),
    Pick(r"^## Clusters", PURPLE, gap=True),
    Pick(r"### `refusal-leakage` cluster", FG),
    Pick(r"Agent pastes its system prompt into refusal responses", GREEN),
    Pick(r"### `unsafe-tool-call` cluster", FG),
    Pick(r"Destructive tool calls executed without user confirmation", GREEN),
    Pick(r"# Sample drafted issue", PURPLE, gap=True),
    Pick(r"^# Agent (loops|pastes|abandons|asserts)", GREEN),
    Pick(r"\*\*Severity\*\*", FG),
    Pick(r"\*\*Labels\*\*", DIM),
    Pick(r"Drafts and report\.md written to", BLUE, gap=True),
]

COMMAND = "uvx docket-runtime demo"
EPILOGUE = [
    ("", FG),
    ("60 traces in -> 4 deduplicated issue drafts out. No API keys, no Docker,", DIM),
    ("no instrumentation. Your taxonomy is the YAML file in your repo.", DIM),
]


def run_demo() -> list[str]:
    """Run `docket demo` and return its merged, timestamp-stripped lines.

    Runs inside a tempdir with the default relative `--out docket-demo`,
    so the "written to `docket-demo/`" line reads exactly as it would on
    a user's machine (no temp paths in the recording).
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no user input
            [sys.executable, "-m", "docket", "demo", "--out", "docket-demo"],
            capture_output=True,
            text=True,
            cwd=tmp,
            timeout=300,
            check=False,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"docket demo exited {proc.returncode}")
    merged = proc.stderr.splitlines() + proc.stdout.splitlines()
    return [_TIMESTAMP.sub("", line) for line in merged]


def curate(lines: list[str]) -> list[tuple[str, str, bool]]:
    """Match PICKS in order against `lines`; return (text, color, gap) rows.

    Each pick takes the first not-yet-consumed matching line. Display
    order is the PICKS order (the true temporal order of the run), not
    the merged capture order — stdout and stderr are captured as separate
    streams, so their relative interleaving is not preserved.
    """
    out: list[tuple[str, str, bool]] = []
    used: set[int] = set()
    for pick in PICKS:
        regex = re.compile(pick.pattern)
        for i, line in enumerate(lines):
            if i in used or not regex.search(line):
                continue
            text = line.strip()
            if len(text) > pick.max_len:
                text = text[: pick.max_len - 1] + "…"
            out.append((text, pick.color, pick.gap))
            used.add(i)
            break
        else:
            sys.stderr.write(f"note: no line matched {pick.pattern!r}; skipping\n")
    return out


def render_svg(rows: list[tuple[str, str, bool]]) -> str:
    body_rows: list[str] = []
    y = PAD_TOP

    # Typed command: one tspan per character, revealed left to right.
    char_spans = []
    for i, ch in enumerate(COMMAND):
        char_spans.append(
            f'<tspan class="c" style="animation-delay:{0.35 + i * 0.045:.3f}s">'
            f"{html.escape(ch)}</tspan>"
        )
    body_rows.append(
        f'<text x="{PAD_X}" y="{y}" fill="{FG}">'
        f'<tspan fill="{GREEN}">$ </tspan>{"".join(char_spans)}</text>'
    )
    y += int(LINE_H * 1.4)

    t = 0.35 + len(COMMAND) * 0.045 + 0.5  # output starts after the "enter"
    for text, color, gap in rows:
        if gap:
            y += LINE_H // 2
            t += 0.25
        body_rows.append(
            f'<text class="l" style="animation-delay:{t:.2f}s" '
            f'x="{PAD_X}" y="{y}" fill="{color}">{html.escape(text)}</text>'
        )
        y += LINE_H
        t += 0.22
    for text, color in EPILOGUE:
        if not text:
            y += LINE_H // 2
            continue
        t += 0.3
        body_rows.append(
            f'<text class="l" style="animation-delay:{t:.2f}s" '
            f'x="{PAD_X}" y="{y}" fill="{color}">{html.escape(text)}</text>'
        )
        y += LINE_H

    height = y + PAD_BOTTOM
    total = t + 1.0
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}"
     viewBox="0 0 {WIDTH} {height}" font-family="{FONT}" font-size="{FONT_SIZE}">
  <title>docket demo — 60 synthetic traces triaged into deduplicated issue drafts, no credentials</title>
  <style>
    .l, .c {{ opacity: 0; animation: a 0.15s ease-out forwards; }}
    @keyframes a {{ to {{ opacity: 1; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .l, .c {{ animation: none; opacity: 1; }}
    }}
  </style>
  <rect width="{WIDTH}" height="{height}" rx="10" fill="{BG}" stroke="{BORDER}"/>
  <rect width="{WIDTH}" height="36" rx="10" fill="{CHROME}"/>
  <rect y="26" width="{WIDTH}" height="10" fill="{CHROME}"/>
  <circle cx="22" cy="18" r="6" fill="#ff5f57"/>
  <circle cx="42" cy="18" r="6" fill="#febc2e"/>
  <circle cx="62" cy="18" r="6" fill="#28c840"/>
  <text x="{WIDTH // 2}" y="22" text-anchor="middle" fill="{DIM}" font-size="12">docket demo — zero credentials, ~{int(total)}s of your time</text>
{chr(10).join("  " + row for row in body_rows)}
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lines = run_demo()
    rows = curate(lines)
    if len(rows) < 15:
        raise SystemExit(
            f"only {len(rows)} lines matched — demo output shape changed; update PICKS"
        )
    svg = render_svg(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"wrote {args.out} ({len(rows)} curated lines, {len(svg)} bytes)")


if __name__ == "__main__":
    main()

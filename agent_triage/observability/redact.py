"""PII redaction applied to trace data before logs and LLM judge input.

Regex-based scrubbing of common PII shapes. Not a substitute for a proper PII
inventory; meant as a defense-in-depth backstop. Phase 3 expands coverage.
"""

import re
from typing import Final

_EMAIL_RE: Final = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# US phone shapes: optional +1/1 country prefix, area code either
# parenthesized ("(555) 123-4567") or bare with -, ., or space separators,
# plus the original bare-10-digit form. 9-digit runs intentionally don't match.
_PHONE_RE: Final = re.compile(
    r"(?:\+1[-.\s]?|\b1[-.\s])?"  # optional country code
    r"(?:\(\d{3}\)\s?|\b\d{3}[-.\s]?)"  # area code
    r"\d{3}[-.\s]?\d{4}\b"
)
_SSN_RE: Final = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT_RE: Final = re.compile(r"\b(?:account|acct)[#:\s]+(\d{6,})\b", re.IGNORECASE)

_PATTERNS: Final = [
    (_EMAIL_RE, "[REDACTED_EMAIL]"),
    (_SSN_RE, "[REDACTED_SSN]"),
    (_ACCOUNT_RE, "[REDACTED_ACCOUNT]"),
    (_PHONE_RE, "[REDACTED_PHONE]"),
]


def redact(text: str) -> str:
    """Scrub common PII shapes from `text`. Returns a new string with redactions applied."""
    out = text
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out

"""Input validation and secret redaction, applied before anything reaches the LLM."""
from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRETS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                      # API-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),                            # AWS access key id
    re.compile(r"(?i)\b(password|secret|token)\s*[:=]\s*\S+"),  # inline credentials
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                    # emails
]


class InputRejected(ValueError):
    """Raised when the user input cannot be processed safely."""


def sanitize_input(text: str, min_chars: int, max_chars: int) -> tuple[str, int]:
    """Return (clean_text, number_of_redactions) or raise InputRejected."""
    clean = _CONTROL.sub("", text or "").strip()
    if len(clean) < min_chars:
        raise InputRejected(f"Description too short: provide at least {min_chars} characters.")
    if len(clean) > max_chars:
        raise InputRejected(f"Description too long: {len(clean)} chars (limit {max_chars}).")
    redactions = 0
    for pattern in _SECRETS:
        clean, n = pattern.subn("[REDACTED]", clean)
        redactions += n
    return clean, redactions

"""Guidance for writing a good project description: sections, template, and a live coverage check.

Pure Python (no Streamlit) so it can be unit-tested. The coverage check is a keyword heuristic
meant as a nudge for the user, never as a gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    include: str          # what to write here
    example: str
    pattern: re.Pattern   # heuristic: does the text seem to cover this?
    optional: bool = False


def _p(expr: str) -> re.Pattern:
    return re.compile(expr, re.IGNORECASE | re.MULTILINE)


SECTIONS: tuple[Section, ...] = (
    Section("goal", "Goal", "What you are building, who uses it, and the problem it solves.",
            "A retailer wants to cut stockouts by forecasting weekly demand per store.",
            _p(r"\b(build|building|platform|app|application|system|tool|portal|service|want|wants|need|needs|goal|users?)\b")),
    Section("features", "Core features", "The 3-6 things it must do, as short bullets.",
            "Ingest daily sales; forecast 4 weeks ahead; show reorder suggestions.",
            _p(r"\b(must|should|features?|ingest|generate|show|display|allow|support|provide|track|manage|upload|search|export)\b|^\s*[-*]\s+\w")),
    Section("scale", "Scale and performance", "Users, data volume, request rate, and measurable targets.",
            "200 stores, 20,000 SKUs, dashboard p95 under 2 s, nightly job under 2 hours.",
            _p(r"\d[\d,.]*\s*(k|m|gb|tb|ms|s|sec|seconds?|minutes?|hours?|users?|stores?|requests?|records?|rows?|skus?|%)\b|\bp9[59]\b|\bper (day|second|hour|month)\b")),
    Section("constraints", "Constraints", "Fixed technology, cloud preference, team size, budget, deadline.",
            "Python backend, Streamlit frontend, AWS preferred, 3 engineers, budget conscious.",
            _p(r"\b(python|streamlit|java|node|react|aws|azure|gcp|cloud|on-?prem|budget|team|engineers?|deadline|constraints?|must use)\b")),
    Section("integrations", "Integrations", "Systems it must connect to: APIs, databases, files, third parties.",
            "Store POS systems via daily file export or REST API.",
            _p(r"\b(api|apis|integrat\w*|connect\w*|database|erp|crm|pos|sftp|webhook|third[- ]party|salesforce|sap|stripe|csv|files?)\b")),
    Section("security", "Security and access", "Roles, sensitive data, compliance rules.",
            "Store managers see their store; regional managers see their region.",
            _p(r"\b(role|roles|auth\w*|access|permission\w*|security|secure|gdpr|hipaa|pci|pii|compliance|encrypt\w*|sso|login)\b")),
    Section("scope", "Out of scope / open questions", "What you do not want, or have not decided yet.",
            "No real-time inventory in v1. Undecided: build vs buy the forecasting model.",
            _p(r"\b(out of scope|not included|non-goals?|won'?t|will not|no [a-z]+ in v1|undecided|open questions?|tbd|not sure)\b"),
            optional=True),
)

TIPS: tuple[str, ...] = (
    "**Use numbers, not adjectives.** \"200 stores and p95 under 2 s\" can be checked by the reviewer; \"fast and scalable\" cannot.",
    "**State fixed decisions.** If the stack is already chosen, say so, or the agent may propose something else.",
    "**Describe one system.** A description that mixes several products produces a blurry design.",
    "**Say what you don't know.** Unknowns become open questions instead of silent guesses.",
    "**Leave out secrets and personal data.** Keys, passwords and emails are redacted automatically, but don't rely on it.",
)

TEMPLATE = """# <Project name>

## Goal
<What you are building, who uses it, and the problem it solves.>

## Core features
- <Feature 1>
- <Feature 2>
- <Feature 3>

## Scale and performance
- <Users, data volume, request rate>
- <Targets, e.g. responds in under 2 s>

## Constraints
- <Fixed tech, cloud preference, team size, budget, deadline>

## Integrations
- <Systems it must connect to>

## Security and access
- <Roles, sensitive data, compliance>

## Out of scope / open questions
- <What you do not want, or have not decided>
"""

_PLACEHOLDER = re.compile(r"<[A-Z][^<>\n]{2,}>")


def coverage(text: str) -> list[tuple[Section, bool]]:
    """Which sections does the text seem to cover?

    Markdown headings and unfilled template placeholders are ignored: a heading alone is not content.
    """
    body = _PLACEHOLDER.sub("", re.sub(r"(?m)^\s*#.*$", "", text or ""))
    return [(s, bool(s.pattern.search(body))) for s in SECTIONS]


def has_unfilled_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text or ""))


def missing_required(text: str) -> list[Section]:
    return [s for s, ok in coverage(text) if not ok and not s.optional]

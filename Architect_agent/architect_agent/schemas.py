"""Typed contracts between the LLM stages (stdlib only).

Every LLM answer is parsed through `from_dict`, which validates types, enums,
required fields and cross-references. Any failure raises SchemaError, which the
agent feeds back to the model as a repair prompt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .observability import TraceEvent, Usage

LAYERS = ("frontend", "backend", "data", "integration", "infra")
PRIORITIES = ("must", "should", "could")
SEVERITIES = ("high", "medium", "low")


class SchemaError(ValueError):
    """LLM output does not match the expected structure."""


# ---- tiny validation helpers -------------------------------------------------
def _obj(d: Any, where: str) -> dict:
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected an object, got {type(d).__name__}")
    return d


def _str(d: dict, key: str, where: str, default: str | None = None) -> str:
    v = d.get(key, default)
    if not isinstance(v, str) or not v.strip():
        raise SchemaError(f"{where}.{key}: required non-empty string")
    return v.strip()


def _enum(d: dict, key: str, choices: tuple, where: str, default: str | None = None) -> str:
    v = d.get(key, default)
    if v not in choices:
        raise SchemaError(f"{where}.{key}: must be one of {list(choices)}, got {v!r}")
    return v


def _list(d: dict, key: str, where: str, min_len: int = 0) -> list:
    v = d.get(key, [])
    if not isinstance(v, list) or len(v) < min_len:
        raise SchemaError(f"{where}.{key}: expected a list with at least {min_len} item(s)")
    return v


def _strs(d: dict, key: str, where: str) -> list[str]:
    items = _list(d, key, where)
    if not all(isinstance(i, str) for i in items):
        raise SchemaError(f"{where}.{key}: all items must be strings")
    return items


# ---- requirements ------------------------------------------------------------
@dataclass
class Requirement:
    id: str
    text: str
    priority: str = "must"

    @classmethod
    def from_dict(cls, d: Any, where: str = "requirement") -> "Requirement":
        d = _obj(d, where)
        return cls(_str(d, "id", where), _str(d, "text", where),
                   _enum(d, "priority", PRIORITIES, where, "must"))


@dataclass
class RequirementsSpec:
    summary: str
    functional: list[Requirement]
    non_functional: list[Requirement]
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Any) -> "RequirementsSpec":
        d = _obj(d, "requirements")
        return cls(
            _str(d, "summary", "requirements"),
            [Requirement.from_dict(r, "functional") for r in _list(d, "functional", "requirements", 1)],
            [Requirement.from_dict(r, "non_functional") for r in _list(d, "non_functional", "requirements", 1)],
            _strs(d, "constraints", "requirements"),
            _strs(d, "assumptions", "requirements"),
            _strs(d, "open_questions", "requirements"),
        )


# ---- architecture ------------------------------------------------------------
@dataclass
class Component:
    name: str
    layer: str
    responsibility: str
    tech: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class FlowStep:
    source: str
    target: str
    message: str


@dataclass
class KeyFlow:
    name: str
    steps: list[FlowStep]


@dataclass
class ADR:
    title: str
    decision: str
    rationale: str
    alternatives: list[str] = field(default_factory=list)


@dataclass
class Risk:
    description: str
    mitigation: str


@dataclass
class ArchitectureDesign:
    style: str
    overview: str
    components: list[Component]
    key_flows: list[KeyFlow] = field(default_factory=list)
    deployment: list[str] = field(default_factory=list)
    adrs: list[ADR] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Any) -> "ArchitectureDesign":
        d = _obj(d, "design")
        comps = []
        for c in _list(d, "components", "design", 1):
            c = _obj(c, "component")
            comps.append(Component(
                _str(c, "name", "component"), _enum(c, "layer", LAYERS, "component"),
                _str(c, "responsibility", "component"), _str(c, "tech", "component"),
                _strs(c, "depends_on", "component")))
        names = [c.name for c in comps]
        if len(set(names)) != len(names):
            raise SchemaError("design.components: names must be unique")
        for c in comps:
            unknown = [x for x in c.depends_on if x not in names]
            if unknown:
                raise SchemaError(f"component {c.name!r} depends on unknown components {unknown}; "
                                  f"use exact names from design.components")
        flows = []
        for f in _list(d, "key_flows", "design"):
            f = _obj(f, "key_flow")
            steps = [FlowStep(_str(_obj(s, "step"), "source", "step"), _str(s, "target", "step"),
                              _str(s, "message", "step")) for s in _list(f, "steps", "key_flow", 1)]
            flows.append(KeyFlow(_str(f, "name", "key_flow"), steps))
        adrs = []
        for a in _list(d, "adrs", "design"):
            a = _obj(a, "adr")
            adrs.append(ADR(_str(a, "title", "adr"), _str(a, "decision", "adr"),
                            _str(a, "rationale", "adr"), _strs(a, "alternatives", "adr")))
        risks = []
        for r in _list(d, "risks", "design"):
            r = _obj(r, "risk")
            risks.append(Risk(_str(r, "description", "risk"), _str(r, "mitigation", "risk")))
        return cls(_str(d, "style", "design"), _str(d, "overview", "design"), comps, flows,
                   _strs(d, "deployment", "design"), adrs, risks)


# ---- review ------------------------------------------------------------------
@dataclass
class Issue:
    severity: str
    description: str
    suggestion: str


@dataclass
class ReviewReport:
    score: int
    approved: bool
    issues: list[Issue] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Any) -> "ReviewReport":
        d = _obj(d, "review")
        score = d.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
            raise SchemaError("review.score: integer between 0 and 10 required")
        if not isinstance(d.get("approved"), bool):
            raise SchemaError("review.approved: boolean required")
        issues = [Issue(_enum(_obj(i, "issue"), "severity", SEVERITIES, "issue"),
                        _str(i, "description", "issue"), _str(i, "suggestion", "issue"))
                  for i in _list(d, "issues", "review")]
        return cls(score, d["approved"], issues)


def to_dict(obj: Any) -> dict:
    return asdict(obj)


# ---- final result ------------------------------------------------------------
@dataclass
class AgentResult:
    run_id: str
    project_name: str
    requirements: RequirementsSpec
    design: ArchitectureDesign
    review: ReviewReport
    review_rounds: int
    approved: bool
    diagrams: dict[str, str]
    markdown: str
    trace: list[TraceEvent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    redactions: int = 0

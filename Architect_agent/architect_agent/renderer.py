"""Render an architecture result as a shareable Markdown document."""
from __future__ import annotations

from .schemas import ArchitectureDesign, RequirementsSpec, ReviewReport


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "_None._"


def to_markdown(project: str, reqs: RequirementsSpec, design: ArchitectureDesign,
                review: ReviewReport, approved: bool, rounds: int,
                diagrams: dict[str, str]) -> str:
    status = "Approved" if approved else "NOT approved: unresolved review issues remain"
    out = [f"# {project}: Technical Architecture", "",
           f"**Review status:** {status} (score {review.score}/10 after {rounds} review round(s))", "",
           "## 1. Summary", reqs.summary, "", "## 2. Requirements", "", "### Functional",
           "| ID | Requirement | Priority |", "|---|---|---|"]
    out += [f"| {r.id} | {_cell(r.text)} | {r.priority} |" for r in reqs.functional]
    out += ["", "### Non-functional", "| ID | Requirement | Priority |", "|---|---|---|"]
    out += [f"| {r.id} | {_cell(r.text)} | {r.priority} |" for r in reqs.non_functional]
    out += ["", "### Constraints", _bullets(reqs.constraints), "", "### Assumptions",
            _bullets(reqs.assumptions), "", "### Open questions", _bullets(reqs.open_questions), "",
            f"## 3. Architecture ({design.style})", design.overview, "",
            "| Component | Layer | Responsibility | Tech | Depends on |", "|---|---|---|---|---|"]
    out += [f"| {_cell(c.name)} | {c.layer} | {_cell(c.responsibility)} | {_cell(c.tech)} | "
            f"{_cell(', '.join(c.depends_on) or '-')} |" for c in design.components]
    out += ["", "## 4. Diagrams"]
    for title, code in diagrams.items():
        out += ["", f"### {title}", "```mermaid", code, "```"]
    out += ["", "## 5. Deployment", _bullets(design.deployment), "", "## 6. Decisions (ADRs)"]
    for i, a in enumerate(design.adrs, 1):
        out += ["", f"**ADR-{i}: {a.title}**", f"- Decision: {a.decision}", f"- Rationale: {a.rationale}"]
        if a.alternatives:
            out.append(f"- Alternatives considered: {', '.join(a.alternatives)}")
    out += ["", "## 7. Risks", "| Risk | Mitigation |", "|---|---|"]
    out += [f"| {_cell(r.description)} | {_cell(r.mitigation)} |" for r in design.risks]
    out += ["", "## 8. Review findings", "| Severity | Issue | Suggestion |", "|---|---|---|"]
    out += [f"| {i.severity} | {_cell(i.description)} | {_cell(i.suggestion)} |" for i in review.issues]
    return "\n".join(out) + "\n"

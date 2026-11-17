"""Deterministic Mermaid generation from the validated design (never LLM-drawn, so always valid)."""
from __future__ import annotations

import re

from .schemas import ArchitectureDesign, KeyFlow

_LAYER_TITLES = {"frontend": "Frontend", "backend": "Backend", "integration": "Integration",
                 "data": "Data", "infra": "Infrastructure"}


def _id(prefix: str, name: str) -> str:
    return prefix + re.sub(r"\W+", "_", name).strip("_")


def _text(value: str) -> str:
    """Strip characters that break Mermaid labels."""
    return re.sub(r"\s+", " ", re.sub(r'["<>{}\[\]`;#|]', " ", value)).strip()


def component_diagram(design: ArchitectureDesign) -> str:
    lines = ["flowchart LR"]
    for layer, title in _LAYER_TITLES.items():
        members = [c for c in design.components if c.layer == layer]
        if not members:
            continue
        lines.append(f'  subgraph {layer}["{title}"]')
        for c in members:
            lines.append(f'    {_id("c_", c.name)}["{_text(c.name)}<br/>{_text(c.tech)}"]')
        lines.append("  end")
    for c in design.components:
        for dep in c.depends_on:
            lines.append(f'  {_id("c_", c.name)} --> {_id("c_", dep)}')
    return "\n".join(lines)


def sequence_diagram(flow: KeyFlow) -> str:
    lines = ["sequenceDiagram"]
    seen: list[str] = []
    for s in flow.steps:
        for who in (s.source, s.target):
            if who not in seen:
                seen.append(who)
    for who in seen:
        lines.append(f'  participant {_id("p_", who)} as {_text(who)}')
    for s in flow.steps:
        lines.append(f'  {_id("p_", s.source)}->>{_id("p_", s.target)}: {_text(s.message)}')
    return "\n".join(lines)


def build_diagrams(design: ArchitectureDesign) -> dict[str, str]:
    diagrams = {"Component architecture": component_diagram(design)}
    for flow in design.key_flows:
        diagrams[f"Flow: {flow.name}"] = sequence_diagram(flow)
    return diagrams

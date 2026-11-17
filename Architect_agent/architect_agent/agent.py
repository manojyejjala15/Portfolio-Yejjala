"""Core agent: validate -> requirements -> design <-> review loop -> diagrams -> markdown.

Production behaviours implemented here:
  * guardrails on input (size limits, control chars, secret redaction)
  * schema-validated LLM output with a bounded "repair" re-ask loop
  * bounded retries with exponential backoff for transient LLM errors
  * review/revise loop with a hard cap, and an explicit `approved` flag on the result
  * per-run trace (stage timings), token accounting and structured logging
  * no shared mutable state between runs (safe to reuse across Streamlit sessions)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import prompts
from .config import Settings
from .diagrams import build_diagrams
from .guardrails import sanitize_input
from .llm import LLMClient, call_with_retries
from .observability import TraceEvent, Usage, get_logger
from .renderer import to_markdown
from .schemas import (AgentResult, ArchitectureDesign, RequirementsSpec, ReviewReport,
                      SchemaError)

EventCallback = Callable[[TraceEvent], None]


class AgentError(RuntimeError):
    """The agent could not produce a valid result."""


@dataclass
class _Run:
    run_id: str
    usage: Usage = field(default_factory=Usage)
    trace: list[TraceEvent] = field(default_factory=list)


def extract_json(text: str) -> object:
    """Parse the first JSON object in `text`, tolerating code fences and stray prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise SchemaError("no JSON object found in the response")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc}") from exc


class ArchitectAgent:
    def __init__(self, llm: LLMClient, settings: Settings | None = None,
                 on_event: EventCallback | None = None):
        self.llm = llm
        self.settings = settings or Settings()
        self.on_event = on_event
        self.log = get_logger(self.settings.log_level)

    # ---- public API ----------------------------------------------------------
    def run(self, description: str, project_name: str = "Untitled project") -> AgentResult:
        s = self.settings
        run = _Run(run_id=uuid.uuid4().hex[:8])
        self.log.info("[%s] run started", run.run_id)

        with self._stage(run, "validate_input"):
            clean, redactions = sanitize_input(description, s.min_input_chars, s.max_input_chars)

        with self._stage(run, "requirements"):
            reqs = self._structured(run, "requirements", RequirementsSpec.from_dict,
                                    prompts.REQUIREMENTS_SYSTEM, prompts.requirements_user(clean))

        design: ArchitectureDesign | None = None
        review: ReviewReport | None = None
        approved = False
        rounds = 0
        while rounds < s.max_review_rounds:
            rounds += 1
            with self._stage(run, f"design (round {rounds})"):
                design = self._structured(run, "design", ArchitectureDesign.from_dict,
                                          prompts.DESIGN_SYSTEM,
                                          prompts.design_user(reqs, review, design))
            with self._stage(run, f"review (round {rounds})"):
                review = self._structured(run, "review", ReviewReport.from_dict,
                                          prompts.REVIEW_SYSTEM, prompts.review_user(reqs, design))
            has_high = any(i.severity == "high" for i in review.issues)
            approved = review.approved and review.score >= s.approval_score and not has_high
            if approved:
                break
            self.log.info("[%s] round %d not approved (score %d)", run.run_id, rounds, review.score)

        assert design is not None and review is not None
        with self._stage(run, "diagrams_and_markdown"):
            diagrams = build_diagrams(design)
            markdown = to_markdown(project_name, reqs, design, review, approved, rounds, diagrams)

        self.log.info("[%s] done approved=%s calls=%d tokens_in=%d tokens_out=%d", run.run_id,
                      approved, run.usage.llm_calls, run.usage.input_tokens, run.usage.output_tokens)
        return AgentResult(run.run_id, project_name, reqs, design, review, rounds, approved,
                           diagrams, markdown, run.trace, run.usage, redactions)

    # ---- internals -----------------------------------------------------------
    @contextmanager
    def _stage(self, run: _Run, name: str) -> Iterator[None]:
        self._emit(run, TraceEvent(name, "started"))
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self._emit(run, TraceEvent(name, "error", int((time.perf_counter() - t0) * 1000), str(exc)))
            raise
        self._emit(run, TraceEvent(name, "ok", int((time.perf_counter() - t0) * 1000)))

    def _emit(self, run: _Run, event: TraceEvent) -> None:
        run.trace.append(event)
        if self.on_event:
            self.on_event(event)

    def _structured(self, run: _Run, stage: str, parse: Callable, system: str, user: str):
        """Call the LLM, validate the answer, and re-ask with the error if it is malformed."""
        s = self.settings
        prompt, last_error = user, None
        for attempt in range(s.repair_attempts + 1):
            resp = call_with_retries(
                lambda: self.llm.complete(stage, system, prompt, s.max_tokens),
                s.llm_max_attempts, s.retry_base_delay_s, self.log)
            run.usage.add(resp.input_tokens, resp.output_tokens)
            try:
                return parse(extract_json(resp.text))
            except SchemaError as exc:
                last_error = exc
                self.log.warning("[%s] %s output invalid (attempt %d): %s", run.run_id, stage,
                                 attempt + 1, exc)
                prompt = (f"{user}\n\nYour previous answer was rejected: {exc}\n"
                          "Return the corrected JSON object only.")
        raise AgentError(f"{stage}: no valid output after {s.repair_attempts + 1} attempts: {last_error}")

"""Logging, per-stage trace events and token usage accounting."""
from __future__ import annotations

import logging
from dataclasses import dataclass


def get_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("architect_agent")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@dataclass
class TraceEvent:
    stage: str
    status: str            # started | ok | error
    duration_ms: int = 0
    detail: str = ""


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

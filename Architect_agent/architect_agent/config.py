"""Central configuration. Everything is overridable through environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _num(name: str, default, cast):
    try:
        return cast(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    model: str = "claude-sonnet-5-5"
    min_input_chars: int = 30
    max_input_chars: int = 8000
    max_review_rounds: int = 2      # design -> review cycles before giving up
    approval_score: int = 7         # reviewer score (0-10) required for approval
    llm_timeout_s: float = 90.0
    llm_max_attempts: int = 3       # attempts per LLM call on transient errors
    retry_base_delay_s: float = 1.0
    repair_attempts: int = 2        # re-asks when the model returns invalid JSON
    max_tokens: int = 4000
    use_mock: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ARCHITECT_MODEL", d.model),
            max_input_chars=_num("ARCHITECT_MAX_INPUT_CHARS", d.max_input_chars, int),
            max_review_rounds=_num("ARCHITECT_MAX_REVIEW_ROUNDS", d.max_review_rounds, int),
            approval_score=_num("ARCHITECT_APPROVAL_SCORE", d.approval_score, int),
            llm_timeout_s=_num("ARCHITECT_LLM_TIMEOUT_S", d.llm_timeout_s, float),
            llm_max_attempts=_num("ARCHITECT_LLM_MAX_ATTEMPTS", d.llm_max_attempts, int),
            use_mock=os.getenv("ARCHITECT_USE_MOCK", "").lower() in {"1", "true", "yes"},
            log_level=os.getenv("ARCHITECT_LOG_LEVEL", d.log_level),
        )

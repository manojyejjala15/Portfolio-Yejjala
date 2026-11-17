"""LLM abstraction: one tiny interface, a real Anthropic client, and retry logic."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

from .config import Settings

T = TypeVar("T")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class TransientLLMError(Exception):
    """Timeouts, rate limits, 5xx: safe to retry."""


class ConfigError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(self, stage: str, system: str, user: str, max_tokens: int = 4000) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set (or enable the mock LLM).")
        import anthropic

        self._anthropic = anthropic
        self._model = settings.model
        # SDK retries are off: call_with_retries handles them so they are logged and bounded.
        self._client = anthropic.Anthropic(
            api_key=settings.api_key, timeout=settings.llm_timeout_s, max_retries=0
        )

    def complete(self, stage: str, system: str, user: str, max_tokens: int = 4000) -> LLMResponse:
        a = self._anthropic
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except (a.APIConnectionError, a.APITimeoutError, a.RateLimitError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except a.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientLLMError(str(exc)) from exc
            raise
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens)


def call_with_retries(fn: Callable[[], T], attempts: int, base_delay: float,
                      logger: logging.Logger) -> T:
    """Exponential backoff with jitter, for transient failures only."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TransientLLMError as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.25 * base_delay)
            logger.warning("transient LLM error (%s); attempt %d/%d failed, retrying in %.1fs",
                           exc, attempt, attempts, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def build_client(settings: Settings) -> LLMClient:
    if settings.use_mock:
        from .mock_llm import MockLLM

        return MockLLM()
    return AnthropicClient(settings)

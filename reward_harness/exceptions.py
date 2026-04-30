from __future__ import annotations


class LLMCallError(RuntimeError):
    """Raised when an LLM API call fails after retries."""


class LLMFatalCallError(LLMCallError):
    """Raised for non-retriable LLM call failures (e.g., auth/model/config errors)."""


class LLMParseError(RuntimeError):
    """Raised when an LLM response cannot be parsed as expected JSON."""

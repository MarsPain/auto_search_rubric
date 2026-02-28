from __future__ import annotations


class LLMCallError(RuntimeError):
    """Raised when an LLM API call fails after retries."""


class LLMParseError(RuntimeError):
    """Raised when an LLM response cannot be parsed as expected JSON."""

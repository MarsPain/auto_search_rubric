"""Backward-compatible re-exports for historical import path.

Canonical data model definitions now live in ``autosr.data_models``.
"""

from .data_models import Criterion, GradingProtocol, PromptExample, ResponseCandidate, Rubric

__all__ = [
    "Criterion",
    "GradingProtocol",
    "Rubric",
    "ResponseCandidate",
    "PromptExample",
]

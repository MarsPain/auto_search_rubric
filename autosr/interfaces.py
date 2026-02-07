from __future__ import annotations

import random
from typing import Protocol

from .models import PromptExample, ResponseCandidate, Rubric


class Verifier(Protocol):
    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        """Return per-criterion binary grades (0/1 or None for N/A)."""


class PreferenceJudge(Protocol):
    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        """
        Return:
        - 1 if left is preferred,
        - -1 if right is preferred,
        - 0 for ties/unknown.
        """


class RubricProposer(Protocol):
    def propose(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
        rubric: Rubric,
        *,
        mode: str,
        rng: random.Random,
    ) -> Rubric:
        """Generate a new rubric variant."""


class RubricInitializer(Protocol):
    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        """Build an initial rubric for a prompt."""


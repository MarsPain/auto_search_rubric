from __future__ import annotations

import random
from typing import Protocol

from .data_models import PromptExample, ResponseCandidate, Rubric
from .types import MutationMode


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
        mode: MutationMode,
        rng: random.Random,
    ) -> Rubric:
        """Generate a new rubric variant.""
        
        Args:
            prompt: The original prompt text
            left: The preferred candidate response
            right: The runner-up candidate response
            rubric: The current rubric to mutate
            mode: The mutation strategy to apply
            rng: Random number generator for deterministic behavior
            
        Returns:
            A new mutated rubric variant
        """


class RubricInitializer(Protocol):
    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        """Build an initial rubric for a prompt."""


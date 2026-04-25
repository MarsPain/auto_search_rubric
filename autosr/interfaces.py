from __future__ import annotations

from collections.abc import Callable
import random
from typing import Any, Protocol, runtime_checkable

from .data_models import PromptExample, ResponseCandidate, Rubric
from .types import MutationMode

CheckpointCallback = Callable[[dict[str, Rubric], dict[str, float], dict[str, list[float]]], None]
StepState = dict[str, Any]


@runtime_checkable
class Searcher(Protocol):
    def search(self, prompts: list[PromptExample]) -> Any:
        """Run the full search and return the algorithm-specific search result."""


@runtime_checkable
class SteppableSearcher(Searcher, Protocol):
    @property
    def supports_step_execution(self) -> bool:
        """Whether this searcher can execute through the harness step contract."""

    @property
    def max_steps(self) -> int:
        """Maximum number of step generations for the current configuration."""

    def initialize_step_state(self, prompts: list[PromptExample]) -> StepState:
        """Create initial state for harness-managed step execution."""

    def step(self, prompts: list[PromptExample], state: StepState, generation: int) -> None:
        """Advance the search state by one generation."""

    def finalize_step_result(self, prompts: list[PromptExample], state: StepState) -> Any:
        """Build the final search result from a step state."""

    def get_algorithm_state(self, state: StepState | None) -> StepState:
        """Serialize algorithm-owned state for checkpointing."""

    def restore_algorithm_state(
        self,
        algorithm_state: StepState,
        *,
        prompt_ids: list[str],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
        history: dict[str, list[float]],
    ) -> StepState | None:
        """Restore algorithm-owned state from checkpoint payload."""


class Verifier(Protocol):
    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, float | None]:
        """Return per-criterion grades (0-5, 0-1, or None for N/A)."""


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

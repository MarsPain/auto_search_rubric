from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..config import ObjectiveConfig
from ..models import Rubric


class SelectionStrategy(Enum):
    """Parent selection strategies for evolutionary search."""

    RANK = auto()  # Original: rank-based selection
    TOURNAMENT = auto()  # Tournament selection (configurable size)
    TOP_K = auto()  # Top-K with diversity protection

    @classmethod
    def from_string(cls, s: str) -> SelectionStrategy:
        """Create from string (case-insensitive)."""
        mapping = {
            "rank": cls.RANK,
            "tournament": cls.TOURNAMENT,
            "top_k": cls.TOP_K,
        }
        return mapping.get(s.lower(), cls.RANK)


class AdaptiveMutationSchedule(Enum):
    """Schedules for adaptive mutation weight adjustment."""

    FIXED = auto()  # Original: fixed cycle through modes
    SUCCESS_FEEDBACK = auto()  # Adapt based on mutation success rate
    EXPLORATION_DECAY = auto()  # High exploration early, exploitation later
    DIVERSITY_DRIVEN = auto()  # Increase weight for modes that improve diversity

    @classmethod
    def from_string(cls, s: str) -> AdaptiveMutationSchedule:
        """Create from string (case-insensitive)."""
        mapping = {
            "fixed": cls.FIXED,
            "success_feedback": cls.SUCCESS_FEEDBACK,
            "exploration_decay": cls.EXPLORATION_DECAY,
            "diversity_driven": cls.DIVERSITY_DRIVEN,
        }
        return mapping.get(s.lower(), cls.FIXED)


@dataclass(slots=True)
class IterativeConfig:
    iterations: int = 6
    accept_only_if_improve: bool = True
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    seed: int = 7


@dataclass(slots=True)
class EvolutionaryConfig:
    # Original parameters
    population_size: int = 8
    generations: int = 20
    mutations_per_round: int = 6
    survival_fraction: float = 0.2
    batch_size: int = 4
    elitism_count: int = 2
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    stagnation_generations: int = 6
    seed: int = 7

    # === Selection Strategy Parameters ===
    selection_strategy: SelectionStrategy = field(
        default_factory=lambda: SelectionStrategy.RANK
    )
    tournament_size: int = 3  # For TOURNAMENT: number of candidates per tournament
    tournament_p: float = 0.8  # Probability of selecting best from tournament
    top_k_ratio: float = 0.3  # For TOP_K: ratio of population to consider as elite pool
    diversity_weight: float = 0.3  # Weight for diversity in selection score (0-1)

    # === Adaptive Mutation Parameters ===
    adaptive_mutation: AdaptiveMutationSchedule = field(
        default_factory=lambda: AdaptiveMutationSchedule.FIXED
    )
    mutation_window_size: int = 10  # History window for success tracking
    min_mutation_weight: float = 0.1  # Minimum weight for any mutation mode
    exploration_phase_ratio: float = 0.3  # Ratio of generations for exploration phase
    diversity_threshold: float = 0.05  # Threshold to trigger diversity-boosting mutations

    def __post_init__(self) -> None:
        # Convert string to enum if needed
        if isinstance(self.selection_strategy, str):
            self.selection_strategy = SelectionStrategy.from_string(self.selection_strategy)
        if isinstance(self.adaptive_mutation, str):
            self.adaptive_mutation = AdaptiveMutationSchedule.from_string(self.adaptive_mutation)

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.mutations_per_round < 1:
            raise ValueError("mutations_per_round must be >= 1")
        if not 0 < self.survival_fraction <= 1:
            raise ValueError("survival_fraction must be in (0, 1]")
        if self.elitism_count < 1:
            raise ValueError("elitism_count must be >= 1")

        # Validate new parameters
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        if not 0 < self.tournament_p <= 1:
            raise ValueError("tournament_p must be in (0, 1]")
        if not 0 < self.top_k_ratio <= 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("diversity_weight must be in [0, 1]")
        if self.mutation_window_size < 1:
            raise ValueError("mutation_window_size must be >= 1")
        if not 0 < self.min_mutation_weight <= 1:
            raise ValueError("min_mutation_weight must be in (0, 1]")
        if not 0 < self.exploration_phase_ratio <= 1:
            raise ValueError("exploration_phase_ratio must be in (0, 1]")
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("diversity_threshold must be in [0, 1]")


@dataclass(slots=True)
class SearchResult:
    best_rubrics: dict[str, Rubric]
    best_scores: dict[str, float]
    history: dict[str, list[float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

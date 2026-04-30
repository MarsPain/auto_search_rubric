from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import ObjectiveConfig, _validate_evolutionary_common
from ..data_models import Rubric
from ..types import AdaptiveMutationSchedule, EvolutionIterationScope, SelectionStrategy


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
    generations: int = 12
    mutations_per_round: int = 6
    mutation_parent_count: int = 3
    survival_fraction: float = 0.2
    batch_size: int = 3
    iteration_scope: EvolutionIterationScope = field(
        default_factory=lambda: EvolutionIterationScope.GLOBAL_BATCH
    )
    stop_when_distinguished: bool = True
    distinguish_margin: float | None = None
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
    diversity_threshold: float = (
        0.05  # Threshold to trigger diversity-boosting mutations
    )

    def __post_init__(self) -> None:
        # Convert string to enum if needed
        if isinstance(self.selection_strategy, str):
            try:
                self.selection_strategy = SelectionStrategy.from_string(
                    self.selection_strategy
                )
            except ValueError:
                self.selection_strategy = SelectionStrategy.RANK
        if isinstance(self.iteration_scope, str):
            try:
                self.iteration_scope = EvolutionIterationScope.from_string(
                    self.iteration_scope
                )
            except ValueError:
                self.iteration_scope = EvolutionIterationScope.GLOBAL_BATCH
        if isinstance(self.adaptive_mutation, str):
            try:
                self.adaptive_mutation = AdaptiveMutationSchedule.from_string(
                    self.adaptive_mutation
                )
            except ValueError:
                self.adaptive_mutation = AdaptiveMutationSchedule.FIXED

        if not isinstance(self.iteration_scope, EvolutionIterationScope):
            raise ValueError(
                "iteration_scope must be an EvolutionIterationScope or its string value"
            )

        _validate_evolutionary_common(
            population_size=self.population_size,
            generations=self.generations,
            mutations_per_round=self.mutations_per_round,
            mutation_parent_count=self.mutation_parent_count,
            survival_fraction=self.survival_fraction,
            elitism_count=self.elitism_count,
            tournament_size=self.tournament_size,
            tournament_p=self.tournament_p,
            top_k_ratio=self.top_k_ratio,
            diversity_weight=self.diversity_weight,
            mutation_window_size=self.mutation_window_size,
            min_mutation_weight=self.min_mutation_weight,
            exploration_phase_ratio=self.exploration_phase_ratio,
            diversity_threshold=self.diversity_threshold,
            distinguish_margin=self.distinguish_margin,
        )


@dataclass(slots=True)
class SearchResult:
    best_rubrics: dict[str, Rubric]
    best_scores: dict[str, float]
    history: dict[str, list[float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

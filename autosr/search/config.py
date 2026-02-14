from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import ObjectiveConfig
from ..models import Rubric


@dataclass(slots=True)
class IterativeConfig:
    iterations: int = 6
    accept_only_if_improve: bool = True
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    seed: int = 7


@dataclass(slots=True)
class EvolutionaryConfig:
    population_size: int = 8
    generations: int = 20
    mutations_per_round: int = 6
    survival_fraction: float = 0.2
    batch_size: int = 4
    elitism_count: int = 2
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    stagnation_generations: int = 6
    seed: int = 7

    def __post_init__(self) -> None:
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


@dataclass(slots=True)
class SearchResult:
    best_rubrics: dict[str, Rubric]
    best_scores: dict[str, float]
    history: dict[str, list[float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

"""Adaptive mutation strategies for evolutionary search.

This module provides dynamic mutation mode selection based on:
- Success feedback from previous mutations
- Exploration/exploitation phase of search
- Population diversity metrics
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..models import Rubric
from ..types import AdaptiveMutationSchedule, MutationMode

if TYPE_CHECKING:
    from .config import EvolutionaryConfig


class MutationScheduler(Protocol):
    """Protocol for mutation mode scheduling strategies."""

    def select_mode(self, diversity_score: float | None = None) -> MutationMode:
        """Select the mutation mode for the next proposal."""

    def record_outcome(
        self,
        mode: MutationMode,
        was_successful: bool,
        score_improvement: float,
    ) -> None:
        """Record the outcome of a mutation."""

    def next_generation(self) -> None:
        """Advance internal generation state."""

    def get_diagnostics(self) -> dict[str, object]:
        """Return scheduler diagnostics for run reports."""


class DiversityMetric(Protocol):
    """Protocol for population diversity metrics."""

    def compute(
        self,
        rubrics: list[Rubric],
        *,
        sample_size: int = 10,
        rng: random.Random | None = None,
    ) -> float:
        """Compute diversity score in [0, 1]."""


@dataclass
class MutationHistory:
    """Tracks history of mutations for adaptive weight adjustment."""

    # mode -> list of (was_successful, score_improvement)
    mode_history: dict[MutationMode, deque[tuple[bool, float]]] = field(
        default_factory=lambda: {
            mode: deque(maxlen=100) for mode in MutationMode
        }
    )
    generation_count: int = 0

    def record(self, mode: MutationMode, was_successful: bool, improvement: float) -> None:
        """Record a mutation outcome."""
        self.mode_history[mode].append((was_successful, improvement))

    def get_success_rate(self, mode: MutationMode, window_size: int) -> float:
        """Get recent success rate for a mutation mode."""
        history = self.mode_history[mode]
        if not history:
            return 0.5  # Neutral default

        recent = list(history)[-window_size:]
        successes = sum(1 for success, _ in recent if success)
        return successes / len(recent)

    def get_average_improvement(self, mode: MutationMode, window_size: int) -> float:
        """Get average improvement for a mutation mode."""
        history = self.mode_history[mode]
        if not history:
            return 0.0

        recent = list(history)[-window_size:]
        improvements = [imp for _, imp in recent]
        return sum(improvements) / len(improvements)

    def increment_generation(self) -> None:
        """Increment generation counter."""
        self.generation_count += 1


class AdaptiveMutationSelector:
    """Selects mutation modes adaptively based on schedule and history."""

    def __init__(
        self,
        config: EvolutionaryConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng
        self.history = MutationHistory()
        self._weights: dict[MutationMode, float] = {
            mode: 1.0 for mode in MutationMode
        }

    def select_mode(self, diversity_score: float | None = None) -> MutationMode:
        """Select a mutation mode based on current strategy.

        Args:
            diversity_score: Optional current population diversity (0-1)

        Returns:
            Selected mutation mode
        """
        schedule = self.config.adaptive_mutation

        if schedule is AdaptiveMutationSchedule.FIXED:
            return self._select_fixed()
        if schedule is AdaptiveMutationSchedule.SUCCESS_FEEDBACK:
            return self._select_success_feedback()
        if schedule is AdaptiveMutationSchedule.EXPLORATION_DECAY:
            return self._select_exploration_decay()
        if schedule is AdaptiveMutationSchedule.DIVERSITY_DRIVEN:
            return self._select_diversity_driven(diversity_score or 0.5)

        # Fallback to fixed
        return self._select_fixed()

    def _select_fixed(self) -> MutationMode:
        """Original fixed cycle selection."""
        idx = self.history.generation_count % len(MutationMode)
        return list(MutationMode)[idx]

    def _select_success_feedback(self) -> MutationMode:
        """Select based on historical success rates.

        Modes with higher success rates get higher selection probability.
        """
        window = self.config.mutation_window_size

        # Compute weighted scores
        scores: dict[MutationMode, float] = {}
        for mode in MutationMode:
            success_rate = self.history.get_success_rate(mode, window)
            avg_improvement = self.history.get_average_improvement(mode, window)
            # Combine success rate with average improvement magnitude
            scores[mode] = success_rate * (1 + abs(avg_improvement))

        # Normalize to probabilities
        total = sum(scores.values())
        if total == 0:
            probs = {mode: 1.0 / len(MutationMode) for mode in MutationMode}
        else:
            probs = {mode: score / total for mode, score in scores.items()}

        # Ensure minimum weight
        min_weight = self.config.min_mutation_weight
        for mode in probs:
            probs[mode] = max(probs[mode], min_weight)

        # Renormalize
        total = sum(probs.values())
        probs = {mode: p / total for mode, p in probs.items()}

        # Select based on probabilities
        r = self.rng.random()
        cumulative = 0.0
        for mode, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return mode

        return list(MutationMode)[-1]  # Fallback

    def _select_exploration_decay(self) -> MutationMode:
        """Select based on exploration phase.

        Early generations: favor exploratory mutations (more diverse)
        Later generations: favor exploitative refinements (fine-tuning)
        """
        gen = self.history.generation_count
        max_gen = self.config.generations
        exploration_ratio = self.config.exploration_phase_ratio

        # Determine if in exploration phase
        exploration_progress = gen / (max_gen * exploration_ratio)
        is_exploration = exploration_progress < 1.0

        # Mode categories based on actual MutationMode enum
        # Exploratory: modes that significantly change rubric structure
        exploratory_modes = [
            MutationMode.DECOMPOSE,
            MutationMode.COUNTEREXAMPLE_TRIGGER,
            MutationMode.WEIGHT_PERTURB,
        ]
        # Exploitative: modes that refine existing rubrics
        exploitative_modes = [
            MutationMode.RAISE_BAR,
            MutationMode.ANTI_FLUFF,
            MutationMode.FACTUAL_FOCUS,
        ]

        if is_exploration:
            # Higher probability for exploratory modes
            weights = {
                mode: 2.0 if mode in exploratory_modes else
                      0.5 if mode in exploitative_modes else 1.0
                for mode in MutationMode
            }
        else:
            # Higher probability for exploitative modes
            weights = {
                mode: 2.0 if mode in exploitative_modes else
                      0.5 if mode in exploratory_modes else 1.0
                for mode in MutationMode
            }

        # Normalize and select
        total = sum(weights.values())
        probs = {mode: w / total for mode, w in weights.items()}

        r = self.rng.random()
        cumulative = 0.0
        for mode, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return mode

        return list(MutationMode)[-1]

    def _select_diversity_driven(self, diversity_score: float) -> MutationMode:
        """Select based on population diversity.

        When diversity is low, favor modes that increase diversity.
        When diversity is high, favor quality-improving modes.
        """
        # Diversity categories based on actual MutationMode enum
        # Diversity-boosting: modes that introduce significant changes
        diversity_boosting = [
            MutationMode.DECOMPOSE,
            MutationMode.COUNTEREXAMPLE_TRIGGER,
            MutationMode.WEIGHT_PERTURB,
        ]
        # Quality-focused: modes that refine and improve existing structure
        quality_focused = [
            MutationMode.RAISE_BAR,
            MutationMode.ANTI_FLUFF,
            MutationMode.FACTUAL_FOCUS,
        ]

        # Compute diversity need (0-1)
        diversity_need = max(0.0, 1.0 - diversity_score / self.config.diversity_threshold)

        # Adjust weights based on diversity need
        weights = {}
        for mode in MutationMode:
            if mode in diversity_boosting:
                weights[mode] = 1.0 + diversity_need  # Higher when diversity is low
            elif mode in quality_focused:
                weights[mode] = 1.0 + (1 - diversity_need)  # Higher when diversity is high
            else:
                weights[mode] = 1.0

        # Normalize and select
        total = sum(weights.values())
        probs = {mode: w / total for mode, w in weights.items()}

        r = self.rng.random()
        cumulative = 0.0
        for mode, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return mode

        return list(MutationMode)[-1]

    def record_outcome(
        self,
        mode: MutationMode,
        was_successful: bool,
        score_improvement: float,
    ) -> None:
        """Record the outcome of a mutation for future adaptation."""
        self.history.record(mode, was_successful, score_improvement)

    def next_generation(self) -> None:
        """Move to next generation."""
        self.history.increment_generation()

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about current state."""
        window = self.config.mutation_window_size
        return {
            "generation": self.history.generation_count,
            "success_rates": {
                mode.name: self.history.get_success_rate(mode, window)
                for mode in MutationMode
            },
            "avg_improvements": {
                mode.name: self.history.get_average_improvement(mode, window)
                for mode in MutationMode
            },
        }


class FingerprintDiversityMetric:
    """Default diversity metric based on rubric fingerprint distance."""

    def compute(
        self,
        rubrics: list[Rubric],
        *,
        sample_size: int = 10,
        rng: random.Random | None = None,
    ) -> float:
        if len(rubrics) < 2:
            return 0.0

        rng = rng or random.Random()

        # Get fingerprints
        fingerprints = [r.fingerprint() for r in rubrics]

        # Sample pairs
        n = len(fingerprints)
        max_pairs = n * (n - 1) // 2
        num_samples = min(sample_size, max_pairs)

        distances = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(distances) < num_samples and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(range(n), 2)

            fp1, fp2 = fingerprints[i], fingerprints[j]
            max_len = max(len(fp1), len(fp2))

            if max_len == 0:
                continue

            # Simple character-level distance
            matches = sum(
                1 for a, b in zip(fp1, fp2) if a == b
            )
            similarity = matches / max_len
            distances.append(1.0 - similarity)

        return sum(distances) / len(distances) if distances else 0.0


def create_mutation_scheduler(
    config: EvolutionaryConfig,
    rng: random.Random,
) -> MutationScheduler:
    """Create the default mutation scheduler from config."""
    return AdaptiveMutationSelector(config, rng)


def create_diversity_metric() -> DiversityMetric:
    """Create the default diversity metric implementation."""
    return FingerprintDiversityMetric()


def compute_population_diversity(
    rubrics: list[Rubric],
    sample_size: int = 10,
    rng: random.Random | None = None,
) -> float:
    """Compute population diversity score.

    Uses average pairwise distance between rubric fingerprints.

    Args:
        rubrics: List of rubrics to evaluate
        sample_size: Number of pairs to sample for efficiency
        rng: Random number generator for sampling

    Returns:
        Diversity score in [0, 1], higher is more diverse
    """
    metric = FingerprintDiversityMetric()
    return metric.compute(rubrics, sample_size=sample_size, rng=rng)

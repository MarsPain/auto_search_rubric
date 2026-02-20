"""Advanced parent selection strategies for evolutionary search.

This module provides selection strategies that balance exploitation (quality)
with exploration (diversity) to improve search performance and convergence stability.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ..evaluator import ObjectiveBreakdown
from ..models import Rubric

if TYPE_CHECKING:
    from .config import EvolutionaryConfig


def select_parents_rank(
    scored_population: list[tuple[Rubric, ObjectiveBreakdown]],
    num_parents: int,
    rng: random.Random,
    config: EvolutionaryConfig,  # noqa: ARG001
) -> list[Rubric]:
    """Original rank-based selection: select best N individuals.

    Args:
        scored_population: List of (rubric, score) tuples, sorted by score descending
        num_parents: Number of parents to select
        rng: Random number generator (unused, for interface consistency)
        config: Evolutionary configuration (unused, for interface consistency)

    Returns:
        List of selected parent rubrics
    """
    return [rubric for rubric, _ in scored_population[:num_parents]]


def select_parents_tournament(
    scored_population: list[tuple[Rubric, ObjectiveBreakdown]],
    num_parents: int,
    rng: random.Random,
    config: EvolutionaryConfig,
) -> list[Rubric]:
    """Tournament selection with configurable selection pressure.

    Tournament selection provides a balance between quality and diversity:
    - Small tournament size -> more diversity, less selection pressure
    - Large tournament size -> less diversity, more selection pressure
    - tournament_p controls probability of selecting best vs random from tournament

    Args:
        scored_population: List of (rubric, score) tuples, sorted by score descending
        num_parents: Number of parents to select
        rng: Random number generator
        config: Configuration with tournament_size and tournament_p

    Returns:
        List of selected parent rubrics
    """
    population = [rubric for rubric, _ in scored_population]
    scores = {id(rubric): score.total for rubric, score in scored_population}
    selected: list[Rubric] = []
    selected_ids: set[int] = set()

    tournament_size = min(config.tournament_size, len(population))

    while len(selected) < num_parents:
        # Run tournament
        tournament_candidates = rng.sample(population, tournament_size)
        tournament_candidates.sort(key=lambda r: scores[id(r)], reverse=True)

        # Select with probability tournament_p for best, otherwise random
        if rng.random() < config.tournament_p:
            winner = tournament_candidates[0]
        else:
            winner = rng.choice(tournament_candidates)

        winner_id = id(winner)
        if winner_id not in selected_ids:
            selected.append(winner)
            selected_ids.add(winner_id)

    return selected


def _compute_diversity_score(
    rubric: Rubric,
    selected: list[Rubric],
) -> float:
    """Compute diversity score as average distance from already selected individuals.

    Uses a simple content-based distance metric based on rubric fingerprint.
    Higher score means more different from existing selections.

    Args:
        rubric: The rubric to evaluate
        selected: Already selected rubrics

    Returns:
        Diversity score in [0, 1], higher is more diverse
    """
    if not selected:
        return 1.0  # Maximum diversity if nothing selected yet

    rubric_fp = rubric.fingerprint()
    distances = []

    for other in selected:
        other_fp = other.fingerprint()
        # Jaccard-like distance based on common substrings
        # Simple implementation: character-level similarity
        max_len = max(len(rubric_fp), len(other_fp))
        if max_len == 0:
            distances.append(0.0)
            continue

        # Count common characters at same positions
        matches = sum(
            1 for a, b in zip(rubric_fp, other_fp) if a == b
        )
        similarity = matches / max_len
        distances.append(1.0 - similarity)

    return sum(distances) / len(distances)


def select_parents_top_k_diverse(
    scored_population: list[tuple[Rubric, ObjectiveBreakdown]],
    num_parents: int,
    rng: random.Random,  # noqa: ARG001
    config: EvolutionaryConfig,
) -> list[Rubric]:
    """Top-K selection with diversity protection.

    First selects from top-K pool based on combined quality + diversity score.
    This ensures we don't lose good solutions while maintaining population diversity.

    Args:
        scored_population: List of (rubric, score) tuples, sorted by score descending
        num_parents: Number of parents to select
        rng: Random number generator (unused, deterministic selection)
        config: Configuration with top_k_ratio and diversity_weight

    Returns:
        List of selected parent rubrics
    """
    if not scored_population:
        return []

    # Define top-K pool
    k = max(1, int(len(scored_population) * config.top_k_ratio))
    top_k_pool = scored_population[:k]

    # Normalize scores to [0, 1]
    max_score = top_k_pool[0][1].total
    min_score = top_k_pool[-1][1].total
    score_range = max_score - min_score if max_score > min_score else 1.0

    selected: list[Rubric] = []
    remaining = list(top_k_pool)

    while len(selected) < num_parents and remaining:
        # Compute combined score for each candidate
        best_idx = 0
        best_combined = -math.inf

        for idx, (rubric, breakdown) in enumerate(remaining):
            # Quality component (normalized)
            quality = (breakdown.total - min_score) / score_range

            # Diversity component
            diversity = _compute_diversity_score(rubric, selected)

            # Combined score
            combined = (
                (1 - config.diversity_weight) * quality +
                config.diversity_weight * diversity
            )

            if combined > best_combined:
                best_combined = combined
                best_idx = idx

        selected.append(remaining.pop(best_idx)[0])

    # If we need more parents, fill with remaining population
    remaining_population = scored_population[k:]
    for rubric, _ in remaining_population:
        if len(selected) >= num_parents:
            break
        if rubric not in selected:
            selected.append(rubric)

    return selected[:num_parents]


# Registry for selection strategies
SELECTION_STRATEGIES = {
    "rank": select_parents_rank,
    "tournament": select_parents_tournament,
    "top_k": select_parents_top_k_diverse,
}


def select_parents(
    strategy: str,
    scored_population: list[tuple[Rubric, ObjectiveBreakdown]],
    num_parents: int,
    rng: random.Random,
    config: EvolutionaryConfig,
) -> list[Rubric]:
    """Select parents using the specified strategy.

    Args:
        strategy: Selection strategy name (rank, tournament, top_k)
        scored_population: List of (rubric, score) tuples
        num_parents: Number of parents to select
        rng: Random number generator
        config: Evolutionary configuration

    Returns:
        List of selected parent rubrics

    Raises:
        ValueError: If unknown strategy specified
    """
    if strategy not in SELECTION_STRATEGIES:
        msg = f"Unknown selection strategy: {strategy}"
        raise ValueError(msg)

    return SELECTION_STRATEGIES[strategy](
        scored_population, num_parents, rng, config
    )

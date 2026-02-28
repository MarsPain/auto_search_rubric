from __future__ import annotations

import logging
import math
import random

from ..evaluator import ObjectiveBreakdown, RubricEvaluator, disagreement_score
from ..interfaces import PreferenceJudge, RubricInitializer, RubricProposer, Verifier
from ..models import PromptExample, Rubric
from ..types import MutationMode
from .adaptive_mutation import (
    DiversityMetric,
    MutationScheduler,
    create_diversity_metric,
    create_mutation_scheduler,
)
from .config import EvolutionaryConfig, SearchResult
from .selection_strategies import select_parents
from .strategies import (
    MUTATION_MODES,
    _append_unique,
    _build_top2_pair,
    _evaluate_rubric,
    _fingerprint,
)

logger = logging.getLogger("autosr.search")


class EvolutionaryRTDSearcher:
    def __init__(
        self,
        proposer: RubricProposer,
        verifier: Verifier,
        judge: PreferenceJudge,
        initializer: RubricInitializer,
        config: EvolutionaryConfig | None = None,
        mutation_scheduler: MutationScheduler | None = None,
        diversity_metric: DiversityMetric | None = None,
    ) -> None:
        self.proposer = proposer
        self.judge = judge
        self.initializer = initializer
        self.config = config or EvolutionaryConfig()
        self.rng = random.Random(self.config.seed)
        self.evaluator = RubricEvaluator(verifier, base_seed=self.config.seed)

        # Use injected scheduler/metric when provided, otherwise create defaults from config.
        self.mutation_scheduler = mutation_scheduler or create_mutation_scheduler(
            self.config,
            self.rng,
        )
        self.diversity_metric = diversity_metric or create_diversity_metric()

        # Track diversity history for diagnostics
        self.diversity_history: list[float] = []

    def search(self, prompts: list[PromptExample]) -> SearchResult:
        logger.info(
            "starting evolutionary search prompts=%d generations=%d population=%d "
            "selection=%s mutation=%s",
            len(prompts),
            self.config.generations,
            self.config.population_size,
            self.config.selection_strategy.name,
            self.config.adaptive_mutation.name,
        )
        population, history, best_rubrics, best_scores = self._init_global_state(prompts)
        stale_rounds = 0

        for generation in range(self.config.generations):
            scored_population = self._score_generation(prompts, population)
            self._log_generation_progress(generation, scored_population)

            # Compute and track diversity
            diversity_scores: dict[str, float] = {}
            for item in prompts:
                diversity = self.diversity_metric.compute(
                    population[item.prompt_id],
                    rng=self.rng,
                )
                diversity_scores[item.prompt_id] = diversity
                self.diversity_history.append(diversity)

            generation_improved = self._update_generation_bests(
                scored_population=scored_population,
                best_rubrics=best_rubrics,
                best_scores=best_scores,
                history=history,
            )
            stale_rounds, should_stop = self._handle_stagnation(generation_improved, stale_rounds)
            if should_stop:
                logger.info(
                    "stopping early at generation=%d stale_rounds=%d threshold=%d",
                    generation + 1,
                    stale_rounds,
                    self.config.stagnation_generations,
                )
                break

            hard_prompt_ids = self._select_hard_prompts(prompts, population, scored_population)
            logger.info(
                "generation=%d selected_hard_prompts=%s",
                generation + 1,
                ",".join(sorted(hard_prompt_ids)) if hard_prompt_ids else "<none>",
            )
            self._evolve_selected_prompts(
                prompts=prompts,
                hard_prompt_ids=hard_prompt_ids,
                scored_population=scored_population,
                population=population,
                diversity_scores=diversity_scores,
                generation=generation,
            )

            # Update mutation selector for next generation
            self.mutation_scheduler.next_generation()

        self._finalize_best_from_population(
            prompts=prompts,
            population=population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
        )

        diagnostics = {
            "mode": "evolutionary",
            "selection_strategy": self.config.selection_strategy.name,
            "adaptive_mutation": self.config.adaptive_mutation.name,
            "mutation_diagnostics": self.mutation_scheduler.get_diagnostics(),
            "avg_diversity": sum(self.diversity_history) / len(self.diversity_history)
            if self.diversity_history else 0.0,
        }

        return SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics=diagnostics,
        )

    def _init_global_state(
        self,
        prompts: list[PromptExample],
    ) -> tuple[dict[str, list[Rubric]], dict[str, list[float]], dict[str, Rubric], dict[str, float]]:
        population: dict[str, list[Rubric]] = {}
        for item in prompts:
            logger.info(
                "initializing population prompt=%s target_size=%d",
                item.prompt_id,
                self.config.population_size,
            )
            population[item.prompt_id] = self._init_population(item)
        history = {item.prompt_id: [] for item in prompts}
        best_rubrics: dict[str, Rubric] = {}
        best_scores = {item.prompt_id: -math.inf for item in prompts}
        return population, history, best_rubrics, best_scores

    def _log_generation_progress(
        self,
        generation: int,
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
    ) -> None:
        per_prompt = []
        for prompt_id, scored in sorted(scored_population.items()):
            best_score = scored[0][1].total
            per_prompt.append(f"{prompt_id}:best_score={best_score:.4f}")
        logger.info(
            "generation=%d/%d %s",
            generation + 1,
            self.config.generations,
            "; ".join(per_prompt),
        )

    def _score_generation(
        self,
        prompts: list[PromptExample],
        population: dict[str, list[Rubric]],
    ) -> dict[str, list[tuple[Rubric, ObjectiveBreakdown]]]:
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]] = {}
        for item in prompts:
            scored_population[item.prompt_id] = self._score_population(item, population[item.prompt_id])
        return scored_population

    def _update_generation_bests(
        self,
        *,
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
        history: dict[str, list[float]],
    ) -> bool:
        generation_improved = False
        for prompt_id, scored in scored_population.items():
            best_rubric, best_breakdown = scored[0]
            history[prompt_id].append(best_breakdown.total)
            if best_breakdown.total > best_scores[prompt_id]:
                best_scores[prompt_id] = best_breakdown.total
                best_rubrics[prompt_id] = best_rubric
                generation_improved = True
        return generation_improved

    def _handle_stagnation(self, generation_improved: bool, stale_rounds: int) -> tuple[int, bool]:
        if generation_improved:
            stale_rounds = 0
        else:
            stale_rounds += 1
        return stale_rounds, stale_rounds >= self.config.stagnation_generations

    def _evolve_selected_prompts(
        self,
        *,
        prompts: list[PromptExample],
        hard_prompt_ids: set[str],
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
        population: dict[str, list[Rubric]],
        diversity_scores: dict[str, float],
        generation: int,
    ) -> None:
        for item in prompts:
            if item.prompt_id not in hard_prompt_ids:
                continue
            scored = scored_population[item.prompt_id]
            population[item.prompt_id] = self._evolve_one_prompt(
                item=item,
                scored=scored,
                population_for_prompt=population[item.prompt_id],
                generation=generation,
                diversity_score=diversity_scores[item.prompt_id],
            )

    def _evolve_one_prompt(
        self,
        *,
        item: PromptExample,
        scored: list[tuple[Rubric, ObjectiveBreakdown]],
        population_for_prompt: list[Rubric],
        generation: int,  # noqa: ARG002
        diversity_score: float,
    ) -> list[Rubric]:
        del population_for_prompt  # Reserved for future prompt-local stateful strategies.
        best_current = scored[0][0]
        best_current_score = scored[0][1].total
        left, right = _build_top2_pair(item=item, rubric=best_current, evaluator=self.evaluator)
        new_candidates: list[Rubric] = []
        mutation_modes_used: list[MutationMode] = []

        for _idx in range(self.config.mutations_per_round):
            # Use adaptive mutation selection
            mode = self.mutation_scheduler.select_mode(diversity_score=diversity_score)
            mutation_modes_used.append(mode)
            mutated = self.proposer.propose(
                item.prompt, left, right, best_current, mode=mode, rng=self.rng
            )
            new_candidates.append(mutated)

        logger.debug(
            "prompt=%s mutation_modes=%s",
            item.prompt_id,
            [m.name for m in mutation_modes_used],
        )

        winners = self._successive_halving(item, new_candidates)

        # Record mutation outcomes for adaptive learning
        winner_scores = self._score_population(item, winners)
        best_winner_score = winner_scores[0][1].total if winner_scores else best_current_score
        improvement = best_winner_score - best_current_score

        for mode in mutation_modes_used:
            self.mutation_scheduler.record_outcome(
                mode=mode,
                was_successful=improvement > 0,
                score_improvement=improvement,
            )

        return self._update_population(scored, winners, self.config.population_size)

    def _finalize_best_from_population(
        self,
        *,
        prompts: list[PromptExample],
        population: dict[str, list[Rubric]],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
    ) -> None:
        """Finalize best rubrics by evaluating the final population.

        This ensures that the best rubrics are selected from the final population,
        not from intermediate generations. This fixes a bug where the population
        evolved in the last generation but the new rubrics were never evaluated.
        """
        for item in prompts:
            scored = self._score_population(item, population[item.prompt_id])
            if scored[0][1].total > best_scores.get(item.prompt_id, -math.inf):
                best_rubrics[item.prompt_id] = scored[0][0]
                best_scores[item.prompt_id] = scored[0][1].total

    def _init_population(self, item: PromptExample) -> list[Rubric]:
        base = self.initializer.initialize(item, rng=self.rng)
        population: list[Rubric] = [base]
        seen = {_fingerprint(base)}
        attempts = 0
        max_attempts = self.config.population_size * 12
        while len(population) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            left, right = item.candidates[0], item.candidates[1]
            mode = MUTATION_MODES[len(population) % len(MUTATION_MODES)]
            mutated = self.proposer.propose(item.prompt, left, right, base, mode=mode, rng=self.rng)
            _append_unique(population, mutated, seen)
        while len(population) < self.config.population_size:
            population.append(base)
        return population

    def _score_population(
        self, item: PromptExample, rubrics: list[Rubric]
    ) -> list[tuple[Rubric, ObjectiveBreakdown]]:
        scored = [
            (
                rubric,
                _evaluate_rubric(
                    item=item,
                    rubric=rubric,
                    evaluator=self.evaluator,
                    judge=self.judge,
                    objective=self.config.objective,
                ),
            )
            for rubric in rubrics
        ]
        scored.sort(key=lambda pair: pair[1].total, reverse=True)
        return scored

    def _successive_halving(self, item: PromptExample, candidates: list[Rubric]) -> list[Rubric]:
        if not candidates:
            return []
        survivors = candidates
        stages = [
            self.config.objective.pair_budget_small,
            self.config.objective.pair_budget_medium,
            self.config.objective.pair_budget_full,
        ]
        for stage_budget in stages:
            scored = [
                (
                    rubric,
                    _evaluate_rubric(
                        item=item,
                        rubric=rubric,
                        evaluator=self.evaluator,
                        judge=self.judge,
                        objective=self.config.objective,
                        pair_budget=stage_budget if stage_budget > 0 else None,
                    ),
                )
                for rubric in survivors
            ]
            scored.sort(key=lambda pair: pair[1].total, reverse=True)
            keep_count = max(1, int(math.ceil(len(scored) * self.config.survival_fraction)))
            survivors = [rubric for rubric, _ in scored[:keep_count]]
            if len(survivors) <= 1:
                break
        return survivors

    def _update_population(
        self,
        scored_existing: list[tuple[Rubric, ObjectiveBreakdown]],
        winners: list[Rubric],
        target_size: int,
    ) -> list[Rubric]:
        """Update population using configured selection strategy.

        Uses the selection_strategy from config to choose parents for next generation.
        """
        new_population: list[Rubric] = []
        seen: set[str] = set()

        # Always keep elite individuals
        for rubric, _ in scored_existing[: self.config.elitism_count]:
            _append_unique(new_population, rubric, seen)

        # Select parents using configured strategy
        num_parents_needed = target_size - len(new_population)
        if num_parents_needed > 0 and scored_existing:
            selected = select_parents(
                strategy=self.config.selection_strategy,
                scored_population=scored_existing,
                num_parents=num_parents_needed,
                rng=self.rng,
                config=self.config,
            )
            for rubric in selected:
                if len(new_population) >= target_size:
                    break
                _append_unique(new_population, rubric, seen)

        # Fill any remaining slots with winners
        for rubric in winners:
            if len(new_population) >= target_size:
                break
            _append_unique(new_population, rubric, seen)

        # Final fallback: fill with existing population
        for rubric, _ in scored_existing:
            if len(new_population) >= target_size:
                break
            _append_unique(new_population, rubric, seen)

        return new_population[:target_size]

    def _select_hard_prompts(
        self,
        prompts: list[PromptExample],
        populations: dict[str, list[Rubric]],
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
    ) -> set[str]:
        hardness: list[tuple[str, float]] = []
        prompt_map = {item.prompt_id: item for item in prompts}
        for prompt_id, scored in scored_population.items():
            best_breakdown = scored[0][1]
            margin_term = 1.0 / (1.0 + best_breakdown.top_margin)
            disagreement = disagreement_score(
                prompt_map[prompt_id], populations[prompt_id], self.evaluator
            )
            hardness_value = margin_term + (2.0 * disagreement)
            hardness.append((prompt_id, hardness_value))
        hardness.sort(key=lambda pair: pair[1], reverse=True)
        chosen = hardness[: self.config.batch_size]
        return {prompt_id for prompt_id, _ in chosen}

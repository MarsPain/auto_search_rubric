from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import random
from typing import Any

from .evaluator import (
    ObjectiveBreakdown,
    ObjectiveConfig,
    RubricEvaluator,
    compute_objective,
    disagreement_score,
    top2_under_rubric,
)
from .interfaces import PreferenceJudge, RubricInitializer, RubricProposer, Verifier
from .models import PromptExample, ResponseCandidate, Rubric

logger = logging.getLogger(__name__)


MUTATION_MODES = (
    "raise_bar",
    "decompose",
    "factual_focus",
    "anti_fluff",
    "counterexample_trigger",
    "weight_perturb",
)


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


class IterativeRTDSearcher:
    def __init__(
        self,
        proposer: RubricProposer,
        verifier: Verifier,
        judge: PreferenceJudge,
        initializer: RubricInitializer,
        config: IterativeConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.judge = judge
        self.initializer = initializer
        self.config = config or IterativeConfig()
        self.rng = random.Random(self.config.seed)
        self.evaluator = RubricEvaluator(verifier, base_seed=self.config.seed)

    def search(self, prompts: list[PromptExample]) -> SearchResult:
        best_rubrics: dict[str, Rubric] = {}
        best_scores: dict[str, float] = {}
        history: dict[str, list[float]] = {}
        logger.info(
            "starting iterative search prompts=%d iterations=%d",
            len(prompts),
            self.config.iterations,
        )
        for item in prompts:
            logger.info("iterative prompt=%s init", item.prompt_id)
            best_rubric, best_score, local_history = self._search_one_prompt_iterative(item)
            best_rubrics[item.prompt_id] = best_rubric
            best_scores[item.prompt_id] = best_score
            history[item.prompt_id] = local_history
            logger.info(
                "iterative prompt=%s complete best_score=%.4f",
                item.prompt_id,
                best_score,
            )

        return SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics={"mode": "iterative"},
        )

    def _search_one_prompt_iterative(
        self, item: PromptExample
    ) -> tuple[Rubric, float, list[float]]:
        rubric = self.initializer.initialize(item, rng=self.rng)
        best_rubric = rubric
        best_score = -math.inf
        local_history: list[float] = []

        for step in range(self.config.iterations):
            breakdown = self._evaluate_rubric(item, rubric)
            local_history.append(breakdown.total)
            logger.info(
                "iterative prompt=%s step=%d/%d current_score=%.4f",
                item.prompt_id,
                step + 1,
                self.config.iterations,
                breakdown.total,
            )
            best_rubric, best_score = self._track_best(
                best_rubric, best_score, rubric, breakdown.total
            )

            left, right = self._build_top2_pair(item, rubric)
            candidate_rubric = self._propose_mutation(item, left, right, rubric, step)
            if not self.config.accept_only_if_improve:
                rubric = candidate_rubric
                continue

            candidate_breakdown = self._evaluate_rubric(item, candidate_rubric)
            best_rubric, best_score = self._track_best(
                best_rubric, best_score, candidate_rubric, candidate_breakdown.total
            )
            if self._should_accept_candidate(
                current_score=breakdown.total,
                candidate_score=candidate_breakdown.total,
            ):
                rubric = candidate_rubric

        final_breakdown = self._evaluate_rubric(item, rubric)
        best_rubric, best_score = self._track_best(
            best_rubric, best_score, rubric, final_breakdown.total
        )
        return best_rubric, best_score, local_history

    def _evaluate_rubric(
        self,
        item: PromptExample,
        rubric: Rubric,
        *,
        pair_budget: int | None = None,
    ) -> ObjectiveBreakdown:
        return _evaluate_rubric(
            item=item,
            rubric=rubric,
            evaluator=self.evaluator,
            judge=self.judge,
            objective=self.config.objective,
            pair_budget=pair_budget,
        )

    def _build_top2_pair(
        self,
        item: PromptExample,
        rubric: Rubric,
    ) -> tuple[ResponseCandidate, ResponseCandidate]:
        return _build_top2_pair(item=item, rubric=rubric, evaluator=self.evaluator)

    def _propose_mutation(
        self,
        item: PromptExample,
        left: ResponseCandidate,
        right: ResponseCandidate,
        rubric: Rubric,
        step: int,
    ) -> Rubric:
        mode = MUTATION_MODES[step % len(MUTATION_MODES)]
        return self.proposer.propose(
            item.prompt,
            left,
            right,
            rubric,
            mode=mode,
            rng=self.rng,
        )

    def _track_best(
        self,
        best_rubric: Rubric,
        best_score: float,
        candidate_rubric: Rubric,
        candidate_score: float,
    ) -> tuple[Rubric, float]:
        if candidate_score > best_score:
            return candidate_rubric, candidate_score
        return best_rubric, best_score

    def _should_accept_candidate(self, *, current_score: float, candidate_score: float) -> bool:
        return candidate_score >= current_score


class EvolutionaryRTDSearcher:
    def __init__(
        self,
        proposer: RubricProposer,
        verifier: Verifier,
        judge: PreferenceJudge,
        initializer: RubricInitializer,
        config: EvolutionaryConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.judge = judge
        self.initializer = initializer
        self.config = config or EvolutionaryConfig()
        self.rng = random.Random(self.config.seed)
        self.evaluator = RubricEvaluator(verifier, base_seed=self.config.seed)

    def search(self, prompts: list[PromptExample]) -> SearchResult:
        logger.info(
            "starting evolutionary search prompts=%d generations=%d population=%d",
            len(prompts),
            self.config.generations,
            self.config.population_size,
        )
        population, history, best_rubrics, best_scores = self._init_global_state(prompts)
        stale_rounds = 0

        for generation in range(self.config.generations):
            scored_population = self._score_generation(prompts, population)
            self._log_generation_progress(generation, scored_population)
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
            )

        self._ensure_best_defaults_when_empty(
            prompts=prompts,
            population=population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
        )

        return SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics={"mode": "evolutionary"},
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
    ) -> None:
        for item in prompts:
            if item.prompt_id not in hard_prompt_ids:
                continue
            scored = scored_population[item.prompt_id]
            population[item.prompt_id] = self._evolve_one_prompt(
                item=item,
                scored=scored,
                population_for_prompt=population[item.prompt_id],
            )

    def _evolve_one_prompt(
        self,
        *,
        item: PromptExample,
        scored: list[tuple[Rubric, ObjectiveBreakdown]],
        population_for_prompt: list[Rubric],
    ) -> list[Rubric]:
        del population_for_prompt  # Reserved for future prompt-local stateful strategies.
        best_current = scored[0][0]
        left, right = _build_top2_pair(item=item, rubric=best_current, evaluator=self.evaluator)
        new_candidates: list[Rubric] = []
        for idx in range(self.config.mutations_per_round):
            mode = MUTATION_MODES[idx % len(MUTATION_MODES)]
            mutated = self.proposer.propose(
                item.prompt, left, right, best_current, mode=mode, rng=self.rng
            )
            new_candidates.append(mutated)
        winners = self._successive_halving(item, new_candidates)
        return self._update_population(scored, winners, self.config.population_size)

    def _ensure_best_defaults_when_empty(
        self,
        *,
        prompts: list[PromptExample],
        population: dict[str, list[Rubric]],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
    ) -> None:
        if best_rubrics:
            return
        for item in prompts:
            scored = self._score_population(item, population[item.prompt_id])
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
        new_population: list[Rubric] = []
        seen: set[str] = set()
        existing_sorted = [rubric for rubric, _ in scored_existing]
        for rubric in existing_sorted[: self.config.elitism_count]:
            _append_unique(new_population, rubric, seen)
        for rubric in winners:
            if len(new_population) >= target_size:
                break
            _append_unique(new_population, rubric, seen)
        for rubric in existing_sorted:
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
            margin_term = 1.0 / (best_breakdown.top_margin + 1e-6)
            disagreement = disagreement_score(
                prompt_map[prompt_id], populations[prompt_id], self.evaluator
            )
            hardness_value = margin_term + (2.0 * disagreement)
            hardness.append((prompt_id, hardness_value))
        hardness.sort(key=lambda pair: pair[1], reverse=True)
        chosen = hardness[: self.config.batch_size]
        return {prompt_id for prompt_id, _ in chosen}


def _lookup_candidate(item: PromptExample, candidate_id: str):
    for candidate in item.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(f"candidate_id {candidate_id} not found in prompt {item.prompt_id}")


def _fingerprint(rubric: Rubric) -> str:
    return rubric.fingerprint()


def _append_unique(population: list[Rubric], rubric: Rubric, seen: set[str]) -> bool:
    fp = _fingerprint(rubric)
    if fp in seen:
        return False
    population.append(rubric)
    seen.add(fp)
    return True


def _evaluate_rubric(
    *,
    item: PromptExample,
    rubric: Rubric,
    evaluator: RubricEvaluator,
    judge: PreferenceJudge,
    objective: ObjectiveConfig,
    pair_budget: int | None = None,
) -> ObjectiveBreakdown:
    return compute_objective(
        item=item,
        rubric=rubric,
        evaluator=evaluator,
        judge=judge,
        config=objective,
        pair_budget=pair_budget,
    )


def _build_top2_pair(
    *,
    item: PromptExample,
    rubric: Rubric,
    evaluator: RubricEvaluator,
) -> tuple[ResponseCandidate, ResponseCandidate]:
    left_id, right_id = top2_under_rubric(item, rubric, evaluator)
    return _lookup_candidate(item, left_id), _lookup_candidate(item, right_id)

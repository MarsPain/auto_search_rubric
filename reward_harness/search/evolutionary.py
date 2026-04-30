from __future__ import annotations

import logging
import math
import random

from ..evaluator import ObjectiveBreakdown, RubricEvaluator, disagreement_score
from ..interfaces import (
    CheckpointCallback,
    PreferenceJudge,
    RubricInitializer,
    RubricProposer,
    StepState,
    Verifier,
)
from ..data_models import PromptExample, Rubric
from ..types import EvolutionIterationScope, MutationMode
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
    _build_margin_entry,
    _build_top2_pair,
    _evaluate_rubric,
    _fingerprint,
    _summarize_margin_improvement,
)

logger = logging.getLogger("reward_harness.search")


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
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> None:
        self.proposer = proposer
        self.judge = judge
        self.initializer = initializer
        self.config = config or EvolutionaryConfig()
        self.rng = random.Random(self.config.seed)
        self.evaluator = RubricEvaluator(verifier, base_seed=self.config.seed)

        # Use injected scheduler/metric when provided, otherwise create defaults from config.
        self._injected_mutation_scheduler = mutation_scheduler
        self.mutation_scheduler = mutation_scheduler or create_mutation_scheduler(
            self.config,
            self.rng,
        )
        self.diversity_metric = diversity_metric or create_diversity_metric()
        self._checkpoint_callback = checkpoint_callback

        # Track diversity history for diagnostics
        self.diversity_history: list[float] = []

    @property
    def supports_step_execution(self) -> bool:
        return self.config.iteration_scope in {
            EvolutionIterationScope.GLOBAL_BATCH,
            EvolutionIterationScope.PROMPT_LOCAL,
        }

    @property
    def max_steps(self) -> int:
        return self.config.generations

    def initialize_step_state(self, prompts: list[PromptExample]) -> StepState:
        if self.config.iteration_scope is EvolutionIterationScope.PROMPT_LOCAL:
            return self._init_prompt_local_step_state(prompts)
        population, history, best_rubrics, best_scores, initial_margins = (
            self._init_global_state(prompts)
        )
        return {
            "iteration_scope": EvolutionIterationScope.GLOBAL_BATCH.value,
            "population": population,
            "history": history,
            "best_rubrics": best_rubrics,
            "best_scores": best_scores,
            "initial_margins": initial_margins,
            "stale_rounds": 0,
            "should_stop": False,
        }

    def step(
        self, prompts: list[PromptExample], state: StepState, generation: int
    ) -> None:
        if state.get("iteration_scope") == EvolutionIterationScope.PROMPT_LOCAL.value:
            self._step_prompt_local(prompts, state)
            return

        population = state["population"]
        history = state["history"]
        best_rubrics = state["best_rubrics"]
        best_scores = state["best_scores"]
        initial_margins = state["initial_margins"]

        scored_population = self._score_generation(prompts, population)
        self._log_generation_progress(generation, scored_population, initial_margins)

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

        stale_rounds = int(state["stale_rounds"])
        stale_rounds, should_stop = self._handle_stagnation(
            generation_improved, stale_rounds
        )
        state["stale_rounds"] = stale_rounds
        state["should_stop"] = should_stop

        if should_stop:
            logger.info(
                "stopping early at generation=%d stale_rounds=%d threshold=%d",
                generation + 1,
                stale_rounds,
                self.config.stagnation_generations,
            )
            return

        hard_prompt_ids = self._select_hard_prompts(
            prompts, population, scored_population
        )
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
            scheduler=self.mutation_scheduler,
        )

        self.mutation_scheduler.next_generation()

    def finalize_step_result(
        self, prompts: list[PromptExample], state: StepState
    ) -> SearchResult:
        if state.get("iteration_scope") == EvolutionIterationScope.PROMPT_LOCAL.value:
            return self._finalize_prompt_local_step_result(state)

        population = state["population"]
        history = state["history"]
        best_rubrics = state["best_rubrics"]
        best_scores = state["best_scores"]
        initial_margins = state["initial_margins"]

        self._finalize_best_from_population(
            prompts=prompts,
            population=population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
        )
        margin_improvement = self._collect_margin_improvement(
            prompts=prompts,
            best_rubrics=best_rubrics,
            initial_margins=initial_margins,
        )

        diagnostics = {
            "mode": "evolutionary",
            "iteration_scope": self.config.iteration_scope.value,
            "selection_strategy": self.config.selection_strategy.name,
            "adaptive_mutation": self.config.adaptive_mutation.name,
            "mutation_diagnostics": self.mutation_scheduler.get_diagnostics(),
            "avg_diversity": sum(self.diversity_history) / len(self.diversity_history)
            if self.diversity_history
            else 0.0,
            "margin_improvement": margin_improvement,
        }

        return SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics=diagnostics,
        )

    def get_algorithm_state(self, state: StepState | None) -> StepState:
        if state is None:
            return {}
        if state.get("iteration_scope") == EvolutionIterationScope.PROMPT_LOCAL.value:
            return self._get_prompt_local_algorithm_state(state)

        population = state.get("population")
        if not isinstance(population, dict):
            return {}

        serialized_population: dict[str, list[dict[str, object]]] = {}
        for prompt_id, rubrics in population.items():
            if not isinstance(prompt_id, str) or not isinstance(rubrics, list):
                continue
            serialized_population[prompt_id] = [rubric.to_dict() for rubric in rubrics]

        return {
            "iteration_scope": EvolutionIterationScope.GLOBAL_BATCH.value,
            "population": serialized_population,
            "initial_margins": {
                key: float(value)
                for key, value in state.get("initial_margins", {}).items()
            },
            "stale_rounds": int(state.get("stale_rounds", 0)),
            "should_stop": bool(state.get("should_stop", False)),
            "diversity_history": list(self.diversity_history),
        }

    def restore_algorithm_state(
        self,
        algorithm_state: StepState,
        *,
        prompt_ids: list[str],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
        history: dict[str, list[float]],
    ) -> StepState | None:
        if (
            algorithm_state.get("iteration_scope")
            == EvolutionIterationScope.PROMPT_LOCAL.value
        ):
            return self._restore_prompt_local_algorithm_state(
                algorithm_state,
                prompt_ids=prompt_ids,
                best_rubrics=best_rubrics,
                best_scores=best_scores,
                history=history,
            )

        population_raw = algorithm_state.get("population")
        if not isinstance(population_raw, dict):
            return None

        population: dict[str, list[Rubric]] = {}
        for prompt_id in prompt_ids:
            raw_rubrics = population_raw.get(prompt_id)
            if not isinstance(raw_rubrics, list):
                return None
            restored_rubrics: list[Rubric] = []
            for payload in raw_rubrics:
                if not isinstance(payload, dict):
                    return None
                restored_rubrics.append(Rubric.from_dict(payload))
            population[prompt_id] = restored_rubrics

        initial_margins_raw = algorithm_state.get("initial_margins", {})
        if not isinstance(initial_margins_raw, dict):
            initial_margins_raw = {}
        diversity_history = algorithm_state.get("diversity_history", [])
        if isinstance(diversity_history, list):
            self.diversity_history = [float(item) for item in diversity_history]

        return {
            "population": population,
            "history": {
                prompt_id: list(history.get(prompt_id, [])) for prompt_id in prompt_ids
            },
            "best_rubrics": dict(best_rubrics),
            "best_scores": dict(best_scores),
            "initial_margins": {
                prompt_id: float(initial_margins_raw.get(prompt_id, 0.0))
                for prompt_id in prompt_ids
            },
            "stale_rounds": int(algorithm_state.get("stale_rounds", 0)),
            "should_stop": bool(algorithm_state.get("should_stop", False)),
        }

    def _init_prompt_local_step_state(self, prompts: list[PromptExample]) -> StepState:
        state: StepState = {
            "iteration_scope": EvolutionIterationScope.PROMPT_LOCAL.value,
            "history": {item.prompt_id: [] for item in prompts},
            "best_rubrics": {},
            "best_scores": {},
            "current_prompt_index": 0,
            "current_prompt_generation": 0,
            "mutation_diagnostics": {},
            "per_prompt_margin_stats": {},
            "improved_prompts": 0,
            "max_steps": len(prompts) * self.config.generations,
            "should_stop": len(prompts) == 0,
        }
        return state

    def _start_prompt_local_prompt(self, item: PromptExample, state: StepState) -> None:
        logger.info("prompt-local init prompt=%s", item.prompt_id)
        population = self._init_population(item)
        state["current_population"] = population
        state["current_initial_margin"] = self._score_population(
            item,
            [population[0]],
        )[0][1].top_margin
        state["current_best_score"] = -math.inf
        state["current_best_rubric"] = population[0]
        state["current_stale_rounds"] = 0
        state["current_prompt_generation"] = 0
        state["current_scheduler"] = self._build_prompt_scheduler()

    def _step_prompt_local(
        self, prompts: list[PromptExample], state: StepState
    ) -> None:
        prompt_index = int(state.get("current_prompt_index", 0))
        if prompt_index >= len(prompts):
            state["should_stop"] = True
            return

        item = prompts[prompt_index]
        if "current_population" not in state:
            self._start_prompt_local_prompt(item, state)

        population = state["current_population"]
        scheduler = state["current_scheduler"]
        generation = int(state.get("current_prompt_generation", 0))
        initial_margin = float(state.get("current_initial_margin", 0.0))
        best_score = float(state.get("current_best_score", -math.inf))
        best_rubric = state.get("current_best_rubric", population[0])

        scored = self._score_population(item, population)
        best_current, best_breakdown = scored[0]
        history = state["history"]
        history.setdefault(item.prompt_id, []).append(best_breakdown.total)

        improved = False
        if best_breakdown.total > best_score:
            best_score = best_breakdown.total
            best_rubric = best_current
            improved = True

        state["current_best_score"] = best_score
        state["current_best_rubric"] = best_rubric
        state["best_scores"][item.prompt_id] = best_score
        state["best_rubrics"][item.prompt_id] = best_rubric

        logger.info(
            "prompt=%s generation=%d/%d best_score=%.4f top_margin=%.6f signed_margin=%.6f delta_vs_init=%.6f signed_delta=%.6f",
            item.prompt_id,
            generation + 1,
            self.config.generations,
            best_breakdown.total,
            best_breakdown.top_margin,
            best_breakdown.signed_top_margin,
            best_breakdown.top_margin - initial_margin,
            best_breakdown.signed_top_margin - initial_margin,
        )

        diversity = self.diversity_metric.compute(population, rng=self.rng)
        self.diversity_history.append(diversity)

        prompt_finished = False
        if self._is_prompt_distinguished(best_breakdown):
            logger.info(
                "prompt=%s stopping early at generation=%d reason=distinguished margin=%.6f signed_margin=%.6f",
                item.prompt_id,
                generation + 1,
                best_breakdown.top_margin,
                best_breakdown.signed_top_margin,
            )
            prompt_finished = True
        else:
            stale_rounds, should_stop = self._handle_stagnation(
                improved,
                int(state.get("current_stale_rounds", 0)),
            )
            state["current_stale_rounds"] = stale_rounds
            if should_stop:
                logger.info(
                    "prompt=%s stopping early at generation=%d stale_rounds=%d threshold=%d",
                    item.prompt_id,
                    generation + 1,
                    stale_rounds,
                    self.config.stagnation_generations,
                )
                prompt_finished = True

        if not prompt_finished:
            population = self._evolve_one_prompt(
                item=item,
                scored=scored,
                population_for_prompt=population,
                generation=generation,
                diversity_score=diversity,
                scheduler=scheduler,
            )
            scheduler.next_generation()
            state["current_population"] = population

        generation += 1
        state["current_prompt_generation"] = generation
        if generation >= self.config.generations:
            prompt_finished = True

        if prompt_finished:
            self._finalize_prompt_local_prompt(
                item=item,
                state=state,
                population=population,
                scheduler=scheduler,
            )
            self._advance_prompt_local_prompt(prompts, state)

    def _finalize_prompt_local_prompt(
        self,
        *,
        item: PromptExample,
        state: StepState,
        population: list[Rubric],
        scheduler: MutationScheduler,
    ) -> None:
        final_scored = self._score_population(item, population)
        final_best, final_breakdown = final_scored[0]
        best_score = float(state.get("current_best_score", -math.inf))
        best_rubric = state.get("current_best_rubric", final_best)
        if final_breakdown.total > best_score:
            best_score = final_breakdown.total
            best_rubric = final_best

        state["best_scores"][item.prompt_id] = best_score
        state["best_rubrics"][item.prompt_id] = best_rubric
        state["mutation_diagnostics"][item.prompt_id] = scheduler.get_diagnostics()

        initial_margin = float(state.get("current_initial_margin", 0.0))
        best_margin = self._score_population(item, [best_rubric])[0][1].top_margin
        margin_stats = _build_margin_entry(
            initial_margin=initial_margin,
            final_margin=best_margin,
            tolerance=self.config.objective.tie_tolerance,
        )
        state["per_prompt_margin_stats"][item.prompt_id] = margin_stats
        if bool(margin_stats.get("improved")):
            state["improved_prompts"] = int(state.get("improved_prompts", 0)) + 1

        processed = len(state["per_prompt_margin_stats"])
        improved_prompts = int(state.get("improved_prompts", 0))
        improvement_rate = (improved_prompts / processed) if processed > 0 else 0.0
        logger.info(
            "prompt-local margin-progress processed=%d improved=%d rate=%.2f%% "
            "prompt=%s initial_margin=%.6f final_margin=%.6f delta=%.6f",
            processed,
            improved_prompts,
            improvement_rate * 100.0,
            item.prompt_id,
            float(margin_stats["initial_margin"]),
            float(margin_stats["final_margin"]),
            float(margin_stats["margin_delta"]),
        )
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(
                dict(state["best_rubrics"]),
                dict(state["best_scores"]),
                {
                    prompt_id: list(scores)
                    for prompt_id, scores in state["history"].items()
                    if scores
                },
            )

    def _advance_prompt_local_prompt(
        self,
        prompts: list[PromptExample],
        state: StepState,
    ) -> None:
        state["current_prompt_index"] = int(state.get("current_prompt_index", 0)) + 1
        state["current_prompt_generation"] = 0
        for key in (
            "current_population",
            "current_initial_margin",
            "current_best_score",
            "current_best_rubric",
            "current_stale_rounds",
            "current_scheduler",
        ):
            state.pop(key, None)
        state["should_stop"] = int(state["current_prompt_index"]) >= len(prompts)

    def _finalize_prompt_local_step_result(self, state: StepState) -> SearchResult:
        margin_improvement = _summarize_margin_improvement(
            state["per_prompt_margin_stats"]
        )
        diagnostics = {
            "mode": "evolutionary",
            "iteration_scope": self.config.iteration_scope.value,
            "selection_strategy": self.config.selection_strategy.name,
            "adaptive_mutation": self.config.adaptive_mutation.name,
            "mutation_diagnostics": state["mutation_diagnostics"],
            "avg_diversity": sum(self.diversity_history) / len(self.diversity_history)
            if self.diversity_history
            else 0.0,
            "margin_improvement": margin_improvement,
        }
        return SearchResult(
            best_rubrics=state["best_rubrics"],
            best_scores=state["best_scores"],
            history=state["history"],
            diagnostics=diagnostics,
        )

    def _get_prompt_local_algorithm_state(self, state: StepState) -> StepState:
        payload: StepState = {
            "iteration_scope": EvolutionIterationScope.PROMPT_LOCAL.value,
            "current_prompt_index": int(state.get("current_prompt_index", 0)),
            "current_prompt_generation": int(state.get("current_prompt_generation", 0)),
            "mutation_diagnostics": state.get("mutation_diagnostics", {}),
            "per_prompt_margin_stats": state.get("per_prompt_margin_stats", {}),
            "improved_prompts": int(state.get("improved_prompts", 0)),
            "max_steps": int(state.get("max_steps", 0)),
            "should_stop": bool(state.get("should_stop", False)),
            "diversity_history": list(self.diversity_history),
        }

        if "current_initial_margin" in state:
            payload["current_initial_margin"] = float(state["current_initial_margin"])
        if "current_best_score" in state:
            payload["current_best_score"] = float(state["current_best_score"])
        if "current_stale_rounds" in state:
            payload["current_stale_rounds"] = int(state["current_stale_rounds"])

        current_population = state.get("current_population")
        if isinstance(current_population, list):
            payload["current_population"] = [
                rubric.to_dict()
                for rubric in current_population
                if isinstance(rubric, Rubric)
            ]

        current_best_rubric = state.get("current_best_rubric")
        if isinstance(current_best_rubric, Rubric):
            payload["current_best_rubric"] = current_best_rubric.to_dict()

        current_scheduler = state.get("current_scheduler")
        if current_scheduler is not None and hasattr(current_scheduler, "get_state"):
            payload["current_scheduler_state"] = current_scheduler.get_state()

        return payload

    def _restore_prompt_local_algorithm_state(
        self,
        algorithm_state: StepState,
        *,
        prompt_ids: list[str],
        best_rubrics: dict[str, Rubric],
        best_scores: dict[str, float],
        history: dict[str, list[float]],
    ) -> StepState | None:
        diversity_history = algorithm_state.get("diversity_history", [])
        if isinstance(diversity_history, list):
            self.diversity_history = [float(item) for item in diversity_history]

        state: StepState = {
            "iteration_scope": EvolutionIterationScope.PROMPT_LOCAL.value,
            "history": {
                prompt_id: list(history.get(prompt_id, [])) for prompt_id in prompt_ids
            },
            "best_rubrics": dict(best_rubrics),
            "best_scores": dict(best_scores),
            "current_prompt_index": int(algorithm_state.get("current_prompt_index", 0)),
            "current_prompt_generation": int(
                algorithm_state.get("current_prompt_generation", 0)
            ),
            "current_initial_margin": float(
                algorithm_state.get("current_initial_margin", 0.0)
            ),
            "current_best_score": float(
                algorithm_state.get("current_best_score", -math.inf)
            ),
            "current_stale_rounds": int(algorithm_state.get("current_stale_rounds", 0)),
            "mutation_diagnostics": algorithm_state.get("mutation_diagnostics", {}),
            "per_prompt_margin_stats": algorithm_state.get(
                "per_prompt_margin_stats", {}
            ),
            "improved_prompts": int(algorithm_state.get("improved_prompts", 0)),
            "max_steps": int(
                algorithm_state.get(
                    "max_steps", len(prompt_ids) * self.config.generations
                )
            ),
            "should_stop": bool(algorithm_state.get("should_stop", False)),
        }

        population_raw = algorithm_state.get("current_population")
        if isinstance(population_raw, list):
            current_population: list[Rubric] = []
            for payload in population_raw:
                if not isinstance(payload, dict):
                    return None
                current_population.append(Rubric.from_dict(payload))
            state["current_population"] = current_population

        best_rubric_raw = algorithm_state.get("current_best_rubric")
        if isinstance(best_rubric_raw, dict):
            state["current_best_rubric"] = Rubric.from_dict(best_rubric_raw)

        if "current_population" in state:
            scheduler = self._build_prompt_scheduler()
            scheduler_state = algorithm_state.get("current_scheduler_state")
            if isinstance(scheduler_state, dict) and hasattr(scheduler, "set_state"):
                scheduler.set_state(scheduler_state)
            state["current_scheduler"] = scheduler

        return state

    def search(self, prompts: list[PromptExample]) -> SearchResult:
        logger.info(
            "starting evolutionary search prompts=%d generations=%d population=%d "
            "selection=%s mutation=%s scope=%s",
            len(prompts),
            self.config.generations,
            self.config.population_size,
            self.config.selection_strategy.name,
            self.config.adaptive_mutation.name,
            self.config.iteration_scope.value,
        )
        if self.config.iteration_scope is EvolutionIterationScope.PROMPT_LOCAL:
            return self._search_prompt_local(prompts)
        return self._search_global_batch(prompts)

    def _search_global_batch(self, prompts: list[PromptExample]) -> SearchResult:
        population, history, best_rubrics, best_scores, initial_margins = (
            self._init_global_state(prompts)
        )
        stale_rounds = 0

        for generation in range(self.config.generations):
            scored_population = self._score_generation(prompts, population)
            self._log_generation_progress(
                generation, scored_population, initial_margins
            )

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
            stale_rounds, should_stop = self._handle_stagnation(
                generation_improved, stale_rounds
            )
            if should_stop:
                logger.info(
                    "stopping early at generation=%d stale_rounds=%d threshold=%d",
                    generation + 1,
                    stale_rounds,
                    self.config.stagnation_generations,
                )
                break

            hard_prompt_ids = self._select_hard_prompts(
                prompts, population, scored_population
            )
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
                scheduler=self.mutation_scheduler,
            )

            # Update mutation selector for next generation
            self.mutation_scheduler.next_generation()

        self._finalize_best_from_population(
            prompts=prompts,
            population=population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
        )
        margin_improvement = self._collect_margin_improvement(
            prompts=prompts,
            best_rubrics=best_rubrics,
            initial_margins=initial_margins,
        )

        diagnostics = {
            "mode": "evolutionary",
            "iteration_scope": self.config.iteration_scope.value,
            "selection_strategy": self.config.selection_strategy.name,
            "adaptive_mutation": self.config.adaptive_mutation.name,
            "mutation_diagnostics": self.mutation_scheduler.get_diagnostics(),
            "avg_diversity": sum(self.diversity_history) / len(self.diversity_history)
            if self.diversity_history
            else 0.0,
            "margin_improvement": margin_improvement,
        }

        return SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics=diagnostics,
        )

    def _search_prompt_local(self, prompts: list[PromptExample]) -> SearchResult:
        state = self._init_prompt_local_step_state(prompts)
        steps = 0
        max_steps = int(state.get("max_steps", 0))
        while not state.get("should_stop", False) and steps < max_steps:
            self._step_prompt_local(prompts, state)
            steps += 1
        return self._finalize_prompt_local_step_result(state)

    def _init_global_state(
        self,
        prompts: list[PromptExample],
    ) -> tuple[
        dict[str, list[Rubric]],
        dict[str, list[float]],
        dict[str, Rubric],
        dict[str, float],
        dict[str, float],
    ]:
        population: dict[str, list[Rubric]] = {}
        initial_margins: dict[str, float] = {}
        for item in prompts:
            logger.info(
                "initializing population prompt=%s target_size=%d",
                item.prompt_id,
                self.config.population_size,
            )
            population[item.prompt_id] = self._init_population(item)
            initial_margins[item.prompt_id] = self._score_population(
                item,
                [population[item.prompt_id][0]],
            )[0][1].top_margin
        history = {item.prompt_id: [] for item in prompts}
        best_rubrics: dict[str, Rubric] = {}
        best_scores = {item.prompt_id: -math.inf for item in prompts}
        return population, history, best_rubrics, best_scores, initial_margins

    def _log_generation_progress(
        self,
        generation: int,
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
        initial_margins: dict[str, float],
    ) -> None:
        per_prompt = []
        improved_prompts = 0
        for prompt_id, scored in sorted(scored_population.items()):
            best_score = scored[0][1].total
            best_margin = scored[0][1].top_margin
            initial_margin = initial_margins.get(prompt_id, 0.0)
            margin_delta = best_margin - initial_margin
            if best_margin > (initial_margin + self.config.objective.tie_tolerance):
                improved_prompts += 1
            signed_margin = scored[0][1].signed_top_margin
            signed_delta = signed_margin - initial_margin
            per_prompt.append(
                f"{prompt_id}:best_score={best_score:.4f},top_margin={best_margin:.6f},"
                f"signed_margin={signed_margin:.6f},delta_vs_init={margin_delta:.6f},"
                f"signed_delta={signed_delta:.6f}"
            )
        logger.info(
            "generation=%d/%d margin_improved_prompts=%d/%d %s",
            generation + 1,
            self.config.generations,
            improved_prompts,
            len(scored_population),
            "; ".join(per_prompt),
        )

    def _collect_margin_improvement(
        self,
        *,
        prompts: list[PromptExample],
        best_rubrics: dict[str, Rubric],
        initial_margins: dict[str, float],
    ) -> dict[str, object]:
        per_prompt: dict[str, dict[str, float | bool]] = {}
        for item in prompts:
            rubric = best_rubrics.get(item.prompt_id)
            if rubric is None:
                continue
            final_margin = self._score_population(item, [rubric])[0][1].top_margin
            per_prompt[item.prompt_id] = _build_margin_entry(
                initial_margin=initial_margins.get(item.prompt_id, 0.0),
                final_margin=final_margin,
                tolerance=self.config.objective.tie_tolerance,
            )
        return _summarize_margin_improvement(per_prompt)

    def _score_generation(
        self,
        prompts: list[PromptExample],
        population: dict[str, list[Rubric]],
    ) -> dict[str, list[tuple[Rubric, ObjectiveBreakdown]]]:
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]] = {}
        for item in prompts:
            scored_population[item.prompt_id] = self._score_population(
                item, population[item.prompt_id]
            )
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

    def _handle_stagnation(
        self, generation_improved: bool, stale_rounds: int
    ) -> tuple[int, bool]:
        if generation_improved:
            stale_rounds = 0
        else:
            stale_rounds += 1
        return stale_rounds, stale_rounds >= self.config.stagnation_generations

    def _build_prompt_scheduler(self) -> MutationScheduler:
        if self._injected_mutation_scheduler is not None:
            return self._injected_mutation_scheduler
        return create_mutation_scheduler(self.config, self.rng)

    def _is_prompt_distinguished(self, breakdown: ObjectiveBreakdown) -> bool:
        if not self.config.stop_when_distinguished:
            return False
        required_margin = self.config.distinguish_margin
        if required_margin is None:
            required_margin = self.config.objective.tie_tolerance
        return (
            breakdown.valid_pairs > 0
            and breakdown.top_margin > required_margin
            and breakdown.signed_top_margin > 0
        )

    def _evolve_selected_prompts(
        self,
        *,
        prompts: list[PromptExample],
        hard_prompt_ids: set[str],
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]],
        population: dict[str, list[Rubric]],
        diversity_scores: dict[str, float],
        generation: int,
        scheduler: MutationScheduler,
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
                scheduler=scheduler,
            )

    def _evolve_one_prompt(
        self,
        *,
        item: PromptExample,
        scored: list[tuple[Rubric, ObjectiveBreakdown]],
        population_for_prompt: list[Rubric],
        generation: int,  # noqa: ARG002
        diversity_score: float,
        scheduler: MutationScheduler,
    ) -> list[Rubric]:
        del (
            population_for_prompt
        )  # Reserved for future prompt-local stateful strategies.
        best_current = scored[0][0]
        best_current_score = scored[0][1].total
        parent_count = min(len(scored), self.config.mutation_parent_count)
        selected_parents = select_parents(
            strategy=self.config.selection_strategy,
            scored_population=scored,
            num_parents=max(1, parent_count),
            rng=self.rng,
            config=self.config,
        )
        parent_pool: list[Rubric] = []
        seen_parent_fps: set[str] = set()
        _append_unique(parent_pool, best_current, seen_parent_fps)
        for parent in selected_parents:
            _append_unique(parent_pool, parent, seen_parent_fps)

        if not parent_pool:
            parent_pool.append(best_current)

        new_candidates: list[Rubric] = []
        mutation_modes_used: list[MutationMode] = []

        for idx in range(self.config.mutations_per_round):
            parent = parent_pool[idx % len(parent_pool)]
            left, right = _build_top2_pair(
                item=item, rubric=parent, evaluator=self.evaluator
            )
            # Use adaptive mutation selection
            mode = scheduler.select_mode(diversity_score=diversity_score)
            mutation_modes_used.append(mode)
            mutated = self.proposer.propose(
                item.prompt, left, right, parent, mode=mode, rng=self.rng
            )
            new_candidates.append(mutated)

        logger.debug(
            "prompt=%s parent_pool_size=%d mutation_modes=%s",
            item.prompt_id,
            len(parent_pool),
            [m.name for m in mutation_modes_used],
        )

        winners = self._successive_halving(item, new_candidates)

        # Record mutation outcomes for adaptive learning
        winner_scores = self._score_population(item, winners)
        best_winner_score = (
            winner_scores[0][1].total if winner_scores else best_current_score
        )
        improvement = best_winner_score - best_current_score

        for mode in mutation_modes_used:
            scheduler.record_outcome(
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
            mutated = self.proposer.propose(
                item.prompt, left, right, base, mode=mode, rng=self.rng
            )
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

    def _successive_halving(
        self, item: PromptExample, candidates: list[Rubric]
    ) -> list[Rubric]:
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
            keep_count = max(
                1, int(math.ceil(len(scored) * self.config.survival_fraction))
            )
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

from __future__ import annotations

import logging
import math
import random

from ..evaluator import ObjectiveBreakdown, RubricEvaluator
from ..interfaces import PreferenceJudge, RubricInitializer, RubricProposer, Verifier
from ..models import PromptExample, ResponseCandidate, Rubric
from ..types import MutationMode
from .config import IterativeConfig, SearchResult
from .strategies import MUTATION_MODES, _build_top2_pair, _evaluate_rubric

logger = logging.getLogger("autosr.search")


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

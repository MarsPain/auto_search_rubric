from __future__ import annotations

from ..config import ObjectiveConfig
from ..evaluator import ObjectiveBreakdown, RubricEvaluator, compute_objective, top2_under_rubric
from ..interfaces import PreferenceJudge
from ..data_models import PromptExample, ResponseCandidate, Rubric
from ..types import MutationMode


# Backward compatibility: MUTATION_MODES now provides the enum members
MUTATION_MODES: tuple[MutationMode, ...] = MutationMode.default_cycle()


def _lookup_candidate(item: PromptExample, candidate_id: str) -> ResponseCandidate:
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

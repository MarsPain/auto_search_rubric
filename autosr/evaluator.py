from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import itertools
import math
import random
import statistics
from typing import Iterable

from .config import ObjectiveConfig
from .interfaces import PreferenceJudge, Verifier
from .data_models import PromptExample, ResponseCandidate, Rubric


@dataclass(slots=True)
class CandidateEvaluation:
    candidate_id: str
    score: float
    variance: float
    majority_grades: dict[str, float | None]
    vote_scores: list[float]
    per_vote_grades: list[dict[str, float | None]] = field(default_factory=list)


@dataclass(slots=True)
class ObjectiveBreakdown:
    total: float
    tail_acc: float
    tail_var: float
    diverse_tail_acc: float
    valid_pairs: int
    diverse_pairs: int
    top_margin: float
    signed_top_margin: float


@dataclass(slots=True)
class PairAlignmentStats:
    correct: int
    valid: int
    diverse_correct: int
    diverse_valid: int


class RubricEvaluator:
    def __init__(self, verifier: Verifier, *, base_seed: int = 7) -> None:
        self.verifier = verifier
        self.base_seed = base_seed
        self._candidate_cache: dict[tuple[str, str, str, int], CandidateEvaluation] = {}

    def evaluate_candidates(
        self,
        item: PromptExample,
        rubric: Rubric,
        *,
        vote_override: int | None = None,
        use_cache: bool = True,
    ) -> list[CandidateEvaluation]:
        out: list[CandidateEvaluation] = []
        for candidate in item.candidates:
            evaluation = self.evaluate_single_candidate(
                prompt_id=item.prompt_id,
                prompt=item.prompt,
                candidate=candidate,
                rubric=rubric,
                vote_override=vote_override,
                use_cache=use_cache,
            )
            out.append(evaluation)
        out.sort(key=lambda ev: ev.score, reverse=True)
        return out

    def evaluate_single_candidate(
        self,
        *,
        prompt_id: str,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        vote_override: int | None = None,
        use_cache: bool = True,
    ) -> CandidateEvaluation:
        votes = vote_override or rubric.grading_protocol.num_votes
        rubric_fp = rubric.fingerprint()
        key = (prompt_id, candidate.candidate_id, rubric_fp, votes)
        if use_cache and key in self._candidate_cache:
            return self._candidate_cache[key]
        evaluation = self._evaluate_single_uncached(
            prompt_id=prompt_id,
            prompt=prompt,
            candidate=candidate,
            rubric=rubric,
            votes=votes,
        )
        if use_cache:
            self._candidate_cache[key] = evaluation
        return evaluation

    def _evaluate_one(
        self,
        item: PromptExample,
        candidate_id: str,
        rubric: Rubric,
        votes: int,
    ) -> CandidateEvaluation:
        candidate = next(c for c in item.candidates if c.candidate_id == candidate_id)
        return self._evaluate_single_uncached(
            prompt_id=item.prompt_id,
            prompt=item.prompt,
            candidate=candidate,
            rubric=rubric,
            votes=votes,
        )

    def _evaluate_single_uncached(
        self,
        *,
        prompt_id: str,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        votes: int,
    ) -> CandidateEvaluation:
        per_vote_grades: list[dict[str, float | None]] = []
        vote_scores: list[float] = []
        for vote_idx in range(votes):
            seed = self._vote_seed(prompt_id, candidate.candidate_id, rubric, vote_idx)
            grades = self.verifier.grade(prompt, candidate, rubric, seed=seed)
            per_vote_grades.append(grades)
            vote_scores.append(rubric.score_from_grades(grades))

        majority: dict[str, float | None] = {}
        for criterion in rubric.criteria:
            votes_for_criterion = [
                grades.get(criterion.criterion_id)
                for grades in per_vote_grades
                if grades.get(criterion.criterion_id) is not None
            ]
            if not votes_for_criterion:
                majority[criterion.criterion_id] = None
                continue
            majority[criterion.criterion_id] = float(sum(votes_for_criterion) / len(votes_for_criterion))

        final_score = rubric.score_from_grades(majority)
        variance = statistics.pvariance(vote_scores) if len(vote_scores) > 1 else 0.0
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            score=final_score,
            variance=variance,
            majority_grades=majority,
            vote_scores=vote_scores,
            per_vote_grades=per_vote_grades,
        )

    def _vote_seed(self, prompt_id: str, candidate_id: str, rubric: Rubric, vote_idx: int) -> int:
        token = f"{self.base_seed}|{prompt_id}|{candidate_id}|{rubric.fingerprint()}|{vote_idx}"
        digest = sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


def compute_objective(
    item: PromptExample,
    rubric: Rubric,
    evaluator: RubricEvaluator,
    judge: PreferenceJudge,
    config: ObjectiveConfig,
    *,
    pair_budget: int | None = None,
) -> ObjectiveBreakdown:
    scored = evaluator.evaluate_candidates(item, rubric)
    tail = _resolve_tail(scored, config.tail_fraction)
    top_margin = _compute_top_margin(scored)
    signed_top_margin = _compute_signed_top_margin(scored, item)
    tail_var = _compute_tail_variance(tail)
    pairs = _build_tail_pairs(tail)
    resolved_budget = config.pair_budget_full if pair_budget is None else pair_budget
    sampled_pairs = _apply_pair_budget(pairs, item.prompt_id, rubric.rubric_id, resolved_budget)
    stats = _evaluate_pair_alignment(
        item=item,
        pair_candidates=sampled_pairs,
        judge=judge,
        tie_tolerance=config.tie_tolerance,
    )
    tail_acc_raw, diverse_tail_acc_raw = _compute_tail_metrics(stats)
    tail_acc = _shrink_toward_neutral(tail_acc_raw, stats.valid, config.pair_confidence_prior)
    diverse_tail_acc = _shrink_toward_neutral(
        diverse_tail_acc_raw,
        stats.valid,
        config.pair_confidence_prior,
    )
    return _compose_objective_breakdown(
        tail_acc=tail_acc,
        tail_var=tail_var,
        diverse_tail_acc=diverse_tail_acc,
        valid_pairs=stats.valid,
        diverse_pairs=stats.diverse_valid,
        top_margin=top_margin,
        signed_top_margin=signed_top_margin,
        lambda_var=config.lambda_var,
        mu_diverse=config.mu_diverse,
    )


def top2_under_rubric(
    item: PromptExample,
    rubric: Rubric,
    evaluator: RubricEvaluator,
) -> tuple[str, str]:
    scored = evaluator.evaluate_candidates(item, rubric)
    if len(scored) < 2:
        raise ValueError(f"prompt {item.prompt_id} has fewer than two candidates")
    return scored[0].candidate_id, scored[1].candidate_id


def disagreement_score(
    item: PromptExample,
    rubrics: Iterable[Rubric],
    evaluator: RubricEvaluator,
) -> float:
    top_ids: list[str] = []
    for rubric in rubrics:
        scored = evaluator.evaluate_candidates(item, rubric)
        top_ids.append(scored[0].candidate_id)
    if not top_ids:
        return 0.0
    counts: dict[str, int] = {}
    for candidate_id in top_ids:
        counts[candidate_id] = counts.get(candidate_id, 0) + 1
    mode_count = max(counts.values())
    return 1.0 - (mode_count / len(top_ids))


def _stable_seed(*parts: str) -> int:
    payload = "|".join(parts)
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _resolve_tail(
    scored: list[CandidateEvaluation],
    tail_fraction: float,
) -> list[CandidateEvaluation]:
    tail_n = max(2, int(math.ceil(len(scored) * tail_fraction)))
    tail_n = min(tail_n, len(scored))
    return scored[:tail_n]


def _compute_top_margin(scored: list[CandidateEvaluation]) -> float:
    if len(scored) <= 1:
        return 0.0
    return scored[0].score - scored[1].score


def _ground_truth_top2(item: PromptExample) -> tuple[str, str] | None:
    """Return (best_id, runner_up_id) based on metadata rank or quality if available."""
    candidates = item.candidates
    if len(candidates) < 2:
        return None

    # Try rank first (lower is better)
    ranked = [(c, c.metadata.get("rank")) for c in candidates]
    if all(r is not None for _, r in ranked):
        ranked.sort(key=lambda x: x[1])  # type: ignore[arg-return-type]
        return ranked[0][0].candidate_id, ranked[1][0].candidate_id

    # Fallback to quality (higher is better)
    scored = [(c, c.metadata.get("quality")) for c in candidates]
    if all(q is not None for _, q in scored):
        scored.sort(key=lambda x: x[1], reverse=True)  # type: ignore[arg-return-type]
        return scored[0][0].candidate_id, scored[1][0].candidate_id

    return None


def _compute_signed_top_margin(
    scored: list[CandidateEvaluation],
    item: PromptExample,
) -> float:
    """Compute margin between ground-truth top-2 under the current rubric scoring.

    When ground-truth ordering is available (via metadata.rank or metadata.quality),
    returns ``score(gt_best) - score(gt_runner_up)``, which is positive when the
    rubric ranks them in the correct direction and negative when reversed.

    When no ground-truth ordering is present, falls back to the unsigned
    ``top_margin`` so that downstream logic (e.g. early stopping) remains valid.
    """
    gt = _ground_truth_top2(item)
    if gt is None:
        return _compute_top_margin(scored)

    best_id, runner_up_id = gt
    lookup = {ev.candidate_id: ev.score for ev in scored}
    best_score = lookup.get(best_id)
    runner_up_score = lookup.get(runner_up_id)
    if best_score is None or runner_up_score is None:
        return _compute_top_margin(scored)
    return best_score - runner_up_score


def _compute_tail_variance(tail: list[CandidateEvaluation]) -> float:
    var_values = [candidate.variance for candidate in tail]
    if not var_values:
        return 0.0
    return float(sum(var_values) / len(var_values))


def _build_tail_pairs(
    tail: list[CandidateEvaluation],
) -> list[tuple[CandidateEvaluation, CandidateEvaluation]]:
    return list(itertools.combinations(tail, 2))


def _apply_pair_budget(
    pairs: list[tuple[CandidateEvaluation, CandidateEvaluation]],
    prompt_id: str,
    rubric_id: str,
    resolved_budget: int,
) -> list[tuple[CandidateEvaluation, CandidateEvaluation]]:
    if resolved_budget <= 0 or len(pairs) <= resolved_budget:
        return pairs
    rng = random.Random(_stable_seed(prompt_id, rubric_id, str(resolved_budget)))
    return rng.sample(pairs, resolved_budget)


def _evaluate_pair_alignment(
    *,
    item: PromptExample,
    pair_candidates: list[tuple[CandidateEvaluation, CandidateEvaluation]],
    judge: PreferenceJudge,
    tie_tolerance: float,
) -> PairAlignmentStats:
    candidate_lookup = {candidate.candidate_id: candidate for candidate in item.candidates}
    stats = PairAlignmentStats(correct=0, valid=0, diverse_correct=0, diverse_valid=0)
    for left_eval, right_eval in pair_candidates:
        left = candidate_lookup[left_eval.candidate_id]
        right = candidate_lookup[right_eval.candidate_id]
        pref = judge.compare(item.prompt, left, right)
        if pref == 0:
            continue
        diff = left_eval.score - right_eval.score
        if abs(diff) <= tie_tolerance:
            model_pref = 0
        else:
            model_pref = 1 if diff > 0 else -1
        stats.valid += 1
        matched = model_pref == pref
        if matched:
            stats.correct += 1
        if left.source != right.source:
            stats.diverse_valid += 1
            if matched:
                stats.diverse_correct += 1
    return stats


def _compute_tail_metrics(stats: PairAlignmentStats) -> tuple[float, float]:
    tail_acc = (stats.correct / stats.valid) if stats.valid > 0 else 0.5
    if stats.diverse_valid > 0:
        diverse_tail_acc = stats.diverse_correct / stats.diverse_valid
    else:
        # If no cross-source pair exists in tail, fall back to tail_acc.
        diverse_tail_acc = tail_acc
    return tail_acc, diverse_tail_acc


def _shrink_toward_neutral(value: float, sample_count: int, prior: float) -> float:
    if prior <= 0:
        return value
    confidence = sample_count / (sample_count + prior)
    return 0.5 + (confidence * (value - 0.5))


def _compose_objective_breakdown(
    *,
    tail_acc: float,
    tail_var: float,
    diverse_tail_acc: float,
    valid_pairs: int,
    diverse_pairs: int,
    top_margin: float,
    signed_top_margin: float,
    lambda_var: float,
    mu_diverse: float,
) -> ObjectiveBreakdown:
    total = tail_acc - (lambda_var * tail_var) + (mu_diverse * diverse_tail_acc)
    return ObjectiveBreakdown(
        total=total,
        tail_acc=tail_acc,
        tail_var=tail_var,
        diverse_tail_acc=diverse_tail_acc,
        valid_pairs=valid_pairs,
        diverse_pairs=diverse_pairs,
        top_margin=top_margin,
        signed_top_margin=signed_top_margin,
    )

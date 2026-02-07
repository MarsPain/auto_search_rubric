from __future__ import annotations

import random
import unittest

from autosr.evaluator import (
    CandidateEvaluation,
    ObjectiveConfig,
    RubricEvaluator,
    compute_objective,
)
from autosr.io_utils import load_dataset
from autosr.mock_components import (
    HeuristicPreferenceJudge,
    HeuristicRubricInitializer,
    HeuristicVerifier,
)
from autosr.models import Criterion, PromptExample, ResponseCandidate, Rubric


class _StaticEvaluator:
    def __init__(self, scored: list[CandidateEvaluation]) -> None:
        self._scored = scored

    def evaluate_candidates(self, item, rubric):  # noqa: ANN001 - test double
        return list(self._scored)


class _StaticJudge:
    def __init__(self, pref: int) -> None:
        self._pref = pref

    def compare(self, prompt: str, left: ResponseCandidate, right: ResponseCandidate) -> int:
        return self._pref


class _AlwaysTieJudge:
    def compare(self, prompt: str, left: ResponseCandidate, right: ResponseCandidate) -> int:
        return 0


class TestEvaluatorObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = load_dataset("examples/demo_dataset.json")

    def test_fixed_seed_objective_regression(self) -> None:
        item = self.prompts[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))
        evaluator = RubricEvaluator(HeuristicVerifier(noise=0.0), base_seed=7)
        result = compute_objective(
            item,
            rubric,
            evaluator,
            HeuristicPreferenceJudge(),
            ObjectiveConfig(),
        )
        self.assertAlmostEqual(result.total, 1.25, places=8)
        self.assertAlmostEqual(result.tail_acc, 1.0, places=8)
        self.assertAlmostEqual(result.diverse_tail_acc, 1.0, places=8)
        self.assertEqual(result.valid_pairs, 1)
        self.assertEqual(result.diverse_pairs, 1)

    def test_pair_budget_paths_are_deterministic(self) -> None:
        item = self.prompts[1]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))
        evaluator = RubricEvaluator(HeuristicVerifier(noise=0.0), base_seed=7)
        judge = HeuristicPreferenceJudge()
        config = ObjectiveConfig(tail_fraction=1.0)

        full = compute_objective(item, rubric, evaluator, judge, config, pair_budget=None)
        full_explicit = compute_objective(item, rubric, evaluator, judge, config, pair_budget=15)
        over_budget = compute_objective(item, rubric, evaluator, judge, config, pair_budget=16)
        sampled_once = compute_objective(item, rubric, evaluator, judge, config, pair_budget=5)
        sampled_twice = compute_objective(item, rubric, evaluator, judge, config, pair_budget=5)

        self.assertEqual(full, full_explicit)
        self.assertEqual(full, over_budget)
        self.assertEqual(sampled_once, sampled_twice)
        self.assertNotEqual(full, sampled_once)

    def test_tie_tolerance_within_boundary_treated_as_tie(self) -> None:
        item = _build_two_candidate_item("same", "same")
        rubric = _build_test_rubric()
        tolerance = 1e-8
        scored = [
            CandidateEvaluation("a", score=0.9, variance=0.0, majority_grades={}, vote_scores=[]),
            CandidateEvaluation(
                "b", score=0.9 - (tolerance / 2.0), variance=0.0, majority_grades={}, vote_scores=[]
            ),
        ]
        result = compute_objective(
            item,
            rubric,
            _StaticEvaluator(scored),
            _StaticJudge(pref=1),
            ObjectiveConfig(tail_fraction=1.0, tie_tolerance=tolerance, lambda_var=0.0, mu_diverse=0.0),
        )
        self.assertEqual(result.valid_pairs, 1)
        self.assertEqual(result.tail_acc, 0.0)
        self.assertEqual(result.total, 0.0)

    def test_empty_valid_pairs_falls_back_to_half(self) -> None:
        item = _build_two_candidate_item("same", "diverse")
        rubric = _build_test_rubric()
        scored = [
            CandidateEvaluation("a", score=0.9, variance=0.0, majority_grades={}, vote_scores=[]),
            CandidateEvaluation("b", score=0.2, variance=0.0, majority_grades={}, vote_scores=[]),
        ]
        result = compute_objective(
            item,
            rubric,
            _StaticEvaluator(scored),
            _AlwaysTieJudge(),
            ObjectiveConfig(tail_fraction=1.0, lambda_var=0.0, mu_diverse=0.0),
        )
        self.assertEqual(result.valid_pairs, 0)
        self.assertEqual(result.tail_acc, 0.5)
        self.assertEqual(result.total, 0.5)

    def test_diverse_fallback_equals_tail_acc_when_no_diverse_pairs(self) -> None:
        item = _build_two_candidate_item("same", "same")
        rubric = _build_test_rubric()
        scored = [
            CandidateEvaluation("a", score=0.9, variance=0.0, majority_grades={}, vote_scores=[]),
            CandidateEvaluation("b", score=0.2, variance=0.0, majority_grades={}, vote_scores=[]),
        ]
        result = compute_objective(
            item,
            rubric,
            _StaticEvaluator(scored),
            _StaticJudge(pref=1),
            ObjectiveConfig(tail_fraction=1.0, lambda_var=0.0, mu_diverse=0.4),
        )
        self.assertEqual(result.valid_pairs, 1)
        self.assertEqual(result.diverse_pairs, 0)
        self.assertEqual(result.tail_acc, 1.0)
        self.assertEqual(result.diverse_tail_acc, result.tail_acc)
        self.assertAlmostEqual(result.total, 1.4, places=8)


def _build_test_rubric() -> Rubric:
    return Rubric(
        rubric_id="test_rubric",
        criteria=[Criterion("c1", "criterion", weight=1.0)],
    )


def _build_two_candidate_item(source_a: str, source_b: str) -> PromptExample:
    return PromptExample(
        prompt_id="p_test",
        prompt="test prompt",
        candidates=[
            ResponseCandidate("a", "left", source=source_a),
            ResponseCandidate("b", "right", source=source_b),
        ],
    )


if __name__ == "__main__":
    unittest.main()

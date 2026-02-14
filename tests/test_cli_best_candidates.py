from __future__ import annotations

import random
import unittest

from autosr.cli import compute_best_candidates
from autosr.config import RuntimeConfig, VerifierConfig
from autosr.factory import ComponentFactory
from autosr.io_utils import load_dataset
from autosr.mock_components import HeuristicRubricInitializer, HeuristicVerifier


class TestCliBestCandidates(unittest.TestCase):
    def test_compute_best_candidates_returns_valid_candidate_ids(self) -> None:
        prompts = load_dataset("examples/demo_dataset.json")[:2]
        initializer = HeuristicRubricInitializer()
        rubrics = {
            item.prompt_id: initializer.initialize(item, rng=random.Random(7))
            for item in prompts
        }
        
        # Create factory and compute best candidates using new API
        config = RuntimeConfig(verifier=VerifierConfig(noise=0.0))
        factory = ComponentFactory(config)
        
        best_candidates, candidate_scores = compute_best_candidates(
            prompts=prompts,
            rubrics=rubrics,
            factory=factory,
            seed=7,
        )

        self.assertEqual(set(best_candidates.keys()), set(rubrics.keys()))
        prompt_map = {item.prompt_id: item for item in prompts}
        for prompt_id, candidate_id in best_candidates.items():
            candidate_ids = {candidate.candidate_id for candidate in prompt_map[prompt_id].candidates}
            self.assertIn(candidate_id, candidate_ids)

        # Verify candidate_scores structure and content
        self.assertEqual(set(candidate_scores.keys()), set(rubrics.keys()))
        for prompt_id, scores in candidate_scores.items():
            candidate_ids = {candidate.candidate_id for candidate in prompt_map[prompt_id].candidates}
            self.assertEqual(set(scores.keys()), candidate_ids)
            for candidate_id, score in scores.items():
                self.assertIsInstance(score, float)
            # Verify best_candidate matches highest score
            best_candidate_id = best_candidates[prompt_id]
            self.assertEqual(
                max(scores, key=scores.get),  # type: ignore[arg-type]
                best_candidate_id,
            )

    def test_compute_best_candidates_uses_extraction_aware_verifier(self) -> None:
        prompts = load_dataset("examples/demo_dataset.json")[:1]
        initializer = HeuristicRubricInitializer()
        rubrics = {
            item.prompt_id: initializer.initialize(item, rng=random.Random(7))
            for item in prompts
        }

        class _FactorySpy:
            def __init__(self) -> None:
                self.create_verifier_called = False
                self.create_verifier_with_extraction_called = False
                self.received_prompts = None

            def create_verifier(self) -> HeuristicVerifier:
                self.create_verifier_called = True
                raise AssertionError("compute_best_candidates should not call create_verifier")

            def create_verifier_with_extraction(self, prompts_arg):  # noqa: ANN001 - test spy
                self.create_verifier_with_extraction_called = True
                self.received_prompts = prompts_arg
                return HeuristicVerifier(noise=0.0)

        factory = _FactorySpy()

        best_candidates, _ = compute_best_candidates(
            prompts=prompts,
            rubrics=rubrics,
            factory=factory,  # type: ignore[arg-type]
            seed=7,
        )

        self.assertFalse(factory.create_verifier_called)
        self.assertTrue(factory.create_verifier_with_extraction_called)
        self.assertIs(factory.received_prompts, prompts)
        self.assertEqual(set(best_candidates.keys()), set(rubrics.keys()))


if __name__ == "__main__":
    unittest.main()

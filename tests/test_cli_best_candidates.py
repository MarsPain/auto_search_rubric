from __future__ import annotations

import random
import unittest

from autosr.cli import compute_best_candidates
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
        best_candidates = compute_best_candidates(
            prompts=prompts,
            rubrics=rubrics,
            verifier=HeuristicVerifier(noise=0.0),
            seed=7,
        )

        self.assertEqual(set(best_candidates.keys()), set(rubrics.keys()))
        prompt_map = {item.prompt_id: item for item in prompts}
        for prompt_id, candidate_id in best_candidates.items():
            candidate_ids = {candidate.candidate_id for candidate in prompt_map[prompt_id].candidates}
            self.assertIn(candidate_id, candidate_ids)


if __name__ == "__main__":
    unittest.main()

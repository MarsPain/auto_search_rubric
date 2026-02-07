from __future__ import annotations

import unittest

from autosr.io_utils import load_dataset
from autosr.mock_components import (
    HeuristicPreferenceJudge,
    HeuristicRubricInitializer,
    HeuristicVerifier,
    TemplateProposer,
)
from autosr.search import (
    EvolutionaryConfig,
    EvolutionaryRTDSearcher,
    IterativeConfig,
    IterativeRTDSearcher,
)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = load_dataset("examples/demo_dataset.json")
        cls.proposer = TemplateProposer()
        cls.verifier = HeuristicVerifier(noise=0.0)
        cls.judge = HeuristicPreferenceJudge()
        cls.initializer = HeuristicRubricInitializer()

    def test_iterative_runs(self) -> None:
        searcher = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=IterativeConfig(iterations=3, seed=42),
        )
        result = searcher.search(self.prompts[:2])
        self.assertEqual(len(result.best_rubrics), 2)
        self.assertEqual(len(result.best_scores), 2)
        for prompt_id, history in result.history.items():
            self.assertEqual(len(history), 3)
            self.assertIn(prompt_id, result.best_scores)

    def test_iterative_deterministic_outputs(self) -> None:
        config = IterativeConfig(iterations=3, seed=42, accept_only_if_improve=True)
        first = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=config,
        ).search(self.prompts[:2])
        second = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=IterativeConfig(iterations=3, seed=42, accept_only_if_improve=True),
        ).search(self.prompts[:2])

        self.assertEqual(first.best_scores, second.best_scores)
        self.assertEqual(first.history, second.history)
        self.assertEqual(
            {k: v.rubric_id for k, v in first.best_rubrics.items()},
            {k: v.rubric_id for k, v in second.best_rubrics.items()},
        )

    def test_accept_only_if_improve_yields_non_decreasing_history(self) -> None:
        searcher = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=IterativeConfig(iterations=6, seed=17, accept_only_if_improve=True),
        )
        result = searcher.search(self.prompts[:2])
        for history in result.history.values():
            for idx in range(1, len(history)):
                self.assertGreaterEqual(history[idx], history[idx - 1])

    def test_evolutionary_runs(self) -> None:
        searcher = EvolutionaryRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(
                generations=3,
                population_size=4,
                mutations_per_round=3,
                batch_size=2,
                seed=42,
                stagnation_generations=3,
            ),
        )
        result = searcher.search(self.prompts[:2])
        self.assertEqual(len(result.best_rubrics), 2)
        self.assertEqual(len(result.best_scores), 2)
        for score in result.best_scores.values():
            self.assertTrue(score == score)  # not NaN

    def test_evolutionary_deterministic_outputs(self) -> None:
        cfg_kwargs = dict(
            generations=3,
            population_size=4,
            mutations_per_round=3,
            batch_size=2,
            seed=42,
            stagnation_generations=3,
        )
        first = EvolutionaryRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(**cfg_kwargs),
        ).search(self.prompts[:2])
        second = EvolutionaryRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(**cfg_kwargs),
        ).search(self.prompts[:2])

        self.assertEqual(first.best_scores, second.best_scores)
        self.assertEqual(first.history, second.history)
        self.assertEqual(
            {k: v.rubric_id for k, v in first.best_rubrics.items()},
            {k: v.rubric_id for k, v in second.best_rubrics.items()},
        )

    def test_evolutionary_emits_progress_logs(self) -> None:
        searcher = EvolutionaryRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(
                generations=2,
                population_size=4,
                mutations_per_round=2,
                batch_size=1,
                seed=42,
                stagnation_generations=2,
            ),
        )

        with self.assertLogs("autosr.search", level="INFO") as captured:
            searcher.search(self.prompts[:1])

        logs = "\n".join(captured.output)
        self.assertIn("generation=1/2", logs)
        self.assertIn("best_score", logs)


if __name__ == "__main__":
    unittest.main()

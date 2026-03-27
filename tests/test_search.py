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

    def test_iterative_reports_margin_improvement_diagnostics(self) -> None:
        prompts = self.prompts[:2]
        searcher = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=IterativeConfig(iterations=3, seed=42),
        )
        result = searcher.search(prompts)

        stats = result.diagnostics["margin_improvement"]
        self.assertEqual(stats["global"]["total_prompts"], len(prompts))
        self.assertEqual(set(stats["per_prompt"].keys()), {item.prompt_id for item in prompts})

        improved_count = sum(
            1 for item in stats["per_prompt"].values() if item["improved"]
        )
        self.assertEqual(stats["global"]["improved_prompts"], improved_count)
        for item in stats["per_prompt"].values():
            self.assertAlmostEqual(
                item["margin_delta"],
                item["final_margin"] - item["initial_margin"],
            )

    def test_iterative_emits_margin_progress_logs(self) -> None:
        searcher = IterativeRTDSearcher(
            self.proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=IterativeConfig(iterations=2, seed=42),
        )
        with self.assertLogs("autosr.search", level="INFO") as captured:
            searcher.search(self.prompts[:2])
        logs = "\n".join(captured.output)
        self.assertIn("margin-progress processed=", logs)

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
        self.assertIn("margin_improved_prompts", logs)

    def test_evolutionary_reports_margin_improvement_diagnostics(self) -> None:
        prompts = self.prompts[:2]
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
                stop_when_distinguished=False,
            ),
        )
        result = searcher.search(prompts)

        stats = result.diagnostics["margin_improvement"]
        self.assertEqual(stats["global"]["total_prompts"], len(prompts))
        self.assertEqual(set(stats["per_prompt"].keys()), {item.prompt_id for item in prompts})
        improved_count = sum(
            1 for item in stats["per_prompt"].values() if item["improved"]
        )
        self.assertEqual(stats["global"]["improved_prompts"], improved_count)

    def test_evolutionary_prompt_local_scope_decouples_from_batch_size(self) -> None:
        class CountingProposer(TemplateProposer):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def propose(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                self.call_count += 1
                return super().propose(*args, **kwargs)

        prompts = self.prompts[:2]
        shared_kwargs = dict(
            generations=2,
            population_size=4,
            mutations_per_round=2,
            batch_size=1,
            seed=42,
            stagnation_generations=8,
            stop_when_distinguished=False,
        )

        global_proposer = CountingProposer()
        EvolutionaryRTDSearcher(
            global_proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(
                **shared_kwargs,
                iteration_scope="global_batch",
            ),
        ).search(prompts)

        local_proposer = CountingProposer()
        EvolutionaryRTDSearcher(
            local_proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=EvolutionaryConfig(
                **shared_kwargs,
                iteration_scope="prompt_local",
            ),
        ).search(prompts)

        expected_extra_calls = len(prompts) * shared_kwargs["generations"] * shared_kwargs["mutations_per_round"]
        expected_global_calls = shared_kwargs["generations"] * shared_kwargs["mutations_per_round"]
        self.assertEqual(
            local_proposer.call_count - global_proposer.call_count,
            expected_extra_calls - expected_global_calls,
        )
        self.assertGreater(local_proposer.call_count, global_proposer.call_count)

    def test_evolutionary_prompt_local_emits_checkpoint_per_prompt(self) -> None:
        prompts = self.prompts[:2]
        snapshots: list[set[str]] = []

        def checkpoint_callback(  # type: ignore[no-untyped-def]
            best_rubrics,
            best_scores,
            history,
        ) -> None:
            self.assertEqual(set(best_rubrics.keys()), set(best_scores.keys()))
            self.assertEqual(set(best_rubrics.keys()), set(history.keys()))
            snapshots.append(set(best_rubrics.keys()))

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
                stagnation_generations=3,
                iteration_scope="prompt_local",
                stop_when_distinguished=False,
            ),
            checkpoint_callback=checkpoint_callback,
        )

        searcher.search(prompts)
        self.assertEqual(len(snapshots), len(prompts))
        self.assertEqual(len(snapshots[0]), 1)
        self.assertEqual(len(snapshots[1]), 2)
        self.assertTrue(snapshots[0].issubset(snapshots[1]))

    def test_evolutionary_uses_multiple_parents_for_mutation(self) -> None:
        class ParentTrackingProposer(TemplateProposer):
            def __init__(self) -> None:
                super().__init__()
                self.parent_rubric_ids: list[str] = []

            def propose(self, prompt, left, right, rubric, *, mode, rng):  # type: ignore[no-untyped-def]
                self.parent_rubric_ids.append(rubric.rubric_id)
                return super().propose(prompt, left, right, rubric, mode=mode, rng=rng)

        proposer = ParentTrackingProposer()
        config = EvolutionaryConfig(
            generations=1,
            population_size=4,
            mutations_per_round=4,
            mutation_parent_count=3,
            batch_size=1,
            seed=42,
            stagnation_generations=3,
            iteration_scope="prompt_local",
            stop_when_distinguished=False,
        )

        EvolutionaryRTDSearcher(
            proposer,
            self.verifier,
            self.judge,
            self.initializer,
            config=config,
        ).search(self.prompts[:1])

        mutation_parent_ids = proposer.parent_rubric_ids[-config.mutations_per_round :]
        self.assertGreaterEqual(len(set(mutation_parent_ids)), 2)


if __name__ == "__main__":
    unittest.main()

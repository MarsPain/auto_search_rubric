from __future__ import annotations

import importlib
import inspect
import unittest

from autosr import factory as factory_module
from autosr import interfaces
from autosr.config import ObjectiveFunctionConfig, RuntimeConfig, SearchAlgorithmConfig
from autosr.factory import ComponentFactory
from autosr.harness import SearchSession
from autosr.interfaces import SteppableSearcher
from autosr.models import PromptExample, ResponseCandidate
from autosr.search import evolutionary as evolutionary_module
from autosr.search import EvolutionaryRTDSearcher, IterativeRTDSearcher
from autosr.types import EvolutionIterationScope


def _build_prompt() -> PromptExample:
    return PromptExample(
        prompt_id="p-arch",
        prompt="Summarize this conversation.",
        candidates=[
            ResponseCandidate(candidate_id="a", text="First response", source="s1"),
            ResponseCandidate(candidate_id="b", text="Second response", source="s2"),
        ],
    )


class TestArchitectureRefactor(unittest.TestCase):
    def test_factory_passes_objective_config_without_copy(self) -> None:
        config = RuntimeConfig(
            search=SearchAlgorithmConfig(mode="iterative"),
            objective=ObjectiveFunctionConfig(
                tail_fraction=0.6,
                lambda_var=0.3,
                mu_diverse=0.1,
                pair_budget_small=4,
                pair_budget_medium=10,
                pair_budget_full=0,
                tie_tolerance=1e-6,
            ),
        )
        factory = ComponentFactory(config)

        searcher = factory.create_searcher([_build_prompt()])

        self.assertIs(searcher.config.objective, config.objective)

    def test_split_architecture_modules_are_importable(self) -> None:
        modules = [
            "autosr.search.use_cases",
            "autosr.search.strategies",
            "autosr.llm_components.use_cases",
            "autosr.llm_components.parsers",
            "autosr.content_extraction.use_cases",
            "autosr.content_extraction.strategies",
            "autosr.run_records.use_cases",
            "autosr.rm.use_cases",
            "autosr.rm.data_models",
        ]
        for name in modules:
            with self.subTest(module=name):
                imported = importlib.import_module(name)
                self.assertIsNotNone(imported)

    def test_factory_accepts_checkpoint_callback_and_wires_evolutionary_searcher(self) -> None:
        config = RuntimeConfig(search=SearchAlgorithmConfig(mode="evolutionary"))
        factory = ComponentFactory(config)
        callback = lambda _r, _s, _h: None  # noqa: E731

        searcher = factory.create_searcher([_build_prompt()], checkpoint_callback=callback)

        self.assertIsInstance(searcher, EvolutionaryRTDSearcher)
        self.assertIs(searcher._checkpoint_callback, callback)

    def test_factory_accepts_checkpoint_callback_for_iterative_without_wiring(self) -> None:
        config = RuntimeConfig(search=SearchAlgorithmConfig(mode="iterative"))
        factory = ComponentFactory(config)
        callback = lambda _r, _s, _h: None  # noqa: E731

        searcher = factory.create_searcher([_build_prompt()], checkpoint_callback=callback)

        self.assertIsInstance(searcher, IterativeRTDSearcher)

    def test_checkpoint_callback_type_is_defined_once_in_interfaces(self) -> None:
        self.assertTrue(hasattr(interfaces, "CheckpointCallback"))
        self.assertIs(factory_module.CheckpointCallback, interfaces.CheckpointCallback)
        self.assertIs(evolutionary_module.CheckpointCallback, interfaces.CheckpointCallback)

    def test_evolutionary_searcher_exposes_steppable_protocol(self) -> None:
        config = RuntimeConfig(
            search=SearchAlgorithmConfig(
                mode="evolutionary",
                iteration_scope=EvolutionIterationScope.GLOBAL_BATCH,
            )
        )
        searcher = ComponentFactory(config).create_searcher([_build_prompt()])

        self.assertIsInstance(searcher, EvolutionaryRTDSearcher)
        self.assertIsInstance(searcher, SteppableSearcher)

    def test_search_session_does_not_call_evolutionary_private_step_methods(self) -> None:
        source = inspect.getsource(SearchSession)
        forbidden = [
            "._init_global_state",
            "._score_population",
            "._log_generation_progress",
            "._update_generation_bests",
            "._handle_stagnation",
            "._select_hard_prompts",
            "._evolve_selected_prompts",
            "._finalize_best_from_population",
            "._collect_margin_improvement",
        ]

        for private_call in forbidden:
            with self.subTest(private_call=private_call):
                self.assertNotIn(private_call, source)


if __name__ == "__main__":
    unittest.main()

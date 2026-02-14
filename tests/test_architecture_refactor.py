from __future__ import annotations

import importlib
import unittest

from autosr.config import ObjectiveFunctionConfig, RuntimeConfig, SearchAlgorithmConfig
from autosr.factory import ComponentFactory
from autosr.models import PromptExample, ResponseCandidate


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
        ]
        for name in modules:
            with self.subTest(module=name):
                imported = importlib.import_module(name)
                self.assertIsNotNone(imported)


if __name__ == "__main__":
    unittest.main()

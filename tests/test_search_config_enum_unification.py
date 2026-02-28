from __future__ import annotations

import unittest

from autosr.cli import build_parser, build_runtime_config
from autosr.config import SearchAlgorithmConfig
from autosr.search import AdaptiveMutationSchedule, SelectionStrategy


class TestSearchConfigEnumUnification(unittest.TestCase):
    def test_search_algorithm_config_accepts_selection_strategy_enum(self) -> None:
        config = SearchAlgorithmConfig(selection_strategy=SelectionStrategy.TOP_K)
        self.assertIs(config.selection_strategy, SelectionStrategy.TOP_K)
        self.assertIs(
            config.to_evolutionary_kwargs()["selection_strategy"],
            SelectionStrategy.TOP_K,
        )

    def test_search_algorithm_config_accepts_adaptive_mutation_enum(self) -> None:
        config = SearchAlgorithmConfig(
            adaptive_mutation=AdaptiveMutationSchedule.DIVERSITY_DRIVEN
        )
        self.assertIs(
            config.adaptive_mutation,
            AdaptiveMutationSchedule.DIVERSITY_DRIVEN,
        )
        self.assertIs(
            config.to_evolutionary_kwargs()["adaptive_mutation"],
            AdaptiveMutationSchedule.DIVERSITY_DRIVEN,
        )

    def test_cli_string_values_remain_compatible_and_normalized_to_enums(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "d.json",
                "--output",
                "o.json",
                "--selection-strategy",
                "top_k",
                "--adaptive-mutation",
                "diversity_driven",
            ]
        )

        runtime = build_runtime_config(args)
        self.assertIs(runtime.search.selection_strategy, SelectionStrategy.TOP_K)
        self.assertIs(
            runtime.search.adaptive_mutation,
            AdaptiveMutationSchedule.DIVERSITY_DRIVEN,
        )


if __name__ == "__main__":
    unittest.main()

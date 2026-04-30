"""Test consistency between RuntimeConfig and algorithm-level configs.

This module ensures ``SearchAlgorithmConfig`` (CLI/runtime layer) and
``EvolutionaryConfig`` (algorithm layer) do not silently drift in defaults
or validation rules.
"""

from __future__ import annotations

import unittest

from reward_harness.config import SearchAlgorithmConfig
from reward_harness.search.config import EvolutionaryConfig


class TestEvolutionaryConfigConsistency(unittest.TestCase):
    """Verify SearchAlgorithmConfig and EvolutionaryConfig stay aligned."""

    def test_default_generations_match(self) -> None:
        sc = SearchAlgorithmConfig()
        ec = EvolutionaryConfig()
        self.assertEqual(
            sc.generations,
            ec.generations,
            "Default 'generations' must match between SearchAlgorithmConfig "
            "and EvolutionaryConfig to avoid CLI vs algorithm drift",
        )

    def test_default_batch_size_match(self) -> None:
        sc = SearchAlgorithmConfig()
        ec = EvolutionaryConfig()
        self.assertEqual(
            sc.batch_size,
            ec.batch_size,
            "Default 'batch_size' must match between SearchAlgorithmConfig "
            "and EvolutionaryConfig",
        )

    def test_shared_numeric_defaults_match(self) -> None:
        """All overlapping numeric defaults must be identical."""
        sc = SearchAlgorithmConfig()
        ec = EvolutionaryConfig()

        shared_fields = [
            "population_size",
            "mutations_per_round",
            "mutation_parent_count",
            "survival_fraction",
            "elitism_count",
            "stagnation_generations",
            "tournament_size",
            "tournament_p",
            "top_k_ratio",
            "diversity_weight",
            "mutation_window_size",
            "min_mutation_weight",
            "exploration_phase_ratio",
            "diversity_threshold",
        ]

        for field in shared_fields:
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(sc, field),
                    getattr(ec, field),
                    f"Default '{field}' must match between SearchAlgorithmConfig "
                    "and EvolutionaryConfig",
                )

    def test_roundtrip_via_to_evolutionary_kwargs(self) -> None:
        """SearchAlgorithmConfig must produce a valid EvolutionaryConfig."""
        sc = SearchAlgorithmConfig()
        kwargs = sc.to_evolutionary_kwargs()
        ec = EvolutionaryConfig(**kwargs)
        self.assertEqual(sc.generations, ec.generations)
        self.assertEqual(sc.population_size, ec.population_size)

    def test_validation_rejects_common_invalid_values(self) -> None:
        """Both classes must reject the same invalid inputs."""
        invalid_cases = [
            {"population_size": 1},
            {"generations": 0},
            {"mutations_per_round": 0},
            {"mutation_parent_count": 0},
            {"mutation_parent_count": 10, "population_size": 5},
            {"survival_fraction": 0},
            {"survival_fraction": 1.5},
            {"elitism_count": 0},
            {"tournament_size": 1},
            {"tournament_p": 0},
            {"tournament_p": 1.5},
            {"top_k_ratio": 0},
            {"top_k_ratio": 1.5},
            {"diversity_weight": -0.1},
            {"diversity_weight": 1.1},
            {"mutation_window_size": 0},
            {"min_mutation_weight": 0},
            {"min_mutation_weight": 1.5},
            {"exploration_phase_ratio": 0},
            {"exploration_phase_ratio": 1.5},
            {"diversity_threshold": -0.1},
            {"diversity_threshold": 1.1},
            {"distinguish_margin": -1.0},
        ]

        for case in invalid_cases:
            with self.subTest(case=case):
                # SearchAlgorithmConfig must reject
                with self.assertRaises(ValueError):
                    SearchAlgorithmConfig(**case)

                # EvolutionaryConfig must also reject (note: case may contain
                # fields not in EvolutionaryConfig, so we filter)
                ec_fields = {f for f in case if hasattr(EvolutionaryConfig, f)}
                if ec_fields:
                    with self.assertRaises(ValueError):
                        EvolutionaryConfig(**{k: case[k] for k in ec_fields})


if __name__ == "__main__":
    unittest.main()

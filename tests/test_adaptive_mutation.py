from __future__ import annotations

import random
import unittest

from autosr.data_models import Criterion, Rubric
from autosr.search.adaptive_mutation import (
    AdaptiveMutationSelector,
    MutationHistory,
    compute_population_diversity,
)
from autosr.search.config import EvolutionaryConfig
from autosr.types import AdaptiveMutationSchedule, MutationMode


class FixedRandom(random.Random):
    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


def _rubric(rubric_id: str, text: str) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criteria=[Criterion(criterion_id="c1", text=text)],
    )


class AdaptiveMutationTest(unittest.TestCase):
    def test_mutation_history_roundtrips_serialized_state(self) -> None:
        history = MutationHistory()
        history.record(MutationMode.DECOMPOSE, True, 0.3)
        history.record(MutationMode.RAISE_BAR, False, -0.1)
        history.increment_generation()

        restored = MutationHistory.from_state(history.to_state())

        self.assertEqual(1, restored.generation_count)
        self.assertEqual(1.0, restored.get_success_rate(MutationMode.DECOMPOSE, 10))
        self.assertEqual(0.0, restored.get_success_rate(MutationMode.RAISE_BAR, 10))
        self.assertAlmostEqual(
            0.3,
            restored.get_average_improvement(MutationMode.DECOMPOSE, 10),
        )

    def test_selector_restores_checkpoint_state(self) -> None:
        config = EvolutionaryConfig(
            adaptive_mutation=AdaptiveMutationSchedule.SUCCESS_FEEDBACK,
            mutation_window_size=3,
        )
        selector = AdaptiveMutationSelector(config, random.Random(7))
        selector.record_outcome(MutationMode.FACTUAL_FOCUS, True, 0.2)
        selector.next_generation()

        restored = AdaptiveMutationSelector(config, random.Random(7))
        restored.set_state(selector.get_state())

        self.assertEqual(selector.get_state()["history"], restored.get_state()["history"])
        self.assertEqual(
            selector.get_diagnostics()["success_rates"],
            restored.get_diagnostics()["success_rates"],
        )

    def test_success_feedback_favors_modes_with_better_recent_outcomes(self) -> None:
        config = EvolutionaryConfig(
            adaptive_mutation=AdaptiveMutationSchedule.SUCCESS_FEEDBACK,
            mutation_window_size=5,
            min_mutation_weight=0.01,
        )
        selector = AdaptiveMutationSelector(config, FixedRandom(0.2))
        for _ in range(5):
            selector.record_outcome(MutationMode.RAISE_BAR, False, -0.2)
            selector.record_outcome(MutationMode.DECOMPOSE, True, 0.4)

        selected = selector.select_mode()

        self.assertEqual(MutationMode.DECOMPOSE, selected)

    def test_exploration_decay_moves_from_exploration_to_exploitation(self) -> None:
        config = EvolutionaryConfig(
            adaptive_mutation=AdaptiveMutationSchedule.EXPLORATION_DECAY,
            generations=10,
            exploration_phase_ratio=0.3,
        )
        early = AdaptiveMutationSelector(config, FixedRandom(0.1))
        late = AdaptiveMutationSelector(config, FixedRandom(0.1))
        for _ in range(4):
            late.next_generation()

        self.assertEqual(MutationMode.DECOMPOSE, early.select_mode())
        self.assertEqual(MutationMode.RAISE_BAR, late.select_mode())

    def test_diversity_driven_handles_zero_threshold(self) -> None:
        selector = AdaptiveMutationSelector(
            EvolutionaryConfig(
                adaptive_mutation=AdaptiveMutationSchedule.DIVERSITY_DRIVEN,
                diversity_threshold=0.0,
            ),
            FixedRandom(0.1),
        )

        selected = selector.select_mode(diversity_score=0.0)

        self.assertIsInstance(selected, MutationMode)

    def test_population_diversity_handles_degenerate_and_distinct_populations(self) -> None:
        single = [_rubric("r1", "Reward direct answers.")]
        identical = [
            _rubric("r1", "Reward direct answers."),
            _rubric("r1", "Reward direct answers."),
        ]
        distinct = [
            _rubric("r1", "Reward direct answers."),
            _rubric("r2", "Reward evidence-backed, concise answers."),
        ]

        self.assertEqual(0.0, compute_population_diversity(single))
        self.assertEqual(0.0, compute_population_diversity(identical))
        self.assertGreater(
            compute_population_diversity(distinct, rng=random.Random(7)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

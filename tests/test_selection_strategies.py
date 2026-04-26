from __future__ import annotations

import random
import unittest

from autosr.data_models import Criterion, Rubric
from autosr.evaluator import ObjectiveBreakdown
from autosr.search.config import EvolutionaryConfig
from autosr.search.selection_strategies import (
    select_parents_rank,
    select_parents_top_k_diverse,
    select_parents_tournament,
)


def _rubric(rubric_id: str, text: str) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criteria=[
            Criterion(
                criterion_id="c1",
                text=text,
            )
        ],
    )


def _score(total: float) -> ObjectiveBreakdown:
    return ObjectiveBreakdown(
        total=total,
        tail_acc=total,
        tail_var=0.0,
        diverse_tail_acc=total,
        valid_pairs=1,
        diverse_pairs=1,
        top_margin=total,
        signed_top_margin=total,
    )


class SelectionStrategiesTest(unittest.TestCase):
    def test_rank_selects_highest_scored_parents_in_order(self) -> None:
        high = _rubric("high", "Rewards direct answers.")
        mid = _rubric("mid", "Rewards concise answers.")
        low = _rubric("low", "Rewards detailed answers.")

        selected = select_parents_rank(
            [(high, _score(0.9)), (mid, _score(0.5)), (low, _score(0.1))],
            num_parents=2,
            rng=random.Random(7),
            config=EvolutionaryConfig(),
        )

        self.assertEqual([high, mid], selected)

    def test_tournament_deduplicates_semantically_equal_rubrics(self) -> None:
        original = _rubric("same", "Rewards direct answers.")
        duplicate = Rubric.from_dict(original.to_dict())

        selected = select_parents_tournament(
            [(original, _score(0.9)), (duplicate, _score(0.8))],
            num_parents=2,
            rng=random.Random(7),
            config=EvolutionaryConfig(tournament_size=2, tournament_p=1.0),
        )

        self.assertEqual(1, len(selected))
        self.assertEqual(original.fingerprint(), selected[0].fingerprint())

    def test_top_k_diverse_returns_empty_for_empty_population(self) -> None:
        selected = select_parents_top_k_diverse(
            [],
            num_parents=2,
            rng=random.Random(7),
            config=EvolutionaryConfig(),
        )

        self.assertEqual([], selected)


if __name__ == "__main__":
    unittest.main()

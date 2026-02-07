from __future__ import annotations

import unittest

from autosr.models import Criterion, GradingProtocol, Rubric


class TestModels(unittest.TestCase):
    def test_weighted_scoring(self) -> None:
        rubric = Rubric(
            rubric_id="r1",
            criteria=[
                Criterion("c1", "A", weight=2.0),
                Criterion("c2", "B", weight=1.0),
            ],
            grading_protocol=GradingProtocol(allow_na=False, num_votes=1),
        )
        score = rubric.score_from_grades({"c1": 1, "c2": 0})
        self.assertAlmostEqual(score, 2 / 3, places=6)

    def test_na_excluded_when_allowed(self) -> None:
        rubric = Rubric(
            rubric_id="r2",
            criteria=[Criterion("c1", "A"), Criterion("c2", "B")],
            grading_protocol=GradingProtocol(allow_na=True, num_votes=1),
        )
        score = rubric.score_from_grades({"c1": 1, "c2": None})
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_fingerprint_stable(self) -> None:
        rubric = Rubric(
            rubric_id="r3",
            criteria=[Criterion("c1", "A"), Criterion("c2", "B")],
        )
        self.assertEqual(rubric.fingerprint(), rubric.fingerprint())


if __name__ == "__main__":
    unittest.main()


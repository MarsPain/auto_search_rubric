from __future__ import annotations

import unittest

from autosr import data_models, models
from autosr.data_models import Criterion, GradingProtocol, PromptExample, ResponseCandidate, Rubric


class TestDataModelsCompat(unittest.TestCase):
    def test_new_data_models_module_is_importable(self) -> None:
        rubric = Rubric(
            rubric_id="r-data-models",
            criteria=[Criterion(criterion_id="c1", text="Quality")],
            grading_protocol=GradingProtocol(num_votes=1),
        )
        self.assertEqual(rubric.rubric_id, "r-data-models")

    def test_models_module_re_exports_data_models(self) -> None:
        self.assertIs(models.Criterion, data_models.Criterion)
        self.assertIs(models.GradingProtocol, data_models.GradingProtocol)
        self.assertIs(models.Rubric, data_models.Rubric)
        self.assertIs(models.ResponseCandidate, data_models.ResponseCandidate)
        self.assertIs(models.PromptExample, data_models.PromptExample)

    def test_new_and_old_import_paths_work_together(self) -> None:
        candidate = ResponseCandidate(candidate_id="c", text="hello")
        example = PromptExample(prompt_id="p", prompt="Say hello", candidates=[candidate, candidate])
        self.assertEqual(example.prompt_id, "p")


if __name__ == "__main__":
    unittest.main()

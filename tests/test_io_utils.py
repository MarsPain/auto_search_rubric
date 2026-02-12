from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from autosr.io_utils import load_dataset, load_initial_rubrics, save_rubrics
from autosr.mock_components import HeuristicRubricInitializer


class TestLoadDataset(unittest.TestCase):
    """Tests for load_dataset function."""

    def test_load_dataset_basic(self) -> None:
        """Test basic dataset loading."""
        examples = load_dataset("examples/demo_dataset.json")
        self.assertIsInstance(examples, list)
        self.assertGreater(len(examples), 0)
        # Verify structure
        for example in examples:
            self.assertTrue(hasattr(example, 'prompt_id'))
            self.assertTrue(hasattr(example, 'prompt'))
            self.assertTrue(hasattr(example, 'candidates'))
            self.assertGreaterEqual(len(example.candidates), 2)

    def test_load_dataset_preserves_original_prompt(self) -> None:
        """Test that original prompt text is preserved unchanged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dataset = {
                "prompts": [
                    {
                        "prompt_id": "p1",
                        "prompt": "Full prompt text with <content>tags</content>",
                        "candidates": [
                            {"candidate_id": "c1", "text": "Response 1", "source": "test"},
                            {"candidate_id": "c2", "text": "Response 2", "source": "test"},
                        ],
                    }
                ]
            }
            json.dump(dataset, f)
            f.flush()

            examples = load_dataset(f.name)
            # Original prompt should be preserved unchanged
            self.assertEqual(examples[0].prompt, "Full prompt text with <content>tags</content>")

            Path(f.name).unlink()


class TestLoadInitialRubrics(unittest.TestCase):
    """Tests for load_initial_rubrics function."""

    def test_load_best_rubrics_format(self) -> None:
        """Test loading from best_rubrics output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "best_rubrics": [
                    {
                        "prompt_id": "p1",
                        "rubric": {
                            "rubric_id": "r1",
                            "criteria": [
                                {"criterion_id": "c1", "text": "Test criterion", "weight": 1.0}
                            ],
                            "grading_protocol": {"output_format": "json", "num_votes": 3},
                        },
                        "score": 0.85,
                    }
                ]
            }
            json.dump(data, f)
            f.flush()

            rubrics = load_initial_rubrics(f.name)
            self.assertIn("p1", rubrics)
            self.assertEqual(rubrics["p1"].rubric_id, "r1")

            Path(f.name).unlink()

    def test_load_rubrics_array_format(self) -> None:
        """Test loading from rubrics array format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "rubrics": [
                    {
                        "prompt_id": "p1",
                        "rubric": {
                            "rubric_id": "r1",
                            "criteria": [
                                {"criterion_id": "c1", "text": "Test criterion", "weight": 1.0}
                            ],
                        },
                    }
                ]
            }
            json.dump(data, f)
            f.flush()

            rubrics = load_initial_rubrics(f.name)
            self.assertIn("p1", rubrics)

            Path(f.name).unlink()

    def test_load_direct_format(self) -> None:
        """Test loading from direct format {prompt_id: rubric_dict}."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "p1": {
                    "rubric_id": "r1",
                    "criteria": [
                        {"criterion_id": "c1", "text": "Test criterion", "weight": 1.0}
                    ],
                },
                "p2": {
                    "rubric_id": "r2",
                    "criteria": [
                        {"criterion_id": "c2", "text": "Another criterion", "weight": 1.0}
                    ],
                },
            }
            json.dump(data, f)
            f.flush()

            rubrics = load_initial_rubrics(f.name)
            self.assertEqual(len(rubrics), 2)
            self.assertIn("p1", rubrics)
            self.assertIn("p2", rubrics)

            Path(f.name).unlink()


class TestSaveRubrics(unittest.TestCase):
    """Tests for save_rubrics function."""

    def test_save_rubrics_writes_best_rubrics(self) -> None:
        """Test that rubrics are saved in best_rubrics format."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 1.25},
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertIn("best_rubrics", payload)
            self.assertEqual(len(payload["best_rubrics"]), 1)
            self.assertEqual(payload["best_rubrics"][0]["prompt_id"], item.prompt_id)
            self.assertEqual(payload["best_rubrics"][0]["score"], 1.25)
            self.assertIn("rubric", payload["best_rubrics"][0])

    def test_save_rubrics_writes_best_candidates(self) -> None:
        """Test that best candidates are included when provided."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 1.25},
                best_candidates={item.prompt_id: item.candidates[0].candidate_id},
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            # best_candidate_id is stored in the rubric entry
            entry = payload["best_rubrics"][0]
            self.assertEqual(
                entry["best_candidate_id"],
                item.candidates[0].candidate_id
            )

    def test_save_rubrics_writes_candidate_scores(self) -> None:
        """Test that candidate scores are included when provided."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 0.42},
                best_candidates={item.prompt_id: "c1"},
                candidate_scores={item.prompt_id: {"c1": 0.9, "c2": 0.5, "c3": 0.1}},
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            # candidate_scores is stored in the rubric entry
            entry = payload["best_rubrics"][0]
            self.assertEqual(entry["candidate_scores"]["c1"], 0.9)

    def test_save_creates_parent_directories(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "dirs" / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 1.0},
            )
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()

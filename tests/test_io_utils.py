from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from autosr import io_utils
from autosr.io_utils import (
    load_dataset,
    load_initial_rubrics,
    save_run_record_files,
    save_rubrics,
)
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
            self.assertEqual(payload["best_objective_scores"][item.prompt_id], 1.25)
            self.assertEqual(payload["best_scores"][item.prompt_id], 1.25)

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

    def test_save_rubrics_writes_run_manifest(self) -> None:
        """Test that run manifest is included when provided."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))
        run_manifest = {
            "run_id": "20260222T100000_000000Z",
            "backend": {"requested": "auto", "resolved": "mock"},
            "seed": 7,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 1.0},
                run_manifest=run_manifest,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("run_manifest", payload)
            self.assertEqual(payload["run_manifest"]["run_id"], "20260222T100000_000000Z")

    def test_save_rubrics_writes_search_diagnostics(self) -> None:
        """Test that search diagnostics are included when provided."""
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))
        diagnostics = {
            "mode": "iterative",
            "margin_improvement": {
                "global": {
                    "total_prompts": 1,
                    "improved_prompts": 1,
                    "improvement_rate": 1.0,
                },
                "per_prompt": {
                    item.prompt_id: {
                        "initial_margin": 0.1,
                        "final_margin": 0.2,
                        "margin_delta": 0.1,
                        "improved": True,
                    }
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                best_rubrics={item.prompt_id: rubric},
                best_scores={item.prompt_id: 1.0},
                search_diagnostics=diagnostics,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("search_diagnostics", payload)
            self.assertEqual(payload["search_diagnostics"]["mode"], "iterative")
            self.assertEqual(
                payload["search_diagnostics"]["margin_improvement"]["global"]["improved_prompts"],
                1,
            )

    def test_save_run_record_files_writes_manifest_and_script(self) -> None:
        """Test per-run reproducibility files are archived in run_records."""
        run_manifest = {
            "run_id": "20260222T100000_000000Z",
            "backend": {"requested": "auto", "resolved": "mock"},
        }
        script_text = "#!/usr/bin/env bash\nset -euo pipefail\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "best_rubrics.json"
            manifest_path, script_path = save_run_record_files(
                output_path,
                run_manifest=run_manifest,
                reproducible_script=script_text,
            )

            self.assertTrue(manifest_path.exists())
            self.assertTrue(script_path.exists())
            self.assertIn("run_records", str(manifest_path))
            self.assertIn("run_records", str(script_path))
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"],
                "20260222T100000_000000Z",
            )
            self.assertEqual(script_path.read_text(encoding="utf-8"), script_text)

    def test_save_rubrics_atomic_write_is_always_parseable(self) -> None:
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            for score in (0.25, 0.5, 0.75):
                save_rubrics(
                    output_path,
                    best_rubrics={item.prompt_id: rubric},
                    best_scores={item.prompt_id: score},
                )
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["best_objective_scores"][item.prompt_id], score)

            self.assertEqual(list(output_path.parent.glob("out.json.*.tmp")), [])


class TestAtomicWrites(unittest.TestCase):
    def test_atomic_write_primitives_are_public(self) -> None:
        self.assertTrue(hasattr(io_utils, "atomic_write_text"))
        self.assertTrue(hasattr(io_utils, "atomic_write_json"))

    def test_atomic_write_json_writes_payload_and_cleans_temp_file(self) -> None:
        self.assertTrue(hasattr(io_utils, "atomic_write_json"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "payload.json"

            written_path = io_utils.atomic_write_json(output_path, {"value": 3, "label": "ok"})

            self.assertEqual(written_path, output_path)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {"value": 3, "label": "ok"},
            )
            self.assertEqual(list(output_path.parent.glob("payload.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

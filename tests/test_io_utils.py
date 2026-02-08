from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from autosr.io_utils import load_dataset, save_rubrics
from autosr.mock_components import HeuristicRubricInitializer


class TestIOUtils(unittest.TestCase):
    def test_save_rubrics_writes_best_candidates(self) -> None:
        item = load_dataset("examples/demo_dataset.json")[0]
        rubric = HeuristicRubricInitializer().initialize(item, rng=random.Random(123))
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.json"
            save_rubrics(
                output_path,
                rubrics={item.prompt_id: rubric},
                scores={item.prompt_id: 1.25},
                best_candidates={item.prompt_id: item.candidates[0].candidate_id},
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("best_rubrics", payload)
        self.assertIn("best_scores", payload)
        self.assertIn("best_candidates", payload)
        self.assertEqual(payload["best_candidates"][item.prompt_id], item.candidates[0].candidate_id)


if __name__ == "__main__":
    unittest.main()

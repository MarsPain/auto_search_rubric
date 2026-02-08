from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LLM_API_KEY = os.getenv("LLM_API_KEY")


@unittest.skipUnless(LLM_API_KEY, "LLM_API_KEY is required for integration test")
class TestLLMIntegration(unittest.TestCase):
    def test_cli_runs_iterative_and_evolutionary_with_llm(self) -> None:
        model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.json"
            iterative_output = tmp_path / "iterative.json"
            evolutionary_output = tmp_path / "evolutionary.json"
            dataset_path.write_text(json.dumps(_tiny_dataset(), ensure_ascii=False), encoding="utf-8")

            iterative_cmd = [
                "python3",
                "-m",
                "autosr.cli",
                "--dataset",
                str(dataset_path),
                "--mode",
                "iterative",
                "--output",
                str(iterative_output),
                "--backend",
                "auto",
                "--model-default",
                model_name,
                "--llm-max-retries",
                "1",
                "--iterations",
                "1",
            ]
            evolutionary_cmd = [
                "python3",
                "-m",
                "autosr.cli",
                "--dataset",
                str(dataset_path),
                "--mode",
                "evolutionary",
                "--output",
                str(evolutionary_output),
                "--backend",
                "auto",
                "--model-default",
                model_name,
                "--llm-max-retries",
                "1",
                "--generations",
                "1",
                "--population-size",
                "2",
                "--mutations-per-round",
                "1",
                "--batch-size",
                "1",
            ]

            iterative_result = subprocess.run(
                iterative_cmd,
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            evolutionary_result = subprocess.run(
                evolutionary_cmd,
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            self.assertIn("Backend: llm", iterative_result.stdout)
            self.assertIn("Backend: llm", evolutionary_result.stdout)
            self.assertTrue(iterative_output.exists())
            self.assertTrue(evolutionary_output.exists())

            iterative_payload = json.loads(iterative_output.read_text(encoding="utf-8"))
            evolutionary_payload = json.loads(evolutionary_output.read_text(encoding="utf-8"))
            self.assertIn("best_rubrics", iterative_payload)
            self.assertIn("best_scores", iterative_payload)
            self.assertIn("best_rubrics", evolutionary_payload)
            self.assertIn("best_scores", evolutionary_payload)


def _tiny_dataset() -> dict[str, object]:
    return {
        "prompts": [
            {
                "prompt_id": "p_smoke",
                "prompt": "Write a short launch email.",
                "candidates": [
                    {
                        "candidate_id": "a",
                        "source": "strong",
                        "text": "Subject: Launch now. Clear value and CTA.",
                    },
                    {
                        "candidate_id": "b",
                        "source": "base",
                        "text": "We launched a product. Please try it.",
                    },
                ],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()

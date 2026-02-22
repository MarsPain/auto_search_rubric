from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from autosr.cli import DEFAULT_MODEL


LLM_API_KEY = os.getenv("LLM_API_KEY")


@unittest.skipUnless(LLM_API_KEY, "LLM_API_KEY is required for integration test")
class TestLLMIntegration(unittest.TestCase):
    def test_cli_runs_iterative_and_evolutionary_with_llm(self) -> None:
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "dataset.json"
            iterative_output = tmp_path / "iterative.json"
            evolutionary_output = tmp_path / "evolutionary.json"
            dataset_path.write_text(json.dumps(_tiny_dataset(), ensure_ascii=False), encoding="utf-8")

            iterative_cmd = [
                sys.executable,
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
                "--base-url",
                base_url,
                "--model-default",
                model_name,
                "--llm-max-retries",
                "1",
                "--iterations",
                "1",
            ]
            evolutionary_cmd = [
                sys.executable,
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
                "--base-url",
                base_url,
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
                check=False,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if iterative_result.returncode != 0:
                self.fail(
                    "iterative CLI failed.\n"
                    f"command: {' '.join(iterative_cmd)}\n"
                    f"exit_code: {iterative_result.returncode}\n"
                    f"stdout:\n{iterative_result.stdout}\n"
                    f"stderr:\n{iterative_result.stderr}"
                )

            evolutionary_result = subprocess.run(
                evolutionary_cmd,
                check=False,
                cwd=repo_root,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if evolutionary_result.returncode != 0:
                self.fail(
                    "evolutionary CLI failed.\n"
                    f"command: {' '.join(evolutionary_cmd)}\n"
                    f"exit_code: {evolutionary_result.returncode}\n"
                    f"stdout:\n{evolutionary_result.stdout}\n"
                    f"stderr:\n{evolutionary_result.stderr}"
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

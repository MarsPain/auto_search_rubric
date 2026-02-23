from __future__ import annotations

import unittest
from pathlib import Path

from autosr.cli import build_runtime_config
from autosr.run_records.use_cases import build_reproducible_script, build_run_manifest


class TestCliReproducibility(unittest.TestCase):
    def test_build_run_manifest_contains_reproducibility_fields(self) -> None:
        class _Args:
            dataset = "examples/demo_dataset.json"
            output = "artifacts/out.json"
            mode = "evolutionary"
            api_key_env = "LLM_API_KEY"
            backend = "mock"
            base_url = "https://openrouter.ai/api/v1"
            model_default = "stepfun/step-3.5-flash:free"
            model_initializer = None
            model_proposer = None
            model_verifier = None
            model_judge = None
            llm_timeout = 30.0
            llm_max_retries = 2
            seed = 7
            iterations = 6
            generations = 12
            population_size = 8
            mutations_per_round = 6
            batch_size = 3
            selection_strategy = "rank"
            tournament_size = 3
            tournament_p = 0.8
            top_k_ratio = 0.3
            diversity_weight = 0.3
            adaptive_mutation = "fixed"
            mutation_window_size = 10
            min_mutation_weight = 0.1
            exploration_phase_ratio = 0.3
            diversity_threshold = 0.05
            tail_fraction = 0.25
            lambda_var = 0.2
            mu_diverse = 0.25
            noise = 0.08
            initializer_strategy = "backend"
            preset_rubrics = None
            preset_strict = False
            extract_strategy = "identity"
            extract_tag = "content"
            extract_pattern = None
            extract_join_separator = "\n\n"
            verbose = False

        args = _Args()
        config = build_runtime_config(args)
        manifest = build_run_manifest(
            args=args,
            config=config,
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            raw_cli_args=["--dataset", args.dataset, "--output", args.output, "--backend", "mock"],
            run_id="20260222T120000_123456Z",
            repo_root=Path.cwd(),
        )

        self.assertEqual(manifest["run_id"], "20260222T120000_123456Z")
        self.assertEqual(manifest["backend"]["requested"], "mock")
        self.assertEqual(manifest["backend"]["resolved"], "mock")
        self.assertEqual(manifest["seed"], 7)
        self.assertIn("dataset_sha256", manifest["dataset"])
        self.assertIn("normalized_argv", manifest["command"])

    def test_build_reproducible_script_requires_key_for_llm(self) -> None:
        manifest = {
            "workspace": {"repo_root": "/tmp/project"},
            "backend": {"resolved": "llm", "api_key_env": "LLM_API_KEY"},
            "command": {"normalized_argv": ["--dataset", "/tmp/d.json", "--output", "/tmp/o.json"]},
        }
        script = build_reproducible_script(manifest)
        self.assertIn('if [[ -z "${LLM_API_KEY:-}" ]]; then', script)
        self.assertIn("python3", script)
        self.assertIn("autosr.cli", script)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autosr.rl.data_models import TrainingManifest, TrainingResultManifest, EvalReport
from autosr.rl.registry import ExperimentRegistry
from autosr.rl.verl.reward_client import RMHealthzError, RMScoringClient, ScoreError


class MockHTTPResponse:
    """Minimal mock for urllib response objects."""

    def __init__(self, status: int, body: bytes) -> None:
        self.code = status
        self._body = body

    def read(self) -> bytes:
        return self._body


class TestRMScoringClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RMScoringClient(
            endpoint="http://127.0.0.1:8080",
            expected_artifact_id="artifact_001",
            expected_source_session_id="session_001",
            timeout_seconds=5.0,
        )

    def _make_healthz_response(
        self,
        artifact_id: str = "artifact_001",
        source_session_id: str = "session_001",
        status: str = "ok",
        schema_version: str = "1.0",
        rm_api_version: str = "1.0",
    ) -> bytes:
        return json.dumps({
            "status": status,
            "artifact_id": artifact_id,
            "source_session_id": source_session_id,
            "schema_version": schema_version,
            "rm_api_version": rm_api_version,
        }).encode("utf-8")

    @patch("urllib.request.urlopen")
    def test_healthz_check_success(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response()
        )
        result = self.client.healthz_check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["artifact_id"], "artifact_001")
        self.assertEqual(result["source_session_id"], "session_001")

    @patch("urllib.request.urlopen")
    def test_healthz_check_artifact_mismatch(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response(artifact_id="wrong_artifact")
        )
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("artifact_id mismatch", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_healthz_check_session_mismatch(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response(source_session_id="wrong_session")
        )
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("source_session_id mismatch", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_healthz_check_status_not_ok(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response(status="degraded")
        )
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("status is not ok", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_healthz_check_schema_version_mismatch(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response(schema_version="2.0")
        )
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("schema_version mismatch", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_healthz_check_rm_api_version_mismatch(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200, self._make_healthz_response(rm_api_version="2.0")
        )
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("rm_api_version mismatch", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_healthz_check_connection_error(self, mock_urlopen) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with self.assertRaises(RMHealthzError) as ctx:
            self.client.healthz_check()
        self.assertIn("Connection error", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_score_success(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200,
            json.dumps({
                "request_id": "req_001",
                "artifact_id": "artifact_001",
                "score": 0.85,
                "majority_grades": {"c1": 4.0},
            }).encode("utf-8"),
        )
        result = self.client.score(
            prompt_id="p1",
            prompt="What is 2+2?",
            candidate_id="c1",
            text="4",
        )
        self.assertEqual(result["score"], 0.85)
        self.assertEqual(result["request_id"], "req_001")

    @patch("urllib.request.urlopen")
    def test_score_http_error(self, mock_urlopen) -> None:
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://127.0.0.1:8080/score",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        # HTTPError.read() needs to return bytes; patch it on the instance
        with self.assertRaises(ScoreError) as ctx:
            self.client.score(
                prompt_id="p1",
                prompt="What is 2+2?",
                candidate_id="c1",
                text="4",
            )
        self.assertIn("HTTP 500", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_batch_score_success(self, mock_urlopen) -> None:
        mock_urlopen.return_value.__enter__.return_value = MockHTTPResponse(
            200,
            json.dumps({
                "request_id": "req_002",
                "artifact_id": "artifact_001",
                "results": [
                    {"index": 0, "ok": True, "score": 0.9},
                    {"index": 1, "ok": True, "score": 0.7},
                ],
            }).encode("utf-8"),
        )
        items = [
            {
                "prompt_id": "p1",
                "prompt": "Q1",
                "candidate": {"candidate_id": "c1", "text": "A1"},
            },
            {
                "prompt_id": "p2",
                "prompt": "Q2",
                "candidate": {"candidate_id": "c2", "text": "A2"},
            },
        ]
        result = self.client.batch_score(items)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["score"], 0.9)


class TestPrepareTrainingRunCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.output_dir = Path(self.temp_dir) / "outputs"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_prepare_creates_manifest_and_dirs(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.prepare_training_run",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--rm-deploy-id", "deploy_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--trainer-entrypoint", "python train.py",
            "--training-run-id", "train_test_001",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        # Check local dirs
        run_dir = self.output_dir / "train_test_001"
        self.assertTrue((run_dir / "checkpoints").exists())
        self.assertTrue((run_dir / "logs").exists())
        self.assertTrue((run_dir / "eval").exists())
        self.assertTrue((run_dir / "manifests").exists())

        # Check local manifest
        local_manifest = run_dir / "manifests" / "train_test_001.training.json"
        self.assertTrue(local_manifest.exists())
        data = json.loads(local_manifest.read_text())
        self.assertEqual(data["training_run_id"], "train_test_001")
        self.assertEqual(data["rm_artifact_id"], "artifact_001")

        # Check registry
        registry = ExperimentRegistry(base_dir=self.registry_dir)
        manifest = registry.get_manifest("train_test_001")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.rm_artifact_id, "artifact_001")

    def test_prepare_skip_healthz(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.prepare_training_run",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--trainer-entrypoint", "python train.py",
            "--training-run-id", "train_test_002",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Skipped RM server healthz check", result.stdout)


class TestFinalizeTrainingRunCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.registry = ExperimentRegistry(base_dir=self.registry_dir)
        # Pre-record a manifest
        manifest = TrainingManifest(
            training_run_id="train_test_003",
            rm_artifact_id="artifact_001",
            search_session_id="session_001",
            dataset={"dataset_id": "gsm8k", "dataset_version": "v1.0"},
            trainer={"project": "p", "code_version": "v1", "entrypoint": "python train.py"},
        )
        self.registry.record_manifest(manifest)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_finalize_success(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.finalize_training_run",
            "--training-run-id", "train_test_003",
            "--status", "succeeded",
            "--started-at", "2026-04-19T10:00:00+00:00",
            "--finished-at", "2026-04-19T11:00:00+00:00",
            "--duration-seconds", "3600",
            "--trainer-code-version", "abc123",
            "--checkpoint-path", "/tmp/checkpoint",
            "--log-path", "/tmp/log",
            "--training-summary-json", '{"final_loss": 0.1}',
            "--reward-summary-json", '{"mean": 0.8}',
            "--registry-dir", str(self.registry_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        result_obj = self.registry.get_result("train_test_003")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "succeeded")
        self.assertEqual(result_obj.duration_seconds, 3600.0)
        self.assertEqual(result_obj.output["checkpoint_path"], "/tmp/checkpoint")

    def test_finalize_failed(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.finalize_training_run",
            "--training-run-id", "train_test_003",
            "--status", "failed",
            "--started-at", "2026-04-19T10:00:00+00:00",
            "--finished-at", "2026-04-19T10:05:00+00:00",
            "--duration-seconds", "300",
            "--trainer-code-version", "abc123",
            "--failure-type", "OOMError",
            "--failure-message", "CUDA out of memory",
            "--failure-stage", "training",
            "--registry-dir", str(self.registry_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        result_obj = self.registry.get_result("train_test_003")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "failed")
        self.assertIsNotNone(result_obj.failure)
        self.assertEqual(result_obj.failure["type"], "OOMError")
        self.assertEqual(result_obj.failure["stage"], "training")

    def test_finalize_with_eval_report(self) -> None:
        eval_path = Path(self.temp_dir) / "eval_report.json"
        eval_report = EvalReport(
            eval_run_id="eval_001",
            training_run_id="train_test_003",
            benchmark={"name": "gsm8k", "version": "v1"},
            metrics={"accuracy": 0.75},
        )
        eval_path.write_text(eval_report.to_json(indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.finalize_training_run",
            "--training-run-id", "train_test_003",
            "--status", "succeeded",
            "--started-at", "2026-04-19T10:00:00+00:00",
            "--finished-at", "2026-04-19T11:00:00+00:00",
            "--duration-seconds", "3600",
            "--trainer-code-version", "abc123",
            "--checkpoint-path", "/tmp/checkpoint",
            "--training-summary-json", '{"final_loss": 0.1}',
            "--registry-dir", str(self.registry_dir),
            "--eval-report-json", str(eval_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        eval_obj = self.registry.get_eval("eval_001")
        self.assertIsNotNone(eval_obj)
        self.assertEqual(eval_obj.metrics["accuracy"], 0.75)


class TestRunVerlTrainingOrchestration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"
        self.output_dir = Path(self.temp_dir) / "outputs"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dry_run_outputs_commands(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.run_verl_training",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
            "--dry-run",
            "--",
            "echo", "hello",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("[dry-run]", result.stdout)
        self.assertIn("echo hello", result.stdout)

    def test_orchestrate_successful_run(self) -> None:
        # Run a minimal command that succeeds
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.run_verl_training",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
            "--training-run-id", "orch_test_001",
            "--",
            sys.executable, "-c", "import os; print('RM_ENDPOINT:', os.environ.get('RM_ENDPOINT')); print('TRAINING_RUN_ID:', os.environ.get('TRAINING_RUN_ID'))",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("RM_ENDPOINT: http://127.0.0.1:8080", result.stdout)
        self.assertIn("TRAINING_RUN_ID: orch_test_001", result.stdout)

        # Verify manifest and result were recorded
        registry = ExperimentRegistry(base_dir=self.registry_dir)
        manifest = registry.get_manifest("orch_test_001")
        self.assertIsNotNone(manifest)
        result_obj = registry.get_result("orch_test_001")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "succeeded")

    def test_orchestrate_failed_run(self) -> None:
        # Run a command that fails
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.run_verl_training",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
            "--training-run-id", "orch_test_002",
            "--",
            sys.executable, "-c", "import sys; sys.exit(1)",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

        # Verify manifest and failed result were recorded
        registry = ExperimentRegistry(base_dir=self.registry_dir)
        manifest = registry.get_manifest("orch_test_002")
        self.assertIsNotNone(manifest)
        result_obj = registry.get_result("orch_test_002")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "failed")

    def test_orchestrate_preflight_failure_records_failed_result(self) -> None:
        # Force prepare failure deterministically with invalid JSON
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.run_verl_training",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
            "--trainer-config-json", "{invalid_json",
            "--training-run-id", "orch_test_preflight_001",
            "--",
            "echo", "hello",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

        registry = ExperimentRegistry(base_dir=self.registry_dir)
        manifest = registry.get_manifest("orch_test_preflight_001")
        self.assertIsNotNone(manifest)
        result_obj = registry.get_result("orch_test_preflight_001")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "failed")
        self.assertIsNotNone(result_obj.failure)
        self.assertEqual(result_obj.failure["stage"], "preflight")

    def test_orchestrate_missing_trainer_command_records_failed_result(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "autosr.rl.verl.run_verl_training",
            "--rm-endpoint", "http://127.0.0.1:8080",
            "--rm-artifact-id", "artifact_001",
            "--search-session-id", "session_001",
            "--dataset-id", "gsm8k",
            "--dataset-version", "v1.0",
            "--trainer-project", "verl_grpo",
            "--trainer-code-version", "abc123",
            "--output-dir", str(self.output_dir),
            "--registry-dir", str(self.registry_dir),
            "--skip-healthz",
            "--training-run-id", "orch_test_003",
            "--",
            "does_not_exist_cmd",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

        registry = ExperimentRegistry(base_dir=self.registry_dir)
        manifest = registry.get_manifest("orch_test_003")
        self.assertIsNotNone(manifest)
        result_obj = registry.get_result("orch_test_003")
        self.assertIsNotNone(result_obj)
        self.assertEqual(result_obj.status, "failed")
        self.assertIsNotNone(result_obj.failure)
        self.assertEqual(result_obj.failure["type"], "FileNotFoundError")
        self.assertEqual(result_obj.failure["stage"], "training")


if __name__ == "__main__":
    unittest.main()

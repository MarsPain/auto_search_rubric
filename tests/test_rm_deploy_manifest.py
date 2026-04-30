from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from reward_harness.data_models import Criterion, GradingProtocol, Rubric


def _build_test_rubric(rubric_id: str) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criteria=[Criterion(criterion_id="c1", text="quality", weight=1.0)],
        grading_protocol=GradingProtocol(),
    )


def _write_artifact(path: Path, *, artifact_id: str = "rm_001") -> None:
    payload = {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "created_at_utc": "2026-04-16T00:00:00+00:00",
        "source_session_id": "session_123",
        "source_run_id": "run_123",
        "dataset_hash": "dataset_hash_123",
        "config_hash": "config_hash_123",
        "rubric": {"p1": _build_test_rubric("r1").to_dict()},
        "scoring_policy": {"policy_version": "1.0"},
        "normalization": {"method": "identity"},
        "compatibility": {"artifact_type": "rm", "min_rm_api_version": "1.0"},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestDeployManifestDataModel(unittest.TestCase):
    def test_roundtrip(self) -> None:
        from reward_harness.rm.data_models import DeployManifest

        manifest = DeployManifest(
            deploy_id="deploy_20260416T120000_000000Z",
            deployed_at_utc="2026-04-16T12:00:00+00:00",
            artifact_id="rm_001",
            artifact_path="/tmp/rm_001.json",
            deployed_by="alice",
            deployment_target="dev",
            previous_artifact_id=None,
            rollback_policy={"strategy": "rollback_to_previous_artifact"},
            source_session_id="session_123",
            dataset_hash="dataset_hash_123",
            config_hash="config_hash_123",
        )

        restored = DeployManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.deploy_id, manifest.deploy_id)
        self.assertEqual(restored.deployment_target, "dev")

    def test_rejects_invalid_deployment_target(self) -> None:
        from reward_harness.rm.data_models import (
            ArtifactValidationError,
            DeployManifest,
        )

        with self.assertRaises(ArtifactValidationError):
            DeployManifest(
                deploy_id="deploy_20260416T120000_000000Z",
                deployed_at_utc="2026-04-16T12:00:00+00:00",
                artifact_id="rm_001",
                artifact_path="/tmp/rm_001.json",
                deployed_by="alice",
                deployment_target="qa",
                previous_artifact_id=None,
                rollback_policy={"strategy": "rollback_to_previous_artifact"},
                source_session_id="session_123",
                dataset_hash="dataset_hash_123",
                config_hash="config_hash_123",
            )


class TestDeployManifestUseCases(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = Path(self.temp_dir) / "rm_001.json"
        self.deploy_dir = Path(self.temp_dir) / "deploys"
        _write_artifact(self.artifact_path, artifact_id="rm_001")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_first_deploy_has_no_previous_artifact(self) -> None:
        from reward_harness.rm.io import load_deploy_manifest
        from reward_harness.rm.use_cases import record_deploy_manifest

        manifest_path = record_deploy_manifest(
            artifact_path=self.artifact_path,
            deployment_target="dev",
            deployed_by="alice",
            out_dir=self.deploy_dir,
        )
        manifest = load_deploy_manifest(manifest_path)
        self.assertIsNone(manifest.previous_artifact_id)
        self.assertEqual(manifest.artifact_id, "rm_001")

    def test_auto_resolves_previous_artifact_on_same_target(self) -> None:
        from reward_harness.rm.io import load_deploy_manifest
        from reward_harness.rm.use_cases import record_deploy_manifest

        first_manifest_path = record_deploy_manifest(
            artifact_path=self.artifact_path,
            deployment_target="staging",
            deployed_by="alice",
            out_dir=self.deploy_dir,
        )
        first_manifest = load_deploy_manifest(first_manifest_path)

        artifact_2 = Path(self.temp_dir) / "rm_002.json"
        _write_artifact(artifact_2, artifact_id="rm_002")
        second_manifest_path = record_deploy_manifest(
            artifact_path=artifact_2,
            deployment_target="staging",
            deployed_by="alice",
            out_dir=self.deploy_dir,
        )
        second_manifest = load_deploy_manifest(second_manifest_path)

        self.assertEqual(
            second_manifest.previous_artifact_id, first_manifest.artifact_id
        )
        self.assertEqual(second_manifest.artifact_id, "rm_002")

    def test_explicit_previous_artifact_overrides_auto_resolution(self) -> None:
        from reward_harness.rm.io import load_deploy_manifest
        from reward_harness.rm.use_cases import record_deploy_manifest

        record_deploy_manifest(
            artifact_path=self.artifact_path,
            deployment_target="prod",
            deployed_by="alice",
            out_dir=self.deploy_dir,
        )
        artifact_2 = Path(self.temp_dir) / "rm_002.json"
        _write_artifact(artifact_2, artifact_id="rm_002")

        second_manifest_path = record_deploy_manifest(
            artifact_path=artifact_2,
            deployment_target="prod",
            deployed_by="alice",
            previous_artifact_id="rm_manual_previous",
            out_dir=self.deploy_dir,
        )
        second_manifest = load_deploy_manifest(second_manifest_path)
        self.assertEqual(second_manifest.previous_artifact_id, "rm_manual_previous")

    def test_rejects_self_reference_previous_artifact(self) -> None:
        from reward_harness.rm.data_models import ArtifactValidationError
        from reward_harness.rm.use_cases import record_deploy_manifest

        with self.assertRaises(ArtifactValidationError):
            record_deploy_manifest(
                artifact_path=self.artifact_path,
                deployment_target="dev",
                deployed_by="alice",
                previous_artifact_id="rm_001",
                out_dir=self.deploy_dir,
            )

    def test_missing_artifact_path_raises_file_not_found(self) -> None:
        from reward_harness.rm.use_cases import record_deploy_manifest

        with self.assertRaises(FileNotFoundError):
            record_deploy_manifest(
                artifact_path=Path(self.temp_dir) / "missing.json",
                deployment_target="dev",
                deployed_by="alice",
                out_dir=self.deploy_dir,
            )

    def test_corrupted_artifact_raises_validation_error(self) -> None:
        from reward_harness.rm.data_models import ArtifactValidationError
        from reward_harness.rm.use_cases import record_deploy_manifest

        broken_artifact = Path(self.temp_dir) / "broken_artifact.json"
        broken_artifact.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ArtifactValidationError):
            record_deploy_manifest(
                artifact_path=broken_artifact,
                deployment_target="dev",
                deployed_by="alice",
                out_dir=self.deploy_dir,
            )


class TestDeployManifestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = Path(self.temp_dir) / "rm_001.json"
        self.deploy_dir = Path(self.temp_dir) / "deploys"
        _write_artifact(self.artifact_path, artifact_id="rm_001")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_creates_manifest_and_prints_summary(self) -> None:
        from reward_harness.rm.deploy import main
        from reward_harness.rm.io import load_deploy_manifest

        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "deploy.py",
                "--artifact",
                str(self.artifact_path),
                "--deployment-target",
                "prod",
                "--deployed-by",
                "alice",
                "--out-dir",
                str(self.deploy_dir),
            ],
        ):
            with redirect_stdout(stdout):
                main()

        output = stdout.getvalue()
        self.assertIn("deploy_id:", output)
        self.assertIn("artifact_id: rm_001", output)
        self.assertIn("deployment_target: prod", output)

        manifests = sorted(self.deploy_dir.glob("*.json"))
        self.assertEqual(len(manifests), 1)
        manifest = load_deploy_manifest(manifests[0])
        self.assertEqual(manifest.artifact_id, "rm_001")

    def test_cli_errors_on_invalid_target(self) -> None:
        from reward_harness.rm.deploy import main

        stderr = io.StringIO()
        with patch(
            "sys.argv",
            [
                "deploy.py",
                "--artifact",
                str(self.artifact_path),
                "--deployment-target",
                "qa",
                "--out-dir",
                str(self.deploy_dir),
            ],
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()

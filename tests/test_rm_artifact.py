from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autosr.data_models import Criterion, GradingProtocol, Rubric


def _build_test_rubric(rubric_id: str) -> Rubric:
    return Rubric(
        rubric_id=rubric_id,
        criteria=[Criterion(criterion_id="c1", text="quality", weight=1.0)],
        grading_protocol=GradingProtocol(),
    )


def _write_search_output(path: Path, *, with_manifest: bool = True) -> None:
    payload: dict[str, object] = {
        "best_rubrics": [
            {
                "prompt_id": "p1",
                "rubric": _build_test_rubric("r1").to_dict(),
                "score": 0.9,
            },
            {
                "prompt_id": "p2",
                "rubric": _build_test_rubric("r2").to_dict(),
                "score": 0.8,
            },
        ],
        "best_objective_scores": {"p1": 0.9, "p2": 0.8},
        "best_scores": {"p1": 0.9, "p2": 0.8},
    }
    if with_manifest:
        payload["run_manifest"] = {
            "schema_version": "1.0",
            "run_id": "run_20260404_001",
            "dataset": {
                "path": "/tmp/dataset.json",
                "dataset_sha256": "dataset_hash_123",
            },
            "config_snapshot": {
                "search": {"mode": "evolutionary", "generations": 3},
                "objective": {"tail_fraction": 0.25},
            },
            "backend": {"requested": "mock", "resolved": "mock"},
            "llm_snapshot": {"default_model": "dummy"},
            "harness": {"session_id": "session_123"},
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class TestRMArtifactDataModel(unittest.TestCase):
    def test_artifact_roundtrip(self) -> None:
        from autosr.rm.data_models import RMArtifact

        artifact = RMArtifact(
            artifact_id="rm_001",
            created_at_utc="2026-04-04T00:00:00+00:00",
            source_session_id="session_123",
            source_run_id="run_123",
            dataset_hash="dataset_hash",
            config_hash="config_hash",
            rubric={"p1": _build_test_rubric("r1")},
            scoring_policy={"policy_version": "1"},
            normalization={"method": "identity"},
            compatibility={"artifact_type": "rm", "min_rm_api_version": "1.0"},
        )
        restored = RMArtifact.from_dict(artifact.to_dict())
        self.assertEqual(restored.artifact_id, "rm_001")
        self.assertIn("p1", restored.rubric)

    def test_artifact_validation_missing_required_field(self) -> None:
        from autosr.rm.data_models import ArtifactValidationError, RMArtifact

        with self.assertRaises(ArtifactValidationError):
            RMArtifact(
                artifact_id="",
                created_at_utc="2026-04-04T00:00:00+00:00",
                source_session_id="session_123",
                source_run_id="run_123",
                dataset_hash="dataset_hash",
                config_hash="config_hash",
                rubric={"p1": _build_test_rubric("r1")},
                scoring_policy={"policy_version": "1"},
                normalization={"method": "identity"},
                compatibility={"artifact_type": "rm", "min_rm_api_version": "1.0"},
            )


class TestRMArtifactUseCases(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.search_output = Path(self.temp_dir) / "search_output.json"
        self.artifact_output = Path(self.temp_dir) / "rm_artifact.json"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_rm_artifact_from_search_output(self) -> None:
        from autosr.rm.use_cases import build_rm_artifact

        _write_search_output(self.search_output)
        artifact = build_rm_artifact(search_output_path=self.search_output)

        self.assertEqual(artifact.source_session_id, "session_123")
        self.assertEqual(artifact.source_run_id, "run_20260404_001")
        self.assertEqual(artifact.dataset_hash, "dataset_hash_123")
        self.assertIn("p1", artifact.rubric)
        self.assertIn("p2", artifact.rubric)

    def test_validate_rm_artifact_hash_consistency(self) -> None:
        from autosr.rm.data_models import ArtifactValidationError
        from autosr.rm.use_cases import build_rm_artifact, validate_rm_artifact

        _write_search_output(self.search_output)
        artifact = build_rm_artifact(search_output_path=self.search_output)

        # Tamper hash to trigger consistency validation failure.
        artifact.dataset_hash = "tampered_hash"
        with self.assertRaises(ArtifactValidationError):
            validate_rm_artifact(artifact, source_search_output_path=self.search_output)

    def test_export_cli_writes_artifact_file(self) -> None:
        from autosr.rm.io import load_rm_artifact
        from autosr.rm.use_cases import export_rm_artifact

        _write_search_output(self.search_output)
        output_path = export_rm_artifact(
            search_output_path=self.search_output,
            out_artifact_path=self.artifact_output,
        )
        self.assertEqual(output_path, self.artifact_output)
        loaded = load_rm_artifact(self.artifact_output)
        self.assertEqual(loaded.source_run_id, "run_20260404_001")


if __name__ == "__main__":
    unittest.main()

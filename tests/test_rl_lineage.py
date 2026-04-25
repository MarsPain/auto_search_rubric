from __future__ import annotations

from dataclasses import MISSING, fields
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autosr.rl.data_models import (
    EvalReport,
    LineageIndex,
    TrainingManifest,
    TrainingResultManifest,
    TrainingValidationError,
)
from autosr.rl.registry import (
    DuplicateEntryError,
    ExperimentRegistry,
    MissingManifestError,
    RegistryError,
)
from autosr.rl.io import save_eval_report
from autosr.rl.validation import (
    LineageValidationError,
    validate_cross_consistency,
    validate_eval_report,
    validate_payload_before_record,
    validate_training_manifest,
    validate_training_result,
)
from autosr.rl.lineage import LineageView, build_lineage_view, format_lineage_text, list_all_training_runs


def _build_valid_manifest(training_run_id: str = "train_001") -> TrainingManifest:
    return TrainingManifest(
        training_run_id=training_run_id,
        rm_artifact_id="rm_001",
        search_session_id="session_001",
        rm_endpoint="http://localhost:8080",
        rm_deploy_id="deploy_001",
        dataset={
            "dataset_id": "ds_001",
            "dataset_version": "v1.0",
            "split": "train",
        },
        trainer={
            "project": "rl_project",
            "repo_url": "https://github.com/example/rl",
            "code_version": "abc123",
            "entrypoint": "python train.py",
        },
        trainer_config={"lr": 0.001},
        execution={"launcher": "local", "host": "gpu-node-1", "accelerator": "cuda", "num_workers": 4},
        tags=["exp1"],
        notes="test run",
    )


def _build_valid_result(
    training_run_id: str = "train_001",
    status: str = "succeeded",
) -> TrainingResultManifest:
    return TrainingResultManifest(
        training_run_id=training_run_id,
        status=status,
        started_at_utc="2026-04-17T00:00:00+00:00",
        finished_at_utc="2026-04-17T01:00:00+00:00",
        duration_seconds=3600.0,
        trainer_code_version="abc123",
        output={"checkpoint_path": "/tmp/checkpoint", "log_path": "/tmp/log"},
        reward_summary={"mean": 0.8},
        training_summary={"final_loss": 0.1},
    )


def _build_valid_eval(eval_run_id: str = "eval_001", training_run_id: str = "train_001") -> EvalReport:
    return EvalReport(
        eval_run_id=eval_run_id,
        training_run_id=training_run_id,
        benchmark={"name": "hellaswag", "version": "v1", "split": "val"},
        metrics={"accuracy": 0.75},
        artifacts={"report_path": "/tmp/report"},
        summary="good",
        comparison_baseline="baseline_001",
    )


class TestTrainingManifestDataModel(unittest.TestCase):
    def test_valid_manifest(self) -> None:
        m = _build_valid_manifest()
        self.assertEqual(m.training_run_id, "train_001")
        self.assertEqual(m.schema_version, "1.0")

    def test_missing_training_run_id(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingManifest(
                training_run_id="",
                rm_artifact_id="rm_001",
                search_session_id="session_001",
                dataset={"dataset_version": "v1"},
                trainer={"project": "p", "code_version": "v", "entrypoint": "e"},
            )

    def test_missing_dataset_version(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingManifest(
                training_run_id="train_001",
                rm_artifact_id="rm_001",
                search_session_id="session_001",
                dataset={},
                trainer={"project": "p", "code_version": "v", "entrypoint": "e"},
            )

    def test_missing_trainer_project(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingManifest(
                training_run_id="train_001",
                rm_artifact_id="rm_001",
                search_session_id="session_001",
                dataset={"dataset_version": "v1"},
                trainer={"code_version": "v", "entrypoint": "e"},
            )

    def test_invalid_endpoint(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingManifest(
                training_run_id="train_001",
                rm_artifact_id="rm_001",
                search_session_id="session_001",
                rm_endpoint="not-a-url",
                dataset={"dataset_version": "v1"},
                trainer={"project": "p", "code_version": "v", "entrypoint": "e"},
            )

    def test_roundtrip(self) -> None:
        m = _build_valid_manifest()
        d = m.to_dict()
        m2 = TrainingManifest.from_dict(d)
        self.assertEqual(m.to_dict(), m2.to_dict())

    def test_json_roundtrip(self) -> None:
        m = _build_valid_manifest()
        s = m.to_json()
        m2 = TrainingManifest.from_json(s)
        self.assertEqual(m.to_dict(), m2.to_dict())

    def test_from_json_has_single_schema_contract(self) -> None:
        source = inspect.getsource(TrainingManifest)

        self.assertEqual(source.count("def from_json"), 1)
        self.assertEqual(TrainingManifest.from_json.__annotations__["return"], "TrainingManifest")


class TestTrainingResultManifestDataModel(unittest.TestCase):
    def test_valid_succeeded(self) -> None:
        r = _build_valid_result()
        self.assertEqual(r.status, "succeeded")

    def test_valid_failed(self) -> None:
        r = TrainingResultManifest(
            training_run_id="train_001",
            status="failed",
            started_at_utc="2026-04-17T00:00:00+00:00",
            finished_at_utc="2026-04-17T00:01:00+00:00",
            duration_seconds=60.0,
            trainer_code_version="abc123",
            failure={"type": "RuntimeError", "message": "OOM", "stage": "training"},
        )
        self.assertEqual(r.status, "failed")

    def test_valid_canceled(self) -> None:
        r = TrainingResultManifest(
            training_run_id="train_001",
            status="canceled",
            started_at_utc="2026-04-17T00:00:00+00:00",
            finished_at_utc="2026-04-17T00:01:00+00:00",
            duration_seconds=60.0,
            trainer_code_version="abc123",
            failure={"message": "user canceled"},
        )
        self.assertEqual(r.status, "canceled")

    def test_succeeded_requires_summary(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingResultManifest(
                training_run_id="train_001",
                status="succeeded",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T01:00:00+00:00",
                duration_seconds=3600.0,
                trainer_code_version="abc123",
                output={"checkpoint_path": "/tmp/cp"},
                training_summary={},
            )

    def test_succeeded_requires_output_path(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingResultManifest(
                training_run_id="train_001",
                status="succeeded",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T01:00:00+00:00",
                duration_seconds=3600.0,
                trainer_code_version="abc123",
                output={},
                training_summary={"final_loss": 0.1},
            )

    def test_failed_requires_failure_fields(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingResultManifest(
                training_run_id="train_001",
                status="failed",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T00:01:00+00:00",
                duration_seconds=60.0,
                trainer_code_version="abc123",
                failure={"type": "RuntimeError"},
            )

    def test_failed_requires_valid_stage(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingResultManifest(
                training_run_id="train_001",
                status="failed",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T00:01:00+00:00",
                duration_seconds=60.0,
                trainer_code_version="abc123",
                failure={"type": "Error", "message": "x", "stage": "invalid"},
            )

    def test_negative_duration(self) -> None:
        with self.assertRaises(TrainingValidationError):
            TrainingResultManifest(
                training_run_id="train_001",
                status="succeeded",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T01:00:00+00:00",
                duration_seconds=-1.0,
                trainer_code_version="abc123",
                output={"checkpoint_path": "/tmp/cp"},
                training_summary={"final_loss": 0.1},
            )

    def test_roundtrip(self) -> None:
        r = _build_valid_result()
        d = r.to_dict()
        r2 = TrainingResultManifest.from_dict(d)
        self.assertEqual(r.to_dict(), r2.to_dict())


class TestEvalReportDataModel(unittest.TestCase):
    def test_valid(self) -> None:
        e = _build_valid_eval()
        self.assertEqual(e.eval_run_id, "eval_001")

    def test_missing_benchmark_name(self) -> None:
        with self.assertRaises(TrainingValidationError):
            EvalReport(
                eval_run_id="eval_001",
                training_run_id="train_001",
                benchmark={"version": "v1"},
                metrics={"acc": 0.5},
            )

    def test_missing_benchmark_version(self) -> None:
        with self.assertRaises(TrainingValidationError):
            EvalReport(
                eval_run_id="eval_001",
                training_run_id="train_001",
                benchmark={"name": "bench"},
                metrics={"acc": 0.5},
            )

    def test_missing_metrics(self) -> None:
        with self.assertRaises(TrainingValidationError):
            EvalReport(
                eval_run_id="eval_001",
                training_run_id="train_001",
                benchmark={"name": "bench", "version": "v1"},
                metrics={},
            )

    def test_roundtrip(self) -> None:
        e = _build_valid_eval()
        d = e.to_dict()
        e2 = EvalReport.from_dict(d)
        self.assertEqual(e.to_dict(), e2.to_dict())


class TestLineageIndexDataModel(unittest.TestCase):
    def test_roundtrip(self) -> None:
        idx = LineageIndex(
            training_run_id="train_001",
            rm_artifact_id="rm_001",
            rm_deploy_id="deploy_001",
            search_session_id="session_001",
            eval_run_ids=["eval_001", "eval_002"],
        )
        d = idx.to_dict()
        idx2 = LineageIndex.from_dict(d)
        self.assertEqual(d, idx2.to_dict())


class TestExperimentRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_record_and_get_manifest(self) -> None:
        m = _build_valid_manifest("train_001")
        path = self.registry.record_manifest(m)
        self.assertTrue(path.exists())
        got = self.registry.get_manifest("train_001")
        self.assertIsNotNone(got)
        self.assertEqual(got.to_dict(), m.to_dict())

    def test_record_manifest_rejects_duplicate(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        with self.assertRaises(DuplicateEntryError):
            self.registry.record_manifest(m, allow_identical_overwrite=False)

    def test_record_manifest_idempotent(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        path2 = self.registry.record_manifest(m, allow_identical_overwrite=True)
        self.assertTrue(path2.exists())

    def test_record_result_requires_manifest(self) -> None:
        r = _build_valid_result("train_001")
        with self.assertRaises(MissingManifestError):
            self.registry.record_result(r)

    def test_record_result(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = _build_valid_result("train_001")
        path = self.registry.record_result(r)
        self.assertTrue(path.exists())
        got = self.registry.get_result("train_001")
        self.assertEqual(got.to_dict(), r.to_dict())

    def test_record_result_rejects_overwrite(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = _build_valid_result("train_001")
        self.registry.record_result(r)
        with self.assertRaises(DuplicateEntryError):
            self.registry.record_result(r, allow_overwrite=False)

    def test_record_result_allow_overwrite(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = _build_valid_result("train_001")
        self.registry.record_result(r)
        with self.assertRaises(DuplicateEntryError):
            self.registry.record_result(r, allow_overwrite=True)

    def test_record_eval_requires_manifest(self) -> None:
        e = _build_valid_eval("eval_001", "train_001")
        with self.assertRaises(MissingManifestError):
            self.registry.record_eval(e)

    def test_record_eval(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        e = _build_valid_eval("eval_001", "train_001")
        path = self.registry.record_eval(e)
        self.assertTrue(path.exists())

    def test_list_evals_for_training_run(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        e1 = _build_valid_eval("eval_001", "train_001")
        e2 = _build_valid_eval("eval_002", "train_001")
        self.registry.record_eval(e1)
        self.registry.record_eval(e2)
        ids = self.registry.list_evals_for_training_run("train_001")
        self.assertEqual(sorted(ids), ["eval_001", "eval_002"])

    def test_fallback_scan_reads_each_eval_once(self) -> None:
        class CountingRegistry(ExperimentRegistry):
            def __init__(self, base_dir: str) -> None:
                super().__init__(base_dir)
                self.get_eval_calls: list[str] = []

            def get_eval(self, eval_run_id: str) -> EvalReport | None:
                self.get_eval_calls.append(eval_run_id)
                return super().get_eval(eval_run_id)

        registry = CountingRegistry(self.tmpdir.name)
        save_eval_report(registry.evals_dir / "eval_001.json", _build_valid_eval("eval_001", "train_001"))
        save_eval_report(registry.evals_dir / "eval_002.json", _build_valid_eval("eval_002", "train_002"))

        evals = registry.list_evals_for_training_run("train_001")

        self.assertEqual(evals, ["eval_001"])
        self.assertEqual(registry.get_eval_calls.count("eval_001"), 1)
        self.assertEqual(registry.get_eval_calls.count("eval_002"), 1)

    def test_resolve_lineage(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = _build_valid_result("train_001")
        self.registry.record_result(r)
        e = _build_valid_eval("eval_001", "train_001")
        self.registry.record_eval(e)
        lineage = self.registry.resolve_lineage("train_001")
        self.assertIsNotNone(lineage)
        self.assertEqual(lineage["training_run_id"], "train_001")
        self.assertEqual(lineage["upstream"]["rm_artifact_id"], "rm_001")
        self.assertEqual(len(lineage["evals"]), 1)

    def test_list_training_run_ids(self) -> None:
        self.registry.record_manifest(_build_valid_manifest("train_001"))
        self.registry.record_manifest(_build_valid_manifest("train_002"))
        ids = self.registry.list_training_run_ids()
        self.assertEqual(ids, ["train_001", "train_002"])

    def test_missing_manifest_returns_none(self) -> None:
        self.assertIsNone(self.registry.get_manifest("nonexistent"))
        self.assertIsNone(self.registry.get_result("nonexistent"))
        self.assertIsNone(self.registry.get_eval("nonexistent"))

    def test_record_manifest_rejects_unsafe_id(self) -> None:
        m = _build_valid_manifest("bad/escape")
        with self.assertRaises(RegistryError):
            self.registry.record_manifest(m)


class TestValidation(unittest.TestCase):
    def test_validate_training_manifest(self) -> None:
        m = _build_valid_manifest()
        validate_training_manifest(m)

    def test_validate_training_result(self) -> None:
        r = _build_valid_result()
        validate_training_result(r)

    def test_validate_eval_report(self) -> None:
        e = _build_valid_eval()
        validate_eval_report(e)

    def test_validate_cross_consistency_ok(self) -> None:
        m = _build_valid_manifest("train_001")
        r = _build_valid_result("train_001")
        e = _build_valid_eval("eval_001", "train_001")
        validate_cross_consistency(m, r, [e])

    def test_validate_cross_consistency_result_mismatch(self) -> None:
        m = _build_valid_manifest("train_001")
        r = _build_valid_result("train_002")
        with self.assertRaises(LineageValidationError):
            validate_cross_consistency(m, r)

    def test_validate_cross_consistency_eval_mismatch(self) -> None:
        m = _build_valid_manifest("train_001")
        e = _build_valid_eval("eval_001", "train_002")
        with self.assertRaises(LineageValidationError):
            validate_cross_consistency(m, evals=[e])

    def test_validate_payload_manifest(self) -> None:
        payload = _build_valid_manifest().to_dict()
        validate_payload_before_record(payload, kind="manifest")

    def test_validate_payload_result(self) -> None:
        payload = _build_valid_result().to_dict()
        validate_payload_before_record(payload, kind="result")

    def test_validate_payload_eval(self) -> None:
        payload = _build_valid_eval().to_dict()
        validate_payload_before_record(payload, kind="eval")

    def test_validate_payload_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            validate_payload_before_record({}, kind="unknown")


class TestLineageView(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_mutable_fields_use_default_factories(self) -> None:
        field_by_name = {f.name: f for f in fields(LineageView)}

        self.assertIsNot(field_by_name["eval_benchmarks"].default_factory, MISSING)
        self.assertIsNot(field_by_name["upstream_chain"].default_factory, MISSING)

    def test_build_lineage_view(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = _build_valid_result("train_001", status="succeeded")
        self.registry.record_result(r)
        e = _build_valid_eval("eval_001", "train_001")
        self.registry.record_eval(e)

        view = build_lineage_view(self.registry, "train_001")
        self.assertIsNotNone(view)
        self.assertEqual(view.training_run_id, "train_001")
        self.assertEqual(view.status, "succeeded")
        self.assertEqual(view.rm_artifact_id, "rm_001")
        self.assertEqual(view.search_session_id, "session_001")
        self.assertEqual(view.eval_count, 1)
        self.assertEqual(view.eval_benchmarks, ["hellaswag"])

    def test_build_lineage_view_missing(self) -> None:
        view = build_lineage_view(self.registry, "nonexistent")
        self.assertIsNone(view)

    def test_format_lineage_text(self) -> None:
        view = LineageView(
            training_run_id="train_001",
            status="succeeded",
            rm_artifact_id="rm_001",
            rm_deploy_id="deploy_001",
            search_session_id="session_001",
            dataset_version="v1",
            code_version="abc123",
            duration_seconds=3600.0,
            eval_count=1,
            eval_benchmarks=["hellaswag"],
        )
        text = format_lineage_text(view)
        self.assertIn("train_001", text)
        self.assertIn("succeeded", text)
        self.assertIn("hellaswag", text)

    def test_list_all_training_runs(self) -> None:
        self.registry.record_manifest(_build_valid_manifest("train_001"))
        self.registry.record_manifest(_build_valid_manifest("train_002"))
        views = list_all_training_runs(self.registry)
        self.assertEqual(len(views), 2)
        self.assertEqual(sorted(v.training_run_id for v in views), ["train_001", "train_002"])

    def test_lineage_view_with_failure(self) -> None:
        m = _build_valid_manifest("train_001")
        self.registry.record_manifest(m)
        r = TrainingResultManifest(
            training_run_id="train_001",
            status="failed",
            started_at_utc="2026-04-17T00:00:00+00:00",
            finished_at_utc="2026-04-17T00:01:00+00:00",
            duration_seconds=60.0,
            trainer_code_version="abc123",
            failure={"type": "RuntimeError", "message": "OOM", "stage": "training"},
        )
        self.registry.record_result(r)
        view = build_lineage_view(self.registry, "train_001")
        self.assertEqual(view.status, "failed")
        self.assertEqual(view.failure_stage, "training")


class TestCLIScripts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_json(self, name: str, data: dict[str, Any]) -> Path:
        path = self.registry_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def test_record_manifest_cli(self) -> None:
        from autosr.rl.cli.record_manifest import build_parser, main

        manifest_path = self._write_json("manifest.json", _build_valid_manifest("train_cli_001").to_dict())
        parser = build_parser()
        args = parser.parse_args([f"--manifest={manifest_path}", f"--registry-dir={self.registry_dir}"])
        # Just verify parser builds; main writes to stdout/registry
        self.assertEqual(args.manifest, str(manifest_path))

    def test_record_result_cli(self) -> None:
        from autosr.rl.cli.record_result import build_parser

        result_path = self._write_json("result.json", _build_valid_result("train_cli_001").to_dict())
        parser = build_parser()
        args = parser.parse_args([f"--result={result_path}", f"--registry-dir={self.registry_dir}"])
        self.assertEqual(args.result, str(result_path))

    def test_record_eval_cli(self) -> None:
        from autosr.rl.cli.record_eval import build_parser

        eval_path = self._write_json("eval.json", _build_valid_eval("eval_cli_001", "train_cli_001").to_dict())
        parser = build_parser()
        args = parser.parse_args([f"--report={eval_path}", f"--registry-dir={self.registry_dir}"])
        self.assertEqual(args.report, str(eval_path))

    def test_show_lineage_cli(self) -> None:
        from autosr.rl.cli.show_lineage import build_parser

        parser = build_parser()
        args = parser.parse_args([f"--registry-dir={self.registry_dir}"])
        self.assertIsNone(args.training_run_id)

    def test_compat_module_entrypoint_exists_for_docs_command(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "autosr.rl.record_manifest", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Record a TrainingManifest", proc.stdout)


if __name__ == "__main__":
    unittest.main()

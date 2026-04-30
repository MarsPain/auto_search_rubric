from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from reward_harness.rl.comparison import (
    compare_artifacts,
    compare_runs,
    detect_anomalies,
    detect_regression,
    summarize_artifact,
)
from reward_harness.rl.data_models import (
    EvalReport,
    TrainingManifest,
    TrainingResultManifest,
)
from reward_harness.rl.registry import ExperimentRegistry


def _build_manifest(
    training_run_id: str = "train_001",
    artifact_id: str = "rm_001",
    dataset_version: str = "v1.0",
) -> TrainingManifest:
    return TrainingManifest(
        training_run_id=training_run_id,
        rm_artifact_id=artifact_id,
        search_session_id="session_001",
        rm_endpoint="http://localhost:8080",
        rm_deploy_id="deploy_001",
        dataset={
            "dataset_id": "ds_001",
            "dataset_version": dataset_version,
            "split": "train",
        },
        trainer={
            "project": "rl_project",
            "repo_url": "https://github.com/example/rl",
            "code_version": "abc123",
            "entrypoint": "python train.py",
        },
        trainer_config={"lr": 0.001},
        execution={
            "launcher": "local",
            "host": "gpu-node-1",
            "accelerator": "cuda",
            "num_workers": 4,
        },
        tags=["exp1"],
        notes="test run",
    )


def _build_result(
    training_run_id: str = "train_001",
    status: str = "succeeded",
    duration: float = 3600.0,
    failure: dict[str, Any] | None = None,
) -> TrainingResultManifest:
    return TrainingResultManifest(
        training_run_id=training_run_id,
        status=status,
        started_at_utc="2026-04-17T00:00:00+00:00",
        finished_at_utc="2026-04-17T01:00:00+00:00",
        duration_seconds=duration,
        trainer_code_version="abc123",
        output={"checkpoint_path": "/tmp/checkpoint", "log_path": "/tmp/log"},
        reward_summary={"mean": 0.8},
        training_summary={"final_loss": 0.1},
        failure=failure,
    )


def _build_eval(
    eval_run_id: str = "eval_001",
    training_run_id: str = "train_001",
    benchmark: str = "hellaswag",
    metrics: dict[str, float] | None = None,
    baseline: str = "",
) -> EvalReport:
    return EvalReport(
        eval_run_id=eval_run_id,
        training_run_id=training_run_id,
        benchmark={"name": benchmark, "version": "v1", "split": "val"},
        metrics=metrics if metrics is not None else {"accuracy": 0.75},
        artifacts={"report_path": "/tmp/report"},
        summary="good",
        comparison_baseline=baseline,
    )


class TestCompareRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _setup_two_runs(self) -> None:
        m1 = _build_manifest("train_001")
        m2 = _build_manifest("train_002")
        self.registry.record_manifest(m1)
        self.registry.record_manifest(m2)
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75, "f1": 0.70})
        )
        self.registry.record_eval(
            _build_eval("eval_002", "train_002", metrics={"accuracy": 0.80, "f1": 0.72})
        )

    def test_compare_runs_same_benchmark(self) -> None:
        self._setup_two_runs()
        comps = compare_runs(self.registry, "train_001", "train_002")
        self.assertEqual(len(comps), 1)
        comp = comps[0]
        self.assertEqual(comp.benchmark_name, "hellaswag")
        self.assertEqual(comp.run_a_id, "train_001")
        self.assertEqual(comp.run_b_id, "train_002")
        # Should have accuracy and f1
        names = [m.name for m in comp.metric_deltas]
        self.assertIn("accuracy", names)
        self.assertIn("f1", names)

    def test_compare_runs_direction(self) -> None:
        self._setup_two_runs()
        comps = compare_runs(self.registry, "train_001", "train_002")
        acc_delta = [m for m in comps[0].metric_deltas if m.name == "accuracy"][0]
        self.assertEqual(acc_delta.value_a, 0.75)
        self.assertEqual(acc_delta.value_b, 0.80)
        self.assertAlmostEqual(acc_delta.delta, 0.05)
        self.assertEqual(acc_delta.direction, "up")

    def test_compare_runs_filter_benchmark(self) -> None:
        self._setup_two_runs()
        comps = compare_runs(
            self.registry, "train_001", "train_002", benchmark_name="hellaswag"
        )
        self.assertEqual(len(comps), 1)
        comps = compare_runs(
            self.registry, "train_001", "train_002", benchmark_name="nonexistent"
        )
        self.assertEqual(len(comps), 0)

    def test_compare_runs_different_benchmark(self) -> None:
        m1 = _build_manifest("train_001")
        m2 = _build_manifest("train_002")
        self.registry.record_manifest(m1)
        self.registry.record_manifest(m2)
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval(
                "eval_001", "train_001", benchmark="bench_a", metrics={"acc": 0.5}
            )
        )
        self.registry.record_eval(
            _build_eval(
                "eval_002", "train_002", benchmark="bench_b", metrics={"acc": 0.6}
            )
        )
        comps = compare_runs(self.registry, "train_001", "train_002")
        self.assertEqual(len(comps), 0)

    def test_compare_runs_missing_eval(self) -> None:
        m1 = _build_manifest("train_001")
        m2 = _build_manifest("train_002")
        self.registry.record_manifest(m1)
        self.registry.record_manifest(m2)
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"acc": 0.5})
        )
        # train_002 has no eval
        comps = compare_runs(self.registry, "train_001", "train_002")
        self.assertEqual(len(comps), 0)

    def test_compare_runs_skips_non_numeric(self) -> None:
        m1 = _build_manifest("train_001")
        m2 = _build_manifest("train_002")
        self.registry.record_manifest(m1)
        self.registry.record_manifest(m2)
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval(
                "eval_001", "train_001", metrics={"accuracy": 0.75, "notes": "good"}
            )
        )
        self.registry.record_eval(
            _build_eval(
                "eval_002", "train_002", metrics={"accuracy": 0.80, "notes": "better"}
            )
        )
        comps = compare_runs(self.registry, "train_001", "train_002")
        names = [m.name for m in comps[0].metric_deltas]
        self.assertIn("accuracy", names)
        self.assertNotIn("notes", names)


class TestCompareArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_compare_artifacts_two(self) -> None:
        # Artifact rm_a: train_a1 succeeded
        self.registry.record_manifest(_build_manifest("train_a1", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_a1", duration=3600.0))
        self.registry.record_eval(
            _build_eval("eval_a1", "train_a1", metrics={"accuracy": 0.75})
        )

        # Artifact rm_b: train_b1 succeeded
        self.registry.record_manifest(_build_manifest("train_b1", artifact_id="rm_b"))
        self.registry.record_result(_build_result("train_b1", duration=7200.0))
        self.registry.record_eval(
            _build_eval("eval_b1", "train_b1", metrics={"accuracy": 0.80})
        )

        tables = compare_artifacts(self.registry, ["rm_a", "rm_b"])
        self.assertEqual(len(tables), 1)
        t = tables[0]
        self.assertEqual(t.benchmark_name, "hellaswag")
        self.assertEqual(t.metric_name, "accuracy")
        self.assertEqual(len(t.rows), 2)
        values = {aid: val for aid, val in t.rows}
        self.assertEqual(values["rm_a"], 0.75)
        self.assertEqual(values["rm_b"], 0.80)

    def test_compare_artifacts_filter_benchmark(self) -> None:
        self.registry.record_manifest(_build_manifest("train_a1", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_a1"))
        self.registry.record_eval(
            _build_eval("eval_a1", "train_a1", benchmark="math", metrics={"score": 0.9})
        )

        tables = compare_artifacts(self.registry, ["rm_a"], benchmark_name="math")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].benchmark_name, "math")

        tables = compare_artifacts(self.registry, ["rm_a"], benchmark_name="hellaswag")
        self.assertEqual(len(tables), 0)

    def test_compare_artifacts_no_succeeded(self) -> None:
        self.registry.record_manifest(_build_manifest("train_a1", artifact_id="rm_a"))
        self.registry.record_result(
            _build_result(
                "train_a1",
                status="failed",
                failure={
                    "type": "OOM",
                    "message": "out of memory",
                    "stage": "training",
                },
            )
        )
        tables = compare_artifacts(self.registry, ["rm_a"])
        self.assertEqual(len(tables), 0)

    def test_compare_artifacts_partial_metrics(self) -> None:
        self.registry.record_manifest(_build_manifest("train_a1", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_a1"))
        self.registry.record_eval(
            _build_eval("eval_a1", "train_a1", metrics={"accuracy": 0.75, "f1": 0.70})
        )

        self.registry.record_manifest(_build_manifest("train_b1", artifact_id="rm_b"))
        self.registry.record_result(_build_result("train_b1"))
        self.registry.record_eval(
            _build_eval("eval_b1", "train_b1", metrics={"accuracy": 0.80})
        )

        tables = compare_artifacts(self.registry, ["rm_a", "rm_b"])
        # accuracy should be present for both; f1 only for rm_a
        acc_table = [t for t in tables if t.metric_name == "accuracy"][0]
        self.assertEqual(len(acc_table.rows), 2)
        f1_table = [t for t in tables if t.metric_name == "f1"]
        self.assertEqual(len(f1_table), 1)
        self.assertEqual(f1_table[0].rows[0][0], "rm_a")


class TestDetectRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_detect_regression_explicit_baseline(self) -> None:
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"accuracy": 0.80})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"accuracy": 0.70})
        )

        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 1)
        s = signals[0]
        self.assertEqual(s.metric, "accuracy")
        self.assertEqual(s.severity, "critical")
        self.assertAlmostEqual(s.delta_pct, -12.5)

    def test_detect_regression_critical(self) -> None:
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"accuracy": 0.80})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"accuracy": 0.50})
        )

        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].severity, "critical")

    def test_detect_regression_no_regression(self) -> None:
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"accuracy": 0.80})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"accuracy": 0.81})
        )

        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 0)

    def test_detect_regression_threshold_boundary(self) -> None:
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"accuracy": 0.80})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"accuracy": 0.76})
        )

        # delta_pct = -5.0%, exactly at threshold
        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 1)

        # threshold 6.0 should not trigger
        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=6.0
        )
        self.assertEqual(len(signals), 0)

    def test_detect_regression_minimize_metric(self) -> None:
        # loss increases -> regression for minimize metrics
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"loss": 0.10})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"loss": 0.15})
        )

        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].metric, "loss")
        self.assertAlmostEqual(signals[0].delta_pct, 50.0)

    def test_detect_regression_loss_improvement(self) -> None:
        # loss decreases -> improvement, not regression
        self.registry.record_manifest(_build_manifest("train_base"))
        self.registry.record_result(_build_result("train_base"))
        self.registry.record_eval(
            _build_eval("eval_base", "train_base", metrics={"loss": 0.20})
        )

        self.registry.record_manifest(_build_manifest("train_new"))
        self.registry.record_result(_build_result("train_new"))
        self.registry.record_eval(
            _build_eval("eval_new", "train_new", metrics={"loss": 0.10})
        )

        signals = detect_regression(
            self.registry, "train_new", baseline_run_id="train_base", threshold_pct=5.0
        )
        self.assertEqual(len(signals), 0)

    def test_detect_regression_auto_baseline(self) -> None:
        self.registry.record_manifest(
            _build_manifest("train_001", dataset_version="v1.0")
        )
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.80})
        )

        self.registry.record_manifest(
            _build_manifest("train_002", dataset_version="v1.0")
        )
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval("eval_002", "train_002", metrics={"accuracy": 0.70})
        )

        # No explicit baseline -> auto-infer train_001
        signals = detect_regression(self.registry, "train_002", threshold_pct=5.0)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].baseline_run_id, "train_001")

    def test_detect_regression_no_eval(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(_build_result("train_001"))
        # No eval
        signals = detect_regression(
            self.registry, "train_001", baseline_run_id="train_base"
        )
        self.assertEqual(len(signals), 0)

    def test_detect_regression_no_baseline(self) -> None:
        self.registry.record_manifest(
            _build_manifest("train_001", dataset_version="v1.0")
        )
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.70})
        )
        # No other runs with same dataset_version
        signals = detect_regression(self.registry, "train_001", threshold_pct=5.0)
        self.assertEqual(len(signals), 0)


class TestDetectAnomalies(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_detect_anomalies_failed_run(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(
            _build_result(
                "train_001",
                status="failed",
                duration=0.0,
                failure={
                    "type": "OOM",
                    "message": "out of memory",
                    "stage": "training",
                },
            )
        )
        anomalies = detect_anomalies(self.registry)
        self.assertIn("train_001", anomalies)

    def test_detect_anomalies_canceled_run(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(
            TrainingResultManifest(
                training_run_id="train_001",
                status="canceled",
                started_at_utc="2026-04-17T00:00:00+00:00",
                finished_at_utc="2026-04-17T01:00:00+00:00",
                duration_seconds=1800.0,
                trainer_code_version="abc123",
                failure={
                    "type": "user",
                    "message": "user canceled",
                    "stage": "training",
                },
            )
        )
        anomalies = detect_anomalies(self.registry)
        self.assertIn("train_001", anomalies)

    def test_detect_anomalies_no_eval(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(_build_result("train_001"))
        anomalies = detect_anomalies(self.registry)
        self.assertIn("train_001", anomalies)

    def test_detect_anomalies_zero_duration(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(_build_result("train_001", duration=0.0))
        self.registry.record_eval(_build_eval("eval_001", "train_001"))
        anomalies = detect_anomalies(self.registry)
        self.assertIn("train_001", anomalies)

    def test_detect_anomalies_normal_run(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75})
        )
        anomalies = detect_anomalies(self.registry)
        self.assertNotIn("train_001", anomalies)


class TestSummarizeArtifact(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_summarize_artifact_basic(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_001", duration=3600.0))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75})
        )

        summary = summarize_artifact(self.registry, "rm_a")
        self.assertEqual(summary.artifact_id, "rm_a")
        self.assertEqual(summary.run_count, 1)
        self.assertEqual(summary.succeeded_count, 1)
        self.assertEqual(summary.failed_count, 0)
        self.assertEqual(summary.avg_duration, 3600.0)
        self.assertIn("hellaswag", summary.latest_evals_by_benchmark)
        self.assertEqual(
            summary.latest_evals_by_benchmark["hellaswag"]["accuracy"], 0.75
        )

    def test_summarize_artifact_multiple_runs(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_001", duration=3600.0))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75})
        )

        self.registry.record_manifest(_build_manifest("train_002", artifact_id="rm_a"))
        self.registry.record_result(
            _build_result(
                "train_002",
                status="failed",
                duration=1800.0,
                failure={
                    "type": "OOM",
                    "message": "out of memory",
                    "stage": "training",
                },
            )
        )

        summary = summarize_artifact(self.registry, "rm_a")
        self.assertEqual(summary.run_count, 2)
        self.assertEqual(summary.succeeded_count, 1)
        self.assertEqual(summary.failed_count, 1)
        self.assertEqual(summary.avg_duration, 2700.0)

    def test_summarize_artifact_latest_eval(self) -> None:
        # Later run should take precedence for "latest eval"
        self.registry.record_manifest(_build_manifest("train_001", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_001"))
        self.registry.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75})
        )

        self.registry.record_manifest(_build_manifest("train_002", artifact_id="rm_a"))
        self.registry.record_result(_build_result("train_002"))
        self.registry.record_eval(
            _build_eval("eval_002", "train_002", metrics={"accuracy": 0.85})
        )

        summary = summarize_artifact(self.registry, "rm_a")
        self.assertEqual(
            summary.latest_evals_by_benchmark["hellaswag"]["accuracy"], 0.85
        )

    def test_summarize_artifact_empty(self) -> None:
        summary = summarize_artifact(self.registry, "rm_nonexistent")
        self.assertEqual(summary.run_count, 0)
        self.assertEqual(summary.avg_duration, 0.0)


class TestRegistryQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_list_runs_by_artifact(self) -> None:
        self.registry.record_manifest(_build_manifest("train_a1", artifact_id="rm_a"))
        self.registry.record_manifest(_build_manifest("train_b1", artifact_id="rm_b"))
        self.registry.record_manifest(_build_manifest("train_a2", artifact_id="rm_a"))

        runs = self.registry.list_runs_by_artifact("rm_a")
        self.assertEqual(sorted(runs), ["train_a1", "train_a2"])

        runs = self.registry.list_runs_by_artifact("rm_b")
        self.assertEqual(runs, ["train_b1"])

        runs = self.registry.list_runs_by_artifact("rm_nonexistent")
        self.assertEqual(runs, [])

    def test_list_runs_by_dataset_version(self) -> None:
        self.registry.record_manifest(
            _build_manifest("train_001", dataset_version="v1.0")
        )
        self.registry.record_manifest(
            _build_manifest("train_002", dataset_version="v1.0")
        )
        self.registry.record_manifest(
            _build_manifest("train_003", dataset_version="v2.0")
        )

        runs = self.registry.list_runs_by_dataset_version("v1.0")
        self.assertEqual(sorted(runs), ["train_001", "train_002"])

        runs = self.registry.list_runs_by_dataset_version("v2.0")
        self.assertEqual(runs, ["train_003"])

    def test_list_runs_by_status(self) -> None:
        self.registry.record_manifest(_build_manifest("train_001"))
        self.registry.record_result(_build_result("train_001", status="succeeded"))

        self.registry.record_manifest(_build_manifest("train_002"))
        self.registry.record_result(
            _build_result(
                "train_002",
                status="failed",
                failure={
                    "type": "OOM",
                    "message": "out of memory",
                    "stage": "training",
                },
            )
        )

        self.registry.record_manifest(_build_manifest("train_003"))
        # No result -> not matched

        runs = self.registry.list_runs_by_status("succeeded")
        self.assertEqual(runs, ["train_001"])

        runs = self.registry.list_runs_by_status("failed")
        self.assertEqual(runs, ["train_002"])


class TestCLIIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self.tmpdir.name)
        self._populate_registry()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _populate_registry(self) -> None:
        reg = ExperimentRegistry(base_dir=self.registry_dir)
        reg.record_manifest(_build_manifest("train_001"))
        reg.record_result(_build_result("train_001"))
        reg.record_eval(
            _build_eval("eval_001", "train_001", metrics={"accuracy": 0.75})
        )

        reg.record_manifest(_build_manifest("train_002"))
        reg.record_result(_build_result("train_002"))
        reg.record_eval(
            _build_eval("eval_002", "train_002", metrics={"accuracy": 0.80})
        )

        reg.record_manifest(_build_manifest("train_003"))
        reg.record_result(
            _build_result(
                "train_003",
                status="failed",
                failure={
                    "type": "OOM",
                    "message": "out of memory",
                    "stage": "training",
                },
            )
        )

    def _run_cli(self, module: str, *args: str) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            "-m",
            module,
            "--registry-dir",
            str(self.registry_dir),
            *args,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

    def test_compare_runs_cli(self) -> None:
        result = self._run_cli(
            "reward_harness.rl.compare_runs",
            "--run-a",
            "train_001",
            "--run-b",
            "train_002",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["benchmark"], "hellaswag")

    def test_compare_artifacts_cli(self) -> None:
        result = self._run_cli(
            "reward_harness.rl.compare_artifacts", "--artifact-ids", "rm_001", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data), 1)

    def test_check_regression_cli(self) -> None:
        result = self._run_cli(
            "reward_harness.rl.check_regression",
            "--training-run-id",
            "train_001",
            "--baseline-run-id",
            "train_002",
            "--threshold-pct",
            "5.0",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        # train_001 (0.75) vs train_002 (0.80) -> accuracy decreased by 6.25%
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["metric"], "accuracy")

    def test_list_runs_cli(self) -> None:
        result = self._run_cli("reward_harness.rl.list_runs", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data), 3)

    def test_list_runs_filter_status(self) -> None:
        result = self._run_cli(
            "reward_harness.rl.list_runs", "--status", "failed", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["training_run_id"], "train_003")

    def test_list_runs_anomalies_only(self) -> None:
        result = self._run_cli(
            "reward_harness.rl.list_runs", "--anomalies-only", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        ids = [d["training_run_id"] for d in data]
        self.assertIn("train_003", ids)

    def test_show_lineage_with_baseline_delta(self) -> None:
        # Need evals with comparison_baseline set
        reg = ExperimentRegistry(base_dir=self.registry_dir)
        reg.record_eval(
            _build_eval(
                "eval_001b",
                "train_001",
                metrics={"accuracy": 0.75},
                baseline="train_002",
            )
        )
        result = self._run_cli(
            "reward_harness.rl.show_lineage",
            "--training-run-id",
            "train_001",
            "--with-baseline-delta",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertIn("baseline_deltas", data)
        self.assertEqual(len(data["baseline_deltas"]), 1)


if __name__ == "__main__":
    unittest.main()

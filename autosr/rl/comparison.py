from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data_models import EvalReport
from .registry import ExperimentRegistry


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Comparison of a single metric between two runs."""

    name: str
    value_a: float
    value_b: float
    delta: float
    delta_pct: float | None
    direction: str  # "up", "down", "flat", "na"


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Comparison of evaluation metrics between two training runs."""

    run_a_id: str
    run_b_id: str
    benchmark_name: str
    metric_deltas: list[MetricDelta]


@dataclass(frozen=True, slots=True)
class RegressionSignal:
    """A detected regression signal for a specific metric."""

    run_id: str
    baseline_run_id: str
    benchmark: str
    metric: str
    severity: str  # "critical", "warning", "info"
    current_value: float
    baseline_value: float
    delta_pct: float


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    """Aggregated statistics for an RM artifact across all its training runs."""

    artifact_id: str
    run_count: int
    succeeded_count: int
    failed_count: int
    canceled_count: int
    avg_duration: float
    latest_evals_by_benchmark: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ArtifactMetricTable:
    """Tabular comparison of a single metric across multiple artifacts."""

    benchmark_name: str
    metric_name: str
    rows: list[tuple[str, float]]  # (artifact_id, value)


def _metric_direction(metric_name: str) -> str:
    """Heuristic: return 'minimize' or 'maximize'."""
    lower_is_better_keywords = ("loss", "error", "latency", "time", "ppl", "perplexity")
    name_lower = metric_name.lower()
    if any(kw in name_lower for kw in lower_is_better_keywords):
        return "minimize"
    return "maximize"


def _compute_delta(value_a: float, value_b: float) -> tuple[float, float | None, str]:
    """Compute delta, delta_pct, and direction between two metric values."""
    delta = value_b - value_a
    if value_a == 0:
        delta_pct = None
    else:
        delta_pct = (delta / value_a) * 100.0

    if delta_pct is None:
        direction = "na"
    elif abs(delta_pct) < 0.01:
        direction = "flat"
    elif delta > 0:
        direction = "up"
    else:
        direction = "down"

    return delta, delta_pct, direction


def _get_evals_for_run(
    registry: ExperimentRegistry,
    training_run_id: str,
    benchmark_name: str | None = None,
) -> list[EvalReport]:
    """Get all eval reports for a training run, optionally filtered by benchmark."""
    eval_ids = registry.list_evals_for_training_run(training_run_id)
    evals = [registry.get_eval(eid) for eid in eval_ids]
    evals = [ev for ev in evals if ev is not None]
    if benchmark_name is not None:
        evals = [ev for ev in evals if ev.benchmark.get("name") == benchmark_name]
    return evals


def compare_runs(
    registry: ExperimentRegistry,
    run_a_id: str,
    run_b_id: str,
    benchmark_name: str | None = None,
) -> list[RunComparison]:
    """Compare evaluation metrics between two training runs.

    Returns a list of RunComparison objects, one per benchmark that both runs have
    evals for. If benchmark_name is specified, only compares that benchmark.
    """
    evals_a = _get_evals_for_run(registry, run_a_id, benchmark_name)
    evals_b = _get_evals_for_run(registry, run_b_id, benchmark_name)

    # Group by benchmark name
    a_by_benchmark: dict[str, EvalReport] = {}
    for ev in evals_a:
        name = ev.benchmark.get("name", "")
        if name:
            a_by_benchmark[name] = ev

    b_by_benchmark: dict[str, EvalReport] = {}
    for ev in evals_b:
        name = ev.benchmark.get("name", "")
        if name:
            b_by_benchmark[name] = ev

    results: list[RunComparison] = []
    for bench_name in sorted(set(a_by_benchmark.keys()) & set(b_by_benchmark.keys())):
        ev_a = a_by_benchmark[bench_name]
        ev_b = b_by_benchmark[bench_name]

        metric_deltas: list[MetricDelta] = []
        for metric_name in sorted(set(ev_a.metrics.keys()) & set(ev_b.metrics.keys())):
            val_a_raw = ev_a.metrics[metric_name]
            val_b_raw = ev_b.metrics[metric_name]
            try:
                val_a = float(val_a_raw)
                val_b = float(val_b_raw)
            except (TypeError, ValueError):
                continue

            delta, delta_pct, direction = _compute_delta(val_a, val_b)
            metric_deltas.append(
                MetricDelta(
                    name=metric_name,
                    value_a=val_a,
                    value_b=val_b,
                    delta=delta,
                    delta_pct=delta_pct,
                    direction=direction,
                )
            )

        if metric_deltas:
            results.append(
                RunComparison(
                    run_a_id=run_a_id,
                    run_b_id=run_b_id,
                    benchmark_name=bench_name,
                    metric_deltas=metric_deltas,
                )
            )

    return results


def compare_artifacts(
    registry: ExperimentRegistry,
    artifact_ids: list[str],
    benchmark_name: str | None = None,
) -> list[ArtifactMetricTable]:
    """Compare evaluation metrics across multiple RM artifacts.

    For each artifact, uses its latest succeeded run as the representative.
    Returns a list of ArtifactMetricTable objects, one per (benchmark, metric) pair.
    """
    # Find representative run for each artifact
    rep_runs: dict[str, str] = {}  # artifact_id -> training_run_id
    for artifact_id in artifact_ids:
        run_ids = registry.list_runs_by_artifact(artifact_id)
        # Filter to succeeded runs and pick the latest one
        succeeded = []
        for rid in run_ids:
            result = registry.get_result(rid)
            if result is not None and result.status == "succeeded":
                succeeded.append(rid)
        if succeeded:
            # Pick the latest by registry ordering (sorted list)
            rep_runs[artifact_id] = succeeded[-1]

    if not rep_runs:
        return []

    # Gather all evals from representative runs
    all_evals: list[tuple[str, str, EvalReport]] = []  # (artifact_id, run_id, eval)
    for artifact_id, run_id in rep_runs.items():
        evals = _get_evals_for_run(registry, run_id, benchmark_name)
        for ev in evals:
            all_evals.append((artifact_id, run_id, ev))

    # Build metric tables: group by (benchmark_name, metric_name)
    tables: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for artifact_id, _run_id, ev in all_evals:
        bench = ev.benchmark.get("name", "")
        if not bench:
            continue
        for metric_name, value_raw in ev.metrics.items():
            try:
                value = float(value_raw)
            except (TypeError, ValueError):
                continue
            key = (bench, metric_name)
            if key not in tables:
                tables[key] = []
            tables[key].append((artifact_id, value))

    # Sort and deduplicate (in case an artifact has multiple evals for same benchmark)
    result: list[ArtifactMetricTable] = []
    for (bench, metric_name), rows in sorted(tables.items()):
        # Keep only the latest eval per artifact (last in list due to iteration order)
        seen: set[str] = set()
        unique_rows: list[tuple[str, float]] = []
        for artifact_id, value in rows:
            if artifact_id not in seen:
                seen.add(artifact_id)
                unique_rows.append((artifact_id, value))
        result.append(
            ArtifactMetricTable(
                benchmark_name=bench,
                metric_name=metric_name,
                rows=unique_rows,
            )
        )

    return result


def _find_auto_baseline(
    registry: ExperimentRegistry,
    run_id: str,
    benchmark_name: str,
) -> str | None:
    """Find the most recent succeeded run with the same dataset_version and benchmark."""
    manifest = registry.get_manifest(run_id)
    if manifest is None:
        return None

    dataset_version = str(manifest.dataset.get("dataset_version", ""))
    if not dataset_version:
        return None

    # Get all runs with the same dataset_version
    candidate_ids = registry.list_runs_by_dataset_version(dataset_version)

    # Filter to succeeded runs that have evals for the same benchmark
    candidates = []
    for cid in candidate_ids:
        if cid == run_id:
            continue
        result = registry.get_result(cid)
        if result is None or result.status != "succeeded":
            continue
        evals = _get_evals_for_run(registry, cid, benchmark_name)
        if not evals:
            continue
        candidates.append(cid)

    if not candidates:
        return None

    # Return the last one (most recent by registry ordering)
    return candidates[-1]


def detect_regression(
    registry: ExperimentRegistry,
    run_id: str,
    baseline_run_id: str | None = None,
    threshold_pct: float = 5.0,
) -> list[RegressionSignal]:
    """Detect regression signals for a training run.

    If baseline_run_id is not provided, attempts to auto-infer the baseline
    from the most recent succeeded run with the same dataset_version and benchmark.

    A regression is triggered when a metric moves in the wrong direction by
    at least threshold_pct percent.
    """
    run_evals = _get_evals_for_run(registry, run_id)
    if not run_evals:
        return []

    signals: list[RegressionSignal] = []

    for ev in run_evals:
        bench_name = ev.benchmark.get("name", "")
        if not bench_name:
            continue

        # Determine baseline
        if baseline_run_id is not None:
            actual_baseline = baseline_run_id
        else:
            actual_baseline = _find_auto_baseline(registry, run_id, bench_name)

        if actual_baseline is None:
            continue

        baseline_evals = _get_evals_for_run(registry, actual_baseline, bench_name)
        if not baseline_evals:
            continue

        baseline_ev = baseline_evals[0]  # Take the first eval for this benchmark

        for metric_name, current_raw in ev.metrics.items():
            baseline_raw = baseline_ev.metrics.get(metric_name)
            if baseline_raw is None:
                continue

            try:
                current_val = float(current_raw)
                baseline_val = float(baseline_raw)
            except (TypeError, ValueError):
                continue

            delta, delta_pct, direction = _compute_delta(baseline_val, current_val)
            if delta_pct is None:
                continue

            direction_goal = _metric_direction(metric_name)

            is_regression = False
            if direction_goal == "maximize" and direction == "down" and abs(delta_pct) >= threshold_pct:
                is_regression = True
            elif direction_goal == "minimize" and direction == "up" and abs(delta_pct) >= threshold_pct:
                is_regression = True

            if not is_regression:
                continue

            # Determine severity
            if abs(delta_pct) >= threshold_pct * 2:
                severity = "critical"
            elif abs(delta_pct) >= threshold_pct:
                severity = "warning"
            else:
                severity = "info"

            signals.append(
                RegressionSignal(
                    run_id=run_id,
                    baseline_run_id=actual_baseline,
                    benchmark=bench_name,
                    metric=metric_name,
                    severity=severity,
                    current_value=current_val,
                    baseline_value=baseline_val,
                    delta_pct=delta_pct,
                )
            )

    return signals


def detect_anomalies(registry: ExperimentRegistry) -> list[str]:
    """Detect anomalous training runs.

    Returns a list of training_run_ids that exhibit anomalies:
    - Failed or canceled status
    - No eval reports
    - Empty eval metrics
    - Zero duration (possible data corruption)
    """
    anomalies: list[str] = []
    for run_id in registry.list_training_run_ids():
        result = registry.get_result(run_id)

        # Check for non-success status
        if result is not None and result.status != "succeeded":
            anomalies.append(run_id)
            continue

        # Check for zero duration
        if result is not None and result.duration_seconds == 0:
            anomalies.append(run_id)
            continue

        # Check evals
        eval_ids = registry.list_evals_for_training_run(run_id)
        if not eval_ids:
            anomalies.append(run_id)
            continue

        evals = [registry.get_eval(eid) for eid in eval_ids]
        evals = [ev for ev in evals if ev is not None]
        if not evals:
            anomalies.append(run_id)
            continue

        # Check for empty metrics in any eval
        has_empty_metrics = any(not ev.metrics for ev in evals)
        if has_empty_metrics:
            anomalies.append(run_id)
            continue

    return anomalies


def summarize_artifact(registry: ExperimentRegistry, artifact_id: str) -> ArtifactSummary:
    """Build an aggregated summary for an RM artifact."""
    run_ids = registry.list_runs_by_artifact(artifact_id)

    succeeded = 0
    failed = 0
    canceled = 0
    total_duration = 0.0
    durations_count = 0

    # Collect all evals to find latest per benchmark
    all_evals: list[tuple[str, EvalReport]] = []  # (training_run_id, eval)

    for run_id in run_ids:
        result = registry.get_result(run_id)
        if result is not None:
            total_duration += result.duration_seconds
            durations_count += 1
            if result.status == "succeeded":
                succeeded += 1
            elif result.status == "failed":
                failed += 1
            elif result.status == "canceled":
                canceled += 1

        eval_ids = registry.list_evals_for_training_run(run_id)
        for eid in eval_ids:
            ev = registry.get_eval(eid)
            if ev is not None:
                all_evals.append((run_id, ev))

    avg_duration = total_duration / durations_count if durations_count > 0 else 0.0

    # Find latest eval per benchmark
    # Use a simple heuristic: the eval from the latest run (last in run_ids)
    latest_evals_by_benchmark: dict[str, dict[str, Any]] = {}
    # Process in reverse to prioritize later runs
    for run_id, ev in reversed(all_evals):
        bench_name = ev.benchmark.get("name", "")
        if bench_name and bench_name not in latest_evals_by_benchmark:
            latest_evals_by_benchmark[bench_name] = dict(ev.metrics)

    return ArtifactSummary(
        artifact_id=artifact_id,
        run_count=len(run_ids),
        succeeded_count=succeeded,
        failed_count=failed,
        canceled_count=canceled,
        avg_duration=avg_duration,
        latest_evals_by_benchmark=latest_evals_by_benchmark,
    )

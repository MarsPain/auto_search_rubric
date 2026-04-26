from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import ExperimentRegistry


@dataclass(slots=True)
class LineageView:
    """Structured lineage view for a training run."""

    training_run_id: str
    status: str = "unknown"
    rm_artifact_id: str = ""
    rm_deploy_id: str = ""
    search_session_id: str = ""
    dataset_version: str = ""
    code_version: str = ""
    duration_seconds: float = 0.0
    eval_count: int = 0
    eval_benchmarks: list[str] = field(default_factory=list)
    upstream_chain: dict[str, str] = field(default_factory=dict)
    failure_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_run_id": self.training_run_id,
            "status": self.status,
            "rm_artifact_id": self.rm_artifact_id,
            "rm_deploy_id": self.rm_deploy_id,
            "search_session_id": self.search_session_id,
            "dataset_version": self.dataset_version,
            "code_version": self.code_version,
            "duration_seconds": self.duration_seconds,
            "eval_count": self.eval_count,
            "eval_benchmarks": list(self.eval_benchmarks),
            "upstream_chain": dict(self.upstream_chain),
            "failure_stage": self.failure_stage,
        }


def build_lineage_view(
    registry: ExperimentRegistry,
    training_run_id: str,
) -> LineageView | None:
    """Build a LineageView from registry data."""
    manifest = registry.get_manifest(training_run_id)
    if manifest is None:
        return None

    result = registry.get_result(training_run_id)
    evals = [registry.get_eval(eid) for eid in registry.list_evals_for_training_run(training_run_id)]
    evals = [ev for ev in evals if ev is not None]

    view = LineageView(
        training_run_id=training_run_id,
        rm_artifact_id=manifest.rm_artifact_id,
        rm_deploy_id=manifest.rm_deploy_id,
        search_session_id=manifest.search_session_id,
        dataset_version=str(manifest.dataset.get("dataset_version", "")),
        code_version=str(manifest.trainer.get("code_version", "")),
        upstream_chain={
            "rm_artifact_id": manifest.rm_artifact_id,
            "rm_deploy_id": manifest.rm_deploy_id,
            "search_session_id": manifest.search_session_id,
        },
    )

    if result is not None:
        view.status = result.status
        view.duration_seconds = result.duration_seconds
        if result.failure and result.status == "failed":
            view.failure_stage = str(result.failure.get("stage", ""))

    view.eval_count = len(evals)
    view.eval_benchmarks = sorted(
        {str(ev.benchmark.get("name", "")) for ev in evals if ev.benchmark.get("name")}
    )

    return view


def format_lineage_text(view: LineageView) -> str:
    """Format a LineageView as human-readable text."""
    lines: list[str] = [
        f"Training Run: {view.training_run_id}",
        f"  Status: {view.status}",
        f"  Duration: {view.duration_seconds:.1f}s",
        f"  RM Artifact: {view.rm_artifact_id}",
        f"  RM Deploy:   {view.rm_deploy_id or '(none)'}",
        f"  Search Session: {view.search_session_id}",
        f"  Dataset Version: {view.dataset_version}",
        f"  Code Version: {view.code_version}",
    ]
    if view.failure_stage:
        lines.append(f"  Failure Stage: {view.failure_stage}")
    lines.append(f"  Evaluations: {view.eval_count}")
    if view.eval_benchmarks:
        for bench in view.eval_benchmarks:
            lines.append(f"    - {bench}")
    return "\n".join(lines)


def list_all_training_runs(registry: ExperimentRegistry) -> list[LineageView]:
    """List lineage views for all recorded training runs."""
    views: list[LineageView] = []
    for run_id in registry.list_training_run_ids():
        view = build_lineage_view(registry, run_id)
        if view is not None:
            views.append(view)
    return views

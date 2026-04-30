from __future__ import annotations

"""RL integration metadata plane for AutoSR.

This package provides:
- Data models: TrainingManifest, TrainingResultManifest, EvalReport, LineageIndex
- Registry: ExperimentRegistry for append-only storage
- Validation: Structural and cross-reference validation
- Lineage: Query and view construction for training run lineage
"""

from .data_models import (
    EvalReport,
    LineageIndex,
    TrainingManifest,
    TrainingResultManifest,
    TrainingValidationError,
)
from .registry import DuplicateEntryError, ExperimentRegistry, MissingManifestError
from .comparison import (
    ArtifactMetricTable,
    ArtifactSummary,
    MetricDelta,
    RegressionSignal,
    RunComparison,
    compare_artifacts,
    compare_runs,
    detect_anomalies,
    detect_regression,
    summarize_artifact,
)
from .lineage import (
    LineageView,
    build_lineage_view,
    format_lineage_text,
    list_all_training_runs,
)

__all__ = [
    "ArtifactMetricTable",
    "ArtifactSummary",
    "EvalReport",
    "LineageIndex",
    "LineageView",
    "MetricDelta",
    "RegressionSignal",
    "RunComparison",
    "TrainingManifest",
    "TrainingResultManifest",
    "TrainingValidationError",
    "compare_artifacts",
    "compare_runs",
    "detect_anomalies",
    "detect_regression",
    "DuplicateEntryError",
    "ExperimentRegistry",
    "MissingManifestError",
    "summarize_artifact",
    "build_lineage_view",
    "format_lineage_text",
    "list_all_training_runs",
]

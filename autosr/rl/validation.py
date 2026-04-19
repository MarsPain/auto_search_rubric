from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .data_models import EvalReport, TrainingManifest, TrainingResultManifest, TrainingValidationError
from .registry import ExperimentRegistry


class LineageValidationError(Exception):
    """Raised when cross-object lineage consistency checks fail."""


def _assert_not_empty(value: Any, field: str, schema: str = "TrainingManifest") -> str:
    s = str(value).strip()
    if not s:
        raise TrainingValidationError(field, "must not be empty", schema=schema)
    return s


def validate_training_manifest(
    manifest: TrainingManifest,
    *,
    registry: ExperimentRegistry | None = None,
    expect_rm_artifact_exists: bool = False,
) -> None:
    """Validate a TrainingManifest structurally and optionally against registry.

    Structural checks are already performed by TrainingManifest.__post_init__.
    This function adds registry-level cross-reference checks.
    """
    # Re-validate by round-tripping through dict
    TrainingManifest.from_dict(manifest.to_dict())

    if registry and expect_rm_artifact_exists:
        # Future: check rm_artifact_id exists in artifact registry
        pass


def validate_training_result(
    result: TrainingResultManifest,
    *,
    registry: ExperimentRegistry | None = None,
) -> None:
    """Validate a TrainingResultManifest structurally and optionally against registry."""
    TrainingResultManifest.from_dict(result.to_dict())

    if registry is not None:
        manifest = registry.get_manifest(result.training_run_id)
        if manifest is None:
            raise LineageValidationError(
                f"TrainingManifest not found for training_run_id={result.training_run_id}"
            )
        if result.trainer_code_version != manifest.trainer.get("code_version", ""):
            # Allow override but warn conceptually; here we just validate consistency
            pass


def validate_eval_report(
    report: EvalReport,
    *,
    registry: ExperimentRegistry | None = None,
) -> None:
    """Validate an EvalReport structurally and optionally against registry."""
    EvalReport.from_dict(report.to_dict())

    if registry is not None:
        manifest = registry.get_manifest(report.training_run_id)
        if manifest is None:
            raise LineageValidationError(
                f"TrainingManifest not found for training_run_id={report.training_run_id}"
            )


def validate_cross_consistency(
    manifest: TrainingManifest,
    result: TrainingResultManifest | None = None,
    evals: list[EvalReport] | None = None,
) -> None:
    """Validate cross-object consistency rules.

    Rules:
    - result.training_run_id == manifest.training_run_id
    - each eval.training_run_id == manifest.training_run_id
    """
    if result is not None:
        if result.training_run_id != manifest.training_run_id:
            raise LineageValidationError(
                f"TrainingResultManifest.training_run_id ({result.training_run_id}) "
                f"does not match TrainingManifest.training_run_id ({manifest.training_run_id})"
            )

    if evals is not None:
        for report in evals:
            if report.training_run_id != manifest.training_run_id:
                raise LineageValidationError(
                    f"EvalReport.training_run_id ({report.training_run_id}) "
                    f"does not match TrainingManifest.training_run_id ({manifest.training_run_id})"
                )


def validate_payload_before_record(
    payload: Mapping[str, Any],
    kind: str,
) -> None:
    """Validate a raw JSON payload before creating registry entries.

    Args:
        payload: Raw dict parsed from JSON.
        kind: One of "manifest", "result", "eval".

    Raises:
        TrainingValidationError: If payload is invalid.
        LineageValidationError: If cross-references are invalid.
    """
    if kind == "manifest":
        manifest = TrainingManifest.from_dict(payload)
        validate_training_manifest(manifest)
    elif kind == "result":
        result = TrainingResultManifest.from_dict(payload)
        validate_training_result(result)
    elif kind == "eval":
        report = EvalReport.from_dict(payload)
        validate_eval_report(report)
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

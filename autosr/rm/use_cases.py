from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..data_models import Rubric
from ..harness.state import compute_config_hash
from .data_models import ArtifactValidationError, RMArtifact
from .io import load_search_output, save_rm_artifact


def _build_artifact_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"rm_{timestamp}"


def _build_config_hash(run_manifest: Mapping[str, Any] | None) -> str:
    if run_manifest is None:
        return "unknown_config_hash"
    hash_material = {
        "config_snapshot": run_manifest.get("config_snapshot", {}),
        "backend": run_manifest.get("backend", {}),
        "llm_snapshot": run_manifest.get("llm_snapshot", {}),
    }
    return compute_config_hash(hash_material)


def _extract_rubric_map(search_output: Mapping[str, Any]) -> dict[str, Rubric]:
    best_rubrics = search_output.get("best_rubrics", [])
    if not isinstance(best_rubrics, list) or not best_rubrics:
        raise ArtifactValidationError("best_rubrics", "search output missing best_rubrics entries")
    rubric_map: dict[str, Rubric] = {}
    for entry in best_rubrics:
        if not isinstance(entry, Mapping):
            raise ArtifactValidationError("best_rubrics", "entry must be an object")
        prompt_id = str(entry.get("prompt_id", "")).strip()
        rubric_payload = entry.get("rubric")
        if not prompt_id:
            raise ArtifactValidationError("best_rubrics.prompt_id", "must not be empty")
        if not isinstance(rubric_payload, Mapping):
            raise ArtifactValidationError("best_rubrics.rubric", "must be an object")
        rubric_map[prompt_id] = Rubric.from_dict(rubric_payload)
    return rubric_map


def build_rm_artifact(
    *,
    search_output_path: str | Path,
    artifact_id: str | None = None,
) -> RMArtifact:
    search_output = load_search_output(search_output_path)
    run_manifest = search_output.get("run_manifest")
    if run_manifest is not None and not isinstance(run_manifest, Mapping):
        raise ArtifactValidationError("run_manifest", "must be an object when present")

    dataset_hash = "unknown_dataset_hash"
    source_run_id = "unknown_run"
    source_session_id = "unknown_session"
    objective_snapshot: dict[str, Any] = {}
    search_snapshot: dict[str, Any] = {}

    if isinstance(run_manifest, Mapping):
        source_run_id = str(run_manifest.get("run_id", source_run_id))
        dataset = run_manifest.get("dataset", {})
        if isinstance(dataset, Mapping):
            dataset_hash = str(dataset.get("dataset_sha256", dataset_hash))
        harness = run_manifest.get("harness", {})
        if isinstance(harness, Mapping):
            source_session_id = str(harness.get("session_id", source_session_id))
        config_snapshot = run_manifest.get("config_snapshot", {})
        if isinstance(config_snapshot, Mapping):
            objective = config_snapshot.get("objective", {})
            if isinstance(objective, Mapping):
                objective_snapshot = dict(objective)
            search = config_snapshot.get("search", {})
            if isinstance(search, Mapping):
                search_snapshot = dict(search)

    search_diagnostics = search_output.get("search_diagnostics", {})
    if not isinstance(search_diagnostics, Mapping):
        search_diagnostics = {}

    artifact = RMArtifact(
        artifact_id=artifact_id or _build_artifact_id(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_session_id=source_session_id,
        source_run_id=source_run_id,
        dataset_hash=dataset_hash,
        config_hash=_build_config_hash(run_manifest if isinstance(run_manifest, Mapping) else None),
        rubric=_extract_rubric_map(search_output),
        scoring_policy={
            "policy_version": "1.0",
            "objective": objective_snapshot,
            "search": {"mode": search_snapshot.get("mode", search_diagnostics.get("mode", "unknown"))},
            "score_key": "best_objective_scores",
        },
        normalization={
            "method": "identity",
            "score_range": [0.0, 1.0],
        },
        compatibility={
            "artifact_type": "rm",
            "min_rm_api_version": "1.0",
            "output_schema": "reward_score_v1",
        },
    )
    return artifact


def validate_rm_artifact(
    artifact: RMArtifact,
    *,
    source_search_output_path: str | Path | None = None,
) -> None:
    # Trigger dataclass validations and schema constraints.
    RMArtifact.from_dict(artifact.to_dict())

    if source_search_output_path is None:
        return

    search_output = load_search_output(source_search_output_path)
    run_manifest = search_output.get("run_manifest")
    if not isinstance(run_manifest, Mapping):
        return

    dataset = run_manifest.get("dataset", {})
    if isinstance(dataset, Mapping):
        expected_dataset_hash = str(dataset.get("dataset_sha256", ""))
        if expected_dataset_hash and artifact.dataset_hash != expected_dataset_hash:
            raise ArtifactValidationError(
                "dataset_hash",
                f"artifact={artifact.dataset_hash} does not match source={expected_dataset_hash}",
            )

    expected_config_hash = _build_config_hash(run_manifest)
    if expected_config_hash and artifact.config_hash != expected_config_hash:
        raise ArtifactValidationError(
            "config_hash",
            f"artifact={artifact.config_hash} does not match source={expected_config_hash}",
        )

    source_rubrics = _extract_rubric_map(search_output)
    if set(source_rubrics.keys()) != set(artifact.rubric.keys()):
        raise ArtifactValidationError("rubric", "prompt_id set does not match source search output")
    for prompt_id, source_rubric in source_rubrics.items():
        target_rubric = artifact.rubric[prompt_id]
        if source_rubric.fingerprint() != target_rubric.fingerprint():
            raise ArtifactValidationError(
                "rubric",
                f"rubric mismatch for prompt_id={prompt_id}",
            )


def export_rm_artifact(
    *,
    search_output_path: str | Path,
    out_artifact_path: str | Path,
    artifact_id: str | None = None,
) -> Path:
    artifact = build_rm_artifact(
        search_output_path=search_output_path,
        artifact_id=artifact_id,
    )
    validate_rm_artifact(
        artifact,
        source_search_output_path=search_output_path,
    )
    return save_rm_artifact(out_artifact_path, artifact)

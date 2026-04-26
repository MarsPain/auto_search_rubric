from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

from ..config import LLMBackendConfig
from ..data_models import Rubric
from ..harness.state import compute_config_hash
from .data_models import ArtifactValidationError, DeployManifest, RMArtifact
from .io import (
    load_deploy_manifest,
    load_rm_artifact,
    load_search_output,
    save_deploy_manifest,
    save_rm_artifact,
)


def _build_artifact_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"rm_{timestamp}"


def _build_deploy_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"deploy_{timestamp}"


def _build_config_hash(run_manifest: Mapping[str, Any] | None) -> str:
    if run_manifest is None:
        return "unknown_config_hash"
    hash_material = {
        "config_snapshot": run_manifest.get("config_snapshot", {}),
        "backend": run_manifest.get("backend", {}),
        "llm_snapshot": run_manifest.get("llm_snapshot", {}),
    }
    return compute_config_hash(hash_material)


def _default_llm_runtime_snapshot() -> dict[str, Any]:
    defaults = LLMBackendConfig()
    default_model = defaults.default_model
    return {
        "base_url": defaults.base_url,
        "timeout": defaults.timeout,
        "max_retries": defaults.max_retries,
        "retry_backoff_base": defaults.retry_backoff_base,
        "retry_backoff_max": defaults.retry_backoff_max,
        "retry_jitter": defaults.retry_jitter,
        "fail_soft": defaults.fail_soft,
        "default_model": default_model,
        "verifier_model": defaults.verifier_model or default_model,
        "prompt_language": defaults.prompt_language,
    }


def _build_runtime_snapshot(run_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if run_manifest is None:
        return {}

    seed_raw = run_manifest.get("seed", 7)
    seed = int(seed_raw) if isinstance(seed_raw, (int, float, str)) else 7

    config_snapshot = run_manifest.get("config_snapshot", {})
    extraction_snapshot = {
        "strategy": "identity",
        "tag_name": "content",
        "pattern": None,
        "join_separator": "\n\n",
    }
    candidate_extraction_snapshot = {
        "strategy": "answer",
        "join_separator": "\n\n",
    }
    if isinstance(config_snapshot, Mapping):
        extraction = config_snapshot.get("extraction", {})
        if isinstance(extraction, Mapping):
            extraction_snapshot = {
                "strategy": str(extraction.get("strategy", "identity")),
                "tag_name": str(extraction.get("tag_name", "content")),
                "pattern": extraction.get("pattern"),
                "join_separator": str(extraction.get("join_separator", "\n\n")),
            }

        candidate_extraction = config_snapshot.get("candidate_extraction", {})
        if isinstance(candidate_extraction, Mapping):
            candidate_extraction_snapshot = {
                "strategy": str(candidate_extraction.get("strategy", "answer")),
                "join_separator": str(candidate_extraction.get("join_separator", "\n\n")),
            }

    llm_snapshot = run_manifest.get("llm_snapshot", {})
    llm_runtime_snapshot = _default_llm_runtime_snapshot()
    if isinstance(llm_snapshot, Mapping):
        default_model = str(llm_snapshot.get("default_model", llm_runtime_snapshot["default_model"]))
        llm_runtime_snapshot = {
            "base_url": str(llm_snapshot.get("base_url", llm_runtime_snapshot["base_url"])),
            "timeout": float(llm_snapshot.get("timeout", llm_runtime_snapshot["timeout"])),
            "max_retries": int(llm_snapshot.get("max_retries", llm_runtime_snapshot["max_retries"])),
            "retry_backoff_base": float(
                llm_snapshot.get("retry_backoff_base", llm_runtime_snapshot["retry_backoff_base"])
            ),
            "retry_backoff_max": float(
                llm_snapshot.get("retry_backoff_max", llm_runtime_snapshot["retry_backoff_max"])
            ),
            "retry_jitter": float(llm_snapshot.get("retry_jitter", llm_runtime_snapshot["retry_jitter"])),
            "fail_soft": bool(llm_snapshot.get("fail_soft", llm_runtime_snapshot["fail_soft"])),
            "default_model": default_model,
            "verifier_model": str(llm_snapshot.get("verifier_model", default_model)),
            "prompt_language": llm_snapshot.get("prompt_language"),
        }

    return {
        "seed": seed,
        "extraction": extraction_snapshot,
        "candidate_extraction": candidate_extraction_snapshot,
        "llm": llm_runtime_snapshot,
    }


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
        runtime_snapshot=_build_runtime_snapshot(
            run_manifest if isinstance(run_manifest, Mapping) else None
        ),
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


def _resolve_previous_artifact_id(
    *,
    out_dir: Path,
    deployment_target: str,
) -> str | None:
    if not out_dir.exists():
        return None

    latest_for_target: DeployManifest | None = None
    for manifest_path in sorted(out_dir.glob("*.json")):
        try:
            manifest = load_deploy_manifest(manifest_path)
        except (ArtifactValidationError, OSError):
            continue
        if manifest.deployment_target != deployment_target:
            continue
        if latest_for_target is None or manifest.deployed_at_utc > latest_for_target.deployed_at_utc:
            latest_for_target = manifest

    if latest_for_target is None:
        return None
    return latest_for_target.artifact_id


def record_deploy_manifest(
    *,
    artifact_path: str | Path,
    deployment_target: str,
    deployed_by: str | None = None,
    previous_artifact_id: str | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
    out_dir: str | Path = "artifacts/rm_deployments",
) -> Path:
    artifact_file = Path(artifact_path)
    artifact = load_rm_artifact(artifact_file)

    resolved_out_dir = Path(out_dir)
    resolved_previous_artifact = previous_artifact_id
    if resolved_previous_artifact is None:
        resolved_previous_artifact = _resolve_previous_artifact_id(
            out_dir=resolved_out_dir,
            deployment_target=deployment_target,
        )

    resolved_deployed_by = deployed_by or os.getenv("USER") or "unknown"
    resolved_rollback_policy = dict(rollback_policy or {"strategy": "rollback_to_previous_artifact"})

    manifest = DeployManifest(
        deploy_id=_build_deploy_id(),
        deployed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_id=artifact.artifact_id,
        artifact_path=str(artifact_file.resolve()),
        deployed_by=resolved_deployed_by,
        deployment_target=deployment_target,
        previous_artifact_id=resolved_previous_artifact,
        rollback_policy=resolved_rollback_policy,
        source_session_id=artifact.source_session_id,
        dataset_hash=artifact.dataset_hash,
        config_hash=artifact.config_hash,
    )
    target_path = resolved_out_dir / f"{manifest.deploy_id}.json"
    return save_deploy_manifest(target_path, manifest)

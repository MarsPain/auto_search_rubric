from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Mapping

from ..data_models import Rubric

DEPLOYMENT_TARGETS = frozenset({"dev", "staging", "prod"})
RUNTIME_SNAPSHOT_KEYS = frozenset({"seed", "extraction", "candidate_extraction", "llm"})
RUNTIME_LLM_KEYS = frozenset(
    {
        "base_url",
        "timeout",
        "max_retries",
        "retry_backoff_base",
        "retry_backoff_max",
        "retry_jitter",
        "fail_soft",
        "default_model",
        "verifier_model",
        "prompt_language",
    }
)


class ArtifactValidationError(Exception):
    """Raised when RM artifact data fails validation."""

    def __init__(self, field: str, message: str, *, schema: str = "RMArtifact") -> None:
        super().__init__(f"{schema} validation failed for '{field}': {message}")
        self.field = field
        self.validation_message = message


def _validate_runtime_snapshot(snapshot: Mapping[str, Any]) -> None:
    missing = sorted(RUNTIME_SNAPSHOT_KEYS - set(snapshot.keys()))
    if missing:
        raise ArtifactValidationError(
            "runtime_snapshot",
            f"missing required keys: {missing}",
        )

    seed = snapshot.get("seed")
    if not isinstance(seed, int):
        raise ArtifactValidationError("runtime_snapshot.seed", "must be an integer")

    extraction = snapshot.get("extraction")
    if not isinstance(extraction, Mapping):
        raise ArtifactValidationError("runtime_snapshot.extraction", "must be an object")

    candidate_extraction = snapshot.get("candidate_extraction")
    if not isinstance(candidate_extraction, Mapping):
        raise ArtifactValidationError(
            "runtime_snapshot.candidate_extraction",
            "must be an object",
        )

    llm = snapshot.get("llm")
    if not isinstance(llm, Mapping):
        raise ArtifactValidationError("runtime_snapshot.llm", "must be an object")

    missing_llm = sorted(RUNTIME_LLM_KEYS - set(llm.keys()))
    if missing_llm:
        raise ArtifactValidationError(
            "runtime_snapshot.llm",
            f"missing required keys: {missing_llm}",
        )


@dataclass(slots=True)
class RMArtifact:
    """Schema v1 for deployable reward-model artifacts."""

    artifact_id: str
    source_session_id: str
    source_run_id: str
    dataset_hash: str
    config_hash: str
    rubric: dict[str, Rubric]
    scoring_policy: dict[str, Any]
    normalization: dict[str, Any]
    compatibility: dict[str, Any]
    runtime_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ArtifactValidationError("artifact_id", "must not be empty")
        if not self.source_run_id.strip():
            raise ArtifactValidationError("source_run_id", "must not be empty")
        if not self.dataset_hash.strip():
            raise ArtifactValidationError("dataset_hash", "must not be empty")
        if not self.config_hash.strip():
            raise ArtifactValidationError("config_hash", "must not be empty")
        if not self.rubric:
            raise ArtifactValidationError("rubric", "must include at least one prompt rubric")
        if not isinstance(self.runtime_snapshot, dict):
            raise ArtifactValidationError("runtime_snapshot", "must be an object")
        if self.runtime_snapshot:
            _validate_runtime_snapshot(self.runtime_snapshot)
        if self.schema_version != "1.0":
            raise ArtifactValidationError(
                "schema_version",
                f"unsupported schema_version={self.schema_version}, expected 1.0",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at_utc": self.created_at_utc,
            "source_session_id": self.source_session_id,
            "source_run_id": self.source_run_id,
            "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash,
            "rubric": {
                prompt_id: prompt_rubric.to_dict()
                for prompt_id, prompt_rubric in self.rubric.items()
            },
            "scoring_policy": self.scoring_policy,
            "normalization": self.normalization,
            "compatibility": self.compatibility,
            "runtime_snapshot": self.runtime_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RMArtifact":
        schema_version = str(data.get("schema_version", "1.0"))
        if schema_version != "1.0":
            raise ArtifactValidationError(
                "schema_version",
                f"unsupported schema_version={schema_version}, expected 1.0",
            )
        rubric_raw = data.get("rubric", {})
        if not isinstance(rubric_raw, Mapping):
            raise ArtifactValidationError("rubric", "must be a mapping from prompt_id to rubric")
        runtime_snapshot_raw = data.get("runtime_snapshot", {})
        if not isinstance(runtime_snapshot_raw, Mapping):
            raise ArtifactValidationError("runtime_snapshot", "must be an object")
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            created_at_utc=str(data.get("created_at_utc", datetime.now(timezone.utc).isoformat())),
            source_session_id=str(data.get("source_session_id", "unknown")),
            source_run_id=str(data.get("source_run_id", "unknown_run")),
            dataset_hash=str(data.get("dataset_hash", "")),
            config_hash=str(data.get("config_hash", "")),
            rubric={
                str(prompt_id): Rubric.from_dict(rubric_payload)
                for prompt_id, rubric_payload in rubric_raw.items()
            },
            scoring_policy=dict(data.get("scoring_policy", {})),
            normalization=dict(data.get("normalization", {})),
            compatibility=dict(data.get("compatibility", {})),
            runtime_snapshot=dict(runtime_snapshot_raw),
            schema_version=schema_version,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "RMArtifact":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError("json", f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)


@dataclass(slots=True)
class DeployManifest:
    """Schema v1 for RM artifact deployment records."""

    deploy_id: str
    deployed_at_utc: str
    artifact_id: str
    artifact_path: str
    deployed_by: str
    deployment_target: str
    previous_artifact_id: str | None
    rollback_policy: dict[str, Any]
    source_session_id: str
    dataset_hash: str
    config_hash: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.deploy_id.strip():
            raise ArtifactValidationError("deploy_id", "must not be empty", schema="DeployManifest")
        if not self.deployed_at_utc.strip():
            raise ArtifactValidationError(
                "deployed_at_utc",
                "must not be empty",
                schema="DeployManifest",
            )
        if not self.artifact_id.strip():
            raise ArtifactValidationError("artifact_id", "must not be empty", schema="DeployManifest")
        if not self.artifact_path.strip():
            raise ArtifactValidationError("artifact_path", "must not be empty", schema="DeployManifest")
        if not self.deployed_by.strip():
            raise ArtifactValidationError("deployed_by", "must not be empty", schema="DeployManifest")
        if self.deployment_target not in DEPLOYMENT_TARGETS:
            raise ArtifactValidationError(
                "deployment_target",
                f"must be one of {sorted(DEPLOYMENT_TARGETS)}",
                schema="DeployManifest",
            )
        if self.previous_artifact_id == self.artifact_id:
            raise ArtifactValidationError(
                "previous_artifact_id",
                "must not equal artifact_id",
                schema="DeployManifest",
            )
        if not isinstance(self.rollback_policy, dict):
            raise ArtifactValidationError(
                "rollback_policy",
                "must be an object",
                schema="DeployManifest",
            )
        if not self.source_session_id.strip():
            raise ArtifactValidationError(
                "source_session_id",
                "must not be empty",
                schema="DeployManifest",
            )
        if not self.dataset_hash.strip():
            raise ArtifactValidationError("dataset_hash", "must not be empty", schema="DeployManifest")
        if not self.config_hash.strip():
            raise ArtifactValidationError("config_hash", "must not be empty", schema="DeployManifest")
        if self.schema_version != "1.0":
            raise ArtifactValidationError(
                "schema_version",
                f"unsupported schema_version={self.schema_version}, expected 1.0",
                schema="DeployManifest",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deploy_id": self.deploy_id,
            "deployed_at_utc": self.deployed_at_utc,
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "deployed_by": self.deployed_by,
            "deployment_target": self.deployment_target,
            "previous_artifact_id": self.previous_artifact_id,
            "rollback_policy": dict(self.rollback_policy),
            "source_session_id": self.source_session_id,
            "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeployManifest":
        schema_version = str(data.get("schema_version", "1.0"))
        rollback_policy = data.get("rollback_policy", {})
        if not isinstance(rollback_policy, Mapping):
            raise ArtifactValidationError(
                "rollback_policy",
                "must be an object",
                schema="DeployManifest",
            )
        previous_artifact_id_raw = data.get("previous_artifact_id")
        previous_artifact_id = (
            None
            if previous_artifact_id_raw is None
            else str(previous_artifact_id_raw)
        )
        return cls(
            deploy_id=str(data.get("deploy_id", "")),
            deployed_at_utc=str(data.get("deployed_at_utc", "")),
            artifact_id=str(data.get("artifact_id", "")),
            artifact_path=str(data.get("artifact_path", "")),
            deployed_by=str(data.get("deployed_by", "")),
            deployment_target=str(data.get("deployment_target", "")),
            previous_artifact_id=previous_artifact_id,
            rollback_policy=dict(rollback_policy),
            source_session_id=str(data.get("source_session_id", "")),
            dataset_hash=str(data.get("dataset_hash", "")),
            config_hash=str(data.get("config_hash", "")),
            schema_version=schema_version,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "DeployManifest":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "json",
                f"invalid JSON: {exc}",
                schema="DeployManifest",
            ) from exc
        return cls.from_dict(data)

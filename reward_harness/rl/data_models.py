from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Mapping

TRAINING_STATUS_SET = frozenset({"succeeded", "failed", "canceled"})
TRAINING_FAILURE_STAGES = frozenset({"preflight", "training", "eval", "unknown"})


class TrainingValidationError(Exception):
    """Raised when RL training manifest data fails validation."""

    def __init__(
        self, field: str, message: str, *, schema: str = "TrainingManifest"
    ) -> None:
        super().__init__(f"{schema} validation failed for '{field}': {message}")
        self.field = field
        self.validation_message = message


def _validate_url(value: str) -> bool:
    """Minimal URL validation: must have a scheme and netloc-ish part."""
    if not value:
        return False
    return value.startswith(("http://", "https://"))


@dataclass(slots=True)
class TrainingManifest:
    """Schema v1 for training run manifests (planned state)."""

    training_run_id: str
    rm_artifact_id: str
    search_session_id: str
    dataset: dict[str, Any]
    trainer: dict[str, Any]
    rm_endpoint: str = ""
    rm_deploy_id: str = ""
    trainer_config: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.training_run_id.strip():
            raise TrainingValidationError("training_run_id", "must not be empty")
        if not self.rm_artifact_id.strip():
            raise TrainingValidationError("rm_artifact_id", "must not be empty")
        if not self.search_session_id.strip():
            raise TrainingValidationError("search_session_id", "must not be empty")
        if self.rm_endpoint and not _validate_url(self.rm_endpoint):
            raise TrainingValidationError("rm_endpoint", "must be a valid HTTP(S) URL")
        if not isinstance(self.dataset, dict):
            raise TrainingValidationError("dataset", "must be an object")
        if not self.dataset.get("dataset_version"):
            raise TrainingValidationError(
                "dataset.dataset_version", "must not be empty"
            )
        if not isinstance(self.trainer, dict):
            raise TrainingValidationError("trainer", "must be an object")
        if not self.trainer.get("project"):
            raise TrainingValidationError("trainer.project", "must not be empty")
        if not self.trainer.get("code_version"):
            raise TrainingValidationError("trainer.code_version", "must not be empty")
        if not self.trainer.get("entrypoint"):
            raise TrainingValidationError("trainer.entrypoint", "must not be empty")
        if self.schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={self.schema_version}, expected 1.0",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_run_id": self.training_run_id,
            "created_at_utc": self.created_at_utc,
            "rm_artifact_id": self.rm_artifact_id,
            "rm_deploy_id": self.rm_deploy_id,
            "search_session_id": self.search_session_id,
            "rm_endpoint": self.rm_endpoint,
            "dataset": dict(self.dataset),
            "trainer": dict(self.trainer),
            "trainer_config": dict(self.trainer_config),
            "execution": dict(self.execution),
            "tags": list(self.tags),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingManifest":
        schema_version = str(data.get("schema_version", "1.0"))
        if schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={schema_version}, expected 1.0",
            )
        return cls(
            schema_version=schema_version,
            training_run_id=str(data.get("training_run_id", "")),
            created_at_utc=str(
                data.get("created_at_utc", datetime.now(timezone.utc).isoformat())
            ),
            rm_artifact_id=str(data.get("rm_artifact_id", "")),
            rm_deploy_id=str(data.get("rm_deploy_id", "")),
            search_session_id=str(data.get("search_session_id", "")),
            rm_endpoint=str(data.get("rm_endpoint", "")),
            dataset=dict(data.get("dataset", {})),
            trainer=dict(data.get("trainer", {})),
            trainer_config=dict(data.get("trainer_config", {})),
            execution=dict(data.get("execution", {})),
            tags=list(data.get("tags", [])),
            notes=str(data.get("notes", "")),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> TrainingManifest:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TrainingValidationError("json", f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)


@dataclass(slots=True)
class TrainingResultManifest:
    """Schema v1 for training run results (observed state)."""

    training_run_id: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    trainer_code_version: str
    output: dict[str, Any] = field(default_factory=dict)
    reward_summary: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.training_run_id.strip():
            raise TrainingValidationError(
                "training_run_id", "must not be empty", schema="TrainingResultManifest"
            )
        if self.status not in TRAINING_STATUS_SET:
            raise TrainingValidationError(
                "status",
                f"must be one of {sorted(TRAINING_STATUS_SET)}, got {self.status!r}",
                schema="TrainingResultManifest",
            )
        if not self.started_at_utc.strip():
            raise TrainingValidationError(
                "started_at_utc", "must not be empty", schema="TrainingResultManifest"
            )
        if not self.finished_at_utc.strip():
            raise TrainingValidationError(
                "finished_at_utc", "must not be empty", schema="TrainingResultManifest"
            )
        if self.duration_seconds < 0:
            raise TrainingValidationError(
                "duration_seconds", "must be >= 0", schema="TrainingResultManifest"
            )
        if self.status == "succeeded":
            if not self.training_summary:
                raise TrainingValidationError(
                    "training_summary",
                    "required when status is succeeded",
                    schema="TrainingResultManifest",
                )
            if not any(
                self.output.get(k)
                for k in ("checkpoint_path", "model_artifact_path", "log_path")
            ):
                raise TrainingValidationError(
                    "output",
                    "must contain at least one output path when status is succeeded",
                    schema="TrainingResultManifest",
                )
        if self.status == "failed":
            if not self.failure:
                raise TrainingValidationError(
                    "failure",
                    "required when status is failed",
                    schema="TrainingResultManifest",
                )
            if not self.failure.get("type"):
                raise TrainingValidationError(
                    "failure.type",
                    "required when status is failed",
                    schema="TrainingResultManifest",
                )
            if not self.failure.get("message"):
                raise TrainingValidationError(
                    "failure.message",
                    "required when status is failed",
                    schema="TrainingResultManifest",
                )
            if not self.failure.get("stage"):
                raise TrainingValidationError(
                    "failure.stage",
                    "required when status is failed",
                    schema="TrainingResultManifest",
                )
            if self.failure.get("stage") not in TRAINING_FAILURE_STAGES:
                raise TrainingValidationError(
                    "failure.stage",
                    f"must be one of {sorted(TRAINING_FAILURE_STAGES)}",
                    schema="TrainingResultManifest",
                )
        if self.status == "canceled":
            if not self.failure or not self.failure.get("message"):
                raise TrainingValidationError(
                    "failure.message",
                    "required when status is canceled (record cancellation reason/initiator)",
                    schema="TrainingResultManifest",
                )
        if self.schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={self.schema_version}, expected 1.0",
                schema="TrainingResultManifest",
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "training_run_id": self.training_run_id,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_seconds": self.duration_seconds,
            "trainer_code_version": self.trainer_code_version,
            "output": dict(self.output),
            "reward_summary": dict(self.reward_summary),
            "training_summary": dict(self.training_summary),
        }
        if self.failure is not None:
            result["failure"] = dict(self.failure)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingResultManifest":
        schema_version = str(data.get("schema_version", "1.0"))
        if schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={schema_version}, expected 1.0",
                schema="TrainingResultManifest",
            )
        failure_raw = data.get("failure")
        failure = None if failure_raw is None else dict(failure_raw)
        return cls(
            schema_version=schema_version,
            training_run_id=str(data.get("training_run_id", "")),
            status=str(data.get("status", "")),
            started_at_utc=str(data.get("started_at_utc", "")),
            finished_at_utc=str(data.get("finished_at_utc", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            trainer_code_version=str(data.get("trainer_code_version", "")),
            output=dict(data.get("output", {})),
            reward_summary=dict(data.get("reward_summary", {})),
            training_summary=dict(data.get("training_summary", {})),
            failure=failure,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "TrainingResultManifest":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TrainingValidationError(
                "json", f"invalid JSON: {exc}", schema="TrainingResultManifest"
            ) from exc
        return cls.from_dict(data)


@dataclass(slots=True)
class EvalReport:
    """Schema v1 for evaluation reports (comparison state)."""

    eval_run_id: str
    training_run_id: str
    benchmark: dict[str, Any]
    metrics: dict[str, Any]
    evaluated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    artifacts: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    comparison_baseline: str = ""
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.eval_run_id.strip():
            raise TrainingValidationError(
                "eval_run_id", "must not be empty", schema="EvalReport"
            )
        if not self.training_run_id.strip():
            raise TrainingValidationError(
                "training_run_id", "must not be empty", schema="EvalReport"
            )
        if not isinstance(self.benchmark, dict):
            raise TrainingValidationError(
                "benchmark", "must be an object", schema="EvalReport"
            )
        if not self.benchmark.get("name"):
            raise TrainingValidationError(
                "benchmark.name", "must not be empty", schema="EvalReport"
            )
        if not self.benchmark.get("version"):
            raise TrainingValidationError(
                "benchmark.version", "must not be empty", schema="EvalReport"
            )
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise TrainingValidationError(
                "metrics", "must be a non-empty object", schema="EvalReport"
            )
        if not self.evaluated_at_utc.strip():
            raise TrainingValidationError(
                "evaluated_at_utc", "must not be empty", schema="EvalReport"
            )
        if self.schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={self.schema_version}, expected 1.0",
                schema="EvalReport",
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "eval_run_id": self.eval_run_id,
            "training_run_id": self.training_run_id,
            "evaluated_at_utc": self.evaluated_at_utc,
            "benchmark": dict(self.benchmark),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "summary": self.summary,
        }
        if self.comparison_baseline:
            result["comparison_baseline"] = self.comparison_baseline
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalReport":
        schema_version = str(data.get("schema_version", "1.0"))
        if schema_version != "1.0":
            raise TrainingValidationError(
                "schema_version",
                f"unsupported schema_version={schema_version}, expected 1.0",
                schema="EvalReport",
            )
        return cls(
            schema_version=schema_version,
            eval_run_id=str(data.get("eval_run_id", "")),
            training_run_id=str(data.get("training_run_id", "")),
            evaluated_at_utc=str(
                data.get("evaluated_at_utc", datetime.now(timezone.utc).isoformat())
            ),
            benchmark=dict(data.get("benchmark", {})),
            metrics=dict(data.get("metrics", {})),
            artifacts=dict(data.get("artifacts", {})),
            summary=str(data.get("summary", "")),
            comparison_baseline=str(data.get("comparison_baseline", "")),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "EvalReport":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TrainingValidationError(
                "json", f"invalid JSON: {exc}", schema="EvalReport"
            ) from exc
        return cls.from_dict(data)


@dataclass(slots=True)
class LineageIndex:
    """Optional acceleration index for lineage queries.

    Not a source of truth; derived from manifests, results, and evals.
    """

    training_run_id: str
    rm_artifact_id: str
    rm_deploy_id: str
    search_session_id: str
    eval_run_ids: list[str] = field(default_factory=list)
    updated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_run_id": self.training_run_id,
            "rm_artifact_id": self.rm_artifact_id,
            "rm_deploy_id": self.rm_deploy_id,
            "search_session_id": self.search_session_id,
            "eval_run_ids": list(self.eval_run_ids),
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LineageIndex":
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            training_run_id=str(data.get("training_run_id", "")),
            rm_artifact_id=str(data.get("rm_artifact_id", "")),
            rm_deploy_id=str(data.get("rm_deploy_id", "")),
            search_session_id=str(data.get("search_session_id", "")),
            eval_run_ids=list(data.get("eval_run_ids", [])),
            updated_at_utc=str(
                data.get("updated_at_utc", datetime.now(timezone.utc).isoformat())
            ),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> "LineageIndex":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TrainingValidationError(
                "json", f"invalid JSON: {exc}", schema="LineageIndex"
            ) from exc
        return cls.from_dict(data)

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import (
    CandidateTextExtractionConfig,
    ContentExtractionConfig,
    LLMBackendConfig,
    RuntimeConfig,
)
from ..data_models import ResponseCandidate
from ..evaluator import RubricEvaluator
from ..factory import ComponentFactory
from .data_models import ArtifactValidationError, RMArtifact
from .io import load_rm_artifact
from .use_cases import validate_rm_artifact


class PromptNotFoundError(ValueError):
    """Raised when prompt_id does not exist in the loaded artifact."""


class RequestAuditLogger:
    """Structured request logger that writes to stdout and optional JSONL file."""

    def __init__(self, log_path: str | Path | None) -> None:
        self._log_path = None if log_path is None else Path(log_path)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        print(payload, flush=True)
        if self._log_path is None:
            return
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")


class RMScoringService:
    """Artifact-bound scoring service using the same evaluator path as search."""

    def __init__(self, *, artifact: RMArtifact, evaluator: RubricEvaluator) -> None:
        self._artifact = artifact
        self._evaluator = evaluator

    @property
    def artifact(self) -> RMArtifact:
        return self._artifact

    def score(
        self,
        *,
        prompt_id: str,
        prompt: str,
        candidate: ResponseCandidate,
    ) -> dict[str, Any]:
        rubric = self._artifact.rubric.get(prompt_id)
        if rubric is None:
            raise PromptNotFoundError(f"prompt_id={prompt_id} not found in artifact")

        evaluation = self._evaluator.evaluate_single_candidate(
            prompt_id=prompt_id,
            prompt=prompt,
            candidate=candidate,
            rubric=rubric,
            use_cache=False,
        )
        return {
            "artifact_id": self._artifact.artifact_id,
            "prompt_id": prompt_id,
            "candidate_id": candidate.candidate_id,
            "score": evaluation.score,
            "majority_grades": evaluation.majority_grades,
            "vote_scores": evaluation.vote_scores,
            "per_vote_grades": evaluation.per_vote_grades,
            "variance": evaluation.variance,
        }


@dataclass(slots=True)
class RMServerRuntime:
    artifact: RMArtifact
    scoring_service: RMScoringService
    request_logger: RequestAuditLogger


class CandidatePayload(BaseModel):
    candidate_id: str
    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreRequestPayload(BaseModel):
    prompt_id: str
    prompt: str
    candidate: CandidatePayload


class BatchScoreRequestPayload(BaseModel):
    items: list[ScoreRequestPayload]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latency_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000.0, 3)


def _require_runtime_snapshot(artifact: RMArtifact) -> dict[str, Any]:
    runtime_snapshot = artifact.runtime_snapshot
    if not runtime_snapshot:
        raise ArtifactValidationError(
            "runtime_snapshot",
            "missing runtime snapshot; export a Stage C artifact before starting RM server",
        )
    return runtime_snapshot


def _get_required_mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"runtime_snapshot.{key}", "must be an object")
    return value


def _build_runtime_config_from_snapshot(
    *,
    runtime_snapshot: dict[str, Any],
    api_key_env: str,
) -> tuple[RuntimeConfig, int]:
    seed = runtime_snapshot.get("seed")
    if not isinstance(seed, int):
        raise ArtifactValidationError("runtime_snapshot.seed", "must be an integer")

    extraction = _get_required_mapping(runtime_snapshot, "extraction")
    candidate_extraction = _get_required_mapping(runtime_snapshot, "candidate_extraction")
    llm = _get_required_mapping(runtime_snapshot, "llm")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")

    runtime_config = RuntimeConfig(
        backend="llm",
        llm=LLMBackendConfig(
            base_url=str(llm.get("base_url", "https://openrouter.ai/api/v1")),
            api_key=api_key,
            timeout=float(llm.get("timeout", 30.0)),
            max_retries=int(llm.get("max_retries", 2)),
            retry_backoff_base=float(llm.get("retry_backoff_base", 0.5)),
            retry_backoff_max=float(llm.get("retry_backoff_max", 8.0)),
            retry_jitter=float(llm.get("retry_jitter", 0.2)),
            fail_soft=bool(llm.get("fail_soft", False)),
            default_model=str(llm.get("default_model", "stepfun/step-3.5-flash:free")),
            verifier_model=str(llm.get("verifier_model", llm.get("default_model", "stepfun/step-3.5-flash:free"))),
            prompt_language=llm.get("prompt_language"),
        ),
        extraction=ContentExtractionConfig(
            strategy=str(extraction.get("strategy", "identity")),
            tag_name=str(extraction.get("tag_name", "content")),
            pattern=extraction.get("pattern"),
            join_separator=str(extraction.get("join_separator", "\n\n")),
        ),
        candidate_extraction=CandidateTextExtractionConfig(
            strategy=str(candidate_extraction.get("strategy", "answer")),
            join_separator=str(candidate_extraction.get("join_separator", "\n\n")),
        ),
    )
    return runtime_config, seed


def build_runtime_from_artifact(
    *,
    artifact_path: str | Path,
    api_key_env: str = "LLM_API_KEY",
    request_log_path: str | Path | None = "artifacts/rm_server_logs/requests.jsonl",
) -> RMServerRuntime:
    artifact = load_rm_artifact(artifact_path)
    validate_rm_artifact(artifact)
    runtime_snapshot = _require_runtime_snapshot(artifact)
    runtime_config, seed = _build_runtime_config_from_snapshot(
        runtime_snapshot=runtime_snapshot,
        api_key_env=api_key_env,
    )
    factory = ComponentFactory(runtime_config)
    verifier = factory.create_verifier_with_extraction()
    evaluator = RubricEvaluator(verifier, base_seed=seed)
    return RMServerRuntime(
        artifact=artifact,
        scoring_service=RMScoringService(artifact=artifact, evaluator=evaluator),
        request_logger=RequestAuditLogger(request_log_path),
    )


def create_app(runtime: RMServerRuntime) -> FastAPI:
    app = FastAPI(title="AutoSR RM Server", version="1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_id": runtime.artifact.artifact_id,
            "schema_version": runtime.artifact.schema_version,
        }

    @app.post("/score")
    def score(payload: ScoreRequestPayload) -> dict[str, Any]:
        start_time = time.perf_counter()
        request_id = uuid4().hex
        try:
            candidate = ResponseCandidate(
                candidate_id=payload.candidate.candidate_id,
                text=payload.candidate.text,
                source=payload.candidate.source,
                metadata=dict(payload.candidate.metadata),
            )
            score_payload = runtime.scoring_service.score(
                prompt_id=payload.prompt_id,
                prompt=payload.prompt,
                candidate=candidate,
            )
            latency_ms = _latency_ms(start_time)
            response = {
                "request_id": request_id,
                **score_payload,
                "latency_ms": latency_ms,
            }
            runtime.request_logger.emit(
                {
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                    "endpoint": "/score",
                    "artifact_id": runtime.artifact.artifact_id,
                    "prompt_id": payload.prompt_id,
                    "latency_ms": latency_ms,
                    "status": "ok",
                    "error_code": None,
                    "error_message": None,
                }
            )
            return response
        except PromptNotFoundError as exc:
            latency_ms = _latency_ms(start_time)
            runtime.request_logger.emit(
                {
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                    "endpoint": "/score",
                    "artifact_id": runtime.artifact.artifact_id,
                    "prompt_id": payload.prompt_id,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error_code": "prompt_not_found",
                    "error_message": str(exc),
                }
            )
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "prompt_not_found",
                    "error_message": str(exc),
                    "request_id": request_id,
                },
            ) from exc

    @app.post("/batch_score")
    def batch_score(payload: BatchScoreRequestPayload) -> dict[str, Any]:
        start_time = time.perf_counter()
        request_id = uuid4().hex
        results: list[dict[str, Any]] = []
        error_count = 0

        for index, item in enumerate(payload.items):
            item_start = time.perf_counter()
            try:
                candidate = ResponseCandidate(
                    candidate_id=item.candidate.candidate_id,
                    text=item.candidate.text,
                    source=item.candidate.source,
                    metadata=dict(item.candidate.metadata),
                )
                score_payload = runtime.scoring_service.score(
                    prompt_id=item.prompt_id,
                    prompt=item.prompt,
                    candidate=candidate,
                )
                results.append(
                    {
                        "index": index,
                        "ok": True,
                        **score_payload,
                        "latency_ms": _latency_ms(item_start),
                    }
                )
            except PromptNotFoundError as exc:
                error_count += 1
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "prompt_id": item.prompt_id,
                        "candidate_id": item.candidate.candidate_id,
                        "error_code": "prompt_not_found",
                        "error_message": str(exc),
                        "latency_ms": _latency_ms(item_start),
                    }
                )

        total_latency_ms = _latency_ms(start_time)
        status = "ok" if error_count == 0 else "partial_error"
        runtime.request_logger.emit(
            {
                "timestamp": _utc_now(),
                "request_id": request_id,
                "endpoint": "/batch_score",
                "artifact_id": runtime.artifact.artifact_id,
                "prompt_count": len(payload.items),
                "latency_ms": total_latency_ms,
                "status": status,
                "error_code": None if error_count == 0 else "partial_error",
                "error_message": None if error_count == 0 else f"errors={error_count}",
            }
        )
        return {
            "request_id": request_id,
            "artifact_id": runtime.artifact.artifact_id,
            "results": results,
            "latency_ms": total_latency_ms,
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start AutoSR RM server")
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to RM artifact JSON",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="Environment variable containing LLM API key",
    )
    parser.add_argument(
        "--request-log-path",
        default="artifacts/rm_server_logs/requests.jsonl",
        help="JSONL request log path",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        runtime = build_runtime_from_artifact(
            artifact_path=args.artifact,
            api_key_env=args.api_key_env,
            request_log_path=args.request_log_path,
        )
    except (ArtifactValidationError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        return

    app = create_app(runtime)
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

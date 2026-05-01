"""CLI entry point for the RM server.

All service implementation lives in :mod:`reward_harness.rm.service`.
This module re-exports the public API for backward compatibility.
"""

from __future__ import annotations

import argparse

from .service import (  # noqa: F401
    ArtifactValidationError,
    BatchScoreRequestPayload,
    CandidatePayload,
    PromptNotFoundError,
    RMScoringService,
    RMServerRuntime,
    RequestAuditLogger,
    ScoreRequestPayload,
    build_runtime_from_artifact,
    create_app,
)


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

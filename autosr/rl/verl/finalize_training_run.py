"""Finalize a training run and backfill results to the AutoSR registry.

This script:
1. Reads training run metadata and status
2. Builds a TrainingResultManifest
3. Optionally builds EvalReport(s)
4. Records them in the experiment registry

Example usage::

    # Success case
    python -m autosr.rl.verl.finalize_training_run \
      --training-run-id verl_20260419_101325 \
      --status succeeded \
      --started-at 2026-04-19T10:13:25+00:00 \
      --finished-at 2026-04-19T11:13:25+00:00 \
      --duration-seconds 3600 \
      --trainer-code-version abc123 \
      --checkpoint-path outputs/verl_20260419_101325/checkpoints/last \
      --log-path outputs/verl_20260419_101325/logs/training.log \
      --training-summary-json '{"final_loss": 0.12}' \
      --reward-summary-json '{"mean": 0.82, "std": 0.15}'

    # Failure case
    python -m autosr.rl.verl.finalize_training_run \
      --training-run-id verl_20260419_101325 \
      --status failed \
      --started-at 2026-04-19T10:13:25+00:00 \
      --finished-at 2026-04-19T10:15:00+00:00 \
      --duration-seconds 95 \
      --trainer-code-version abc123 \
      --failure-type OOMError \
      --failure-message "CUDA out of memory at step 42" \
      --failure-stage training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autosr.rl.data_models import EvalReport, TrainingResultManifest, TrainingValidationError
from autosr.rl.registry import DuplicateEntryError, ExperimentRegistry, MissingManifestError, RegistryError
from autosr.rl.validation import validate_payload_before_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize a training run and record results",
    )
    parser.add_argument(
        "--training-run-id",
        required=True,
        help="Training run ID",
    )
    parser.add_argument(
        "--status",
        required=True,
        choices=["succeeded", "failed", "canceled"],
        help="Final training status",
    )
    parser.add_argument(
        "--started-at",
        required=True,
        help="ISO timestamp when training started",
    )
    parser.add_argument(
        "--finished-at",
        required=True,
        help="ISO timestamp when training finished",
    )
    parser.add_argument(
        "--duration-seconds",
        required=True,
        type=float,
        help="Training duration in seconds",
    )
    parser.add_argument(
        "--trainer-code-version",
        required=True,
        help="Trainer code version used for this run",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Path to final checkpoint (optional)",
    )
    parser.add_argument(
        "--model-artifact-path",
        default="",
        help="Path to exported model artifact (optional)",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Path to training log (optional)",
    )
    parser.add_argument(
        "--training-summary-json",
        default="{}",
        help="JSON string with training summary metrics",
    )
    parser.add_argument(
        "--reward-summary-json",
        default="{}",
        help="JSON string with reward summary",
    )
    parser.add_argument(
        "--failure-type",
        default="",
        help="Failure exception type (required if status=failed)",
    )
    parser.add_argument(
        "--failure-message",
        default="",
        help="Failure message (required if status=failed or canceled)",
    )
    parser.add_argument(
        "--failure-stage",
        default="",
        choices=["", "preflight", "training", "eval", "unknown"],
        help="Stage where failure occurred (required if status=failed)",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/training_runs",
        help="Base directory for the experiment registry",
    )
    parser.add_argument(
        "--eval-report-json",
        default="",
        help="Path to EvalReport JSON file to record (optional)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Parse JSON payloads
    # ------------------------------------------------------------------
    try:
        training_summary: dict[str, Any] = json.loads(args.training_summary_json)
        reward_summary: dict[str, Any] = json.loads(args.reward_summary_json)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON: {exc}")
        return

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    output: dict[str, Any] = {}
    if args.checkpoint_path:
        output["checkpoint_path"] = args.checkpoint_path
    if args.model_artifact_path:
        output["model_artifact_path"] = args.model_artifact_path
    if args.log_path:
        output["log_path"] = args.log_path

    # ------------------------------------------------------------------
    # Build failure dict if needed
    # ------------------------------------------------------------------
    failure: dict[str, Any] | None = None
    if args.status == "failed":
        failure = {
            "type": args.failure_type,
            "message": args.failure_message,
            "stage": args.failure_stage or "unknown",
        }
    elif args.status == "canceled":
        failure = {
            "message": args.failure_message,
        }

    # ------------------------------------------------------------------
    # Build TrainingResultManifest
    # ------------------------------------------------------------------
    try:
        result = TrainingResultManifest(
            training_run_id=args.training_run_id,
            status=args.status,
            started_at_utc=args.started_at,
            finished_at_utc=args.finished_at,
            duration_seconds=args.duration_seconds,
            trainer_code_version=args.trainer_code_version,
            output=output,
            reward_summary=reward_summary,
            training_summary=training_summary,
            failure=failure,
        )
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    # ------------------------------------------------------------------
    # Record result in registry
    # ------------------------------------------------------------------
    registry = ExperimentRegistry(base_dir=args.registry_dir)
    try:
        validate_payload_before_record(result.to_dict(), kind="result")
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    try:
        recorded_path = registry.record_result(result)
    except MissingManifestError as exc:
        parser.error(str(exc))
        return
    except DuplicateEntryError as exc:
        parser.error(str(exc))
        return
    except RegistryError as exc:
        parser.error(str(exc))
        return

    print(f"Recorded result in registry: {recorded_path}")
    print(f"training_run_id: {args.training_run_id}")
    print(f"status: {args.status}")

    # ------------------------------------------------------------------
    # Optional: record EvalReport
    # ------------------------------------------------------------------
    if args.eval_report_json:
        eval_path = Path(args.eval_report_json)
        if not eval_path.exists():
            parser.error(f"EvalReport file not found: {eval_path}")
            return
        try:
            raw: dict[str, Any] = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON in eval report: {exc}")
            return
        try:
            validate_payload_before_record(raw, kind="eval")
            report = EvalReport.from_dict(raw)
        except TrainingValidationError as exc:
            parser.error(str(exc))
            return

        try:
            eval_recorded_path = registry.record_eval(report)
        except MissingManifestError as exc:
            parser.error(str(exc))
            return
        except DuplicateEntryError as exc:
            parser.error(str(exc))
            return
        except RegistryError as exc:
            parser.error(str(exc))
            return

        print(f"Recorded eval in registry: {eval_recorded_path}")
        print(f"eval_run_id: {report.eval_run_id}")


if __name__ == "__main__":
    main()

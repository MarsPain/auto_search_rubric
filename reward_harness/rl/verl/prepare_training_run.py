"""Prepare a training run for external RL repos (reference implementation).

This script:
1. Generates a training_run_id
2. Creates local output directories
3. Performs RM server healthz handshake
4. Builds and records a TrainingManifest

Example usage::

    python -m reward_harness.rl.verl.prepare_training_run \
      --rm-endpoint http://127.0.0.1:8080 \
      --rm-artifact-id artifact_001 \
      --rm-deploy-id deploy_001 \
      --search-session-id session_001 \
      --dataset-id gsm8k \
      --dataset-version v1.0 \
      --trainer-project verl_grpo \
      --trainer-code-version abc123 \
      --trainer-entrypoint "python3 -m verl.trainer.main_ppo" \
      --output-dir outputs/my_run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reward_harness.rl.data_models import TrainingManifest, TrainingValidationError
from reward_harness.rl.registry import (
    DuplicateEntryError,
    ExperimentRegistry,
    RegistryError,
)
from reward_harness.rl.validation import validate_payload_before_record
from reward_harness.rl.verl.reward_client import RMHealthzError, RMScoringClient


def generate_training_run_id(prefix: str = "verl") -> str:
    """Generate a timestamp-based training run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a training run and record its manifest",
    )
    parser.add_argument(
        "--rm-endpoint",
        required=True,
        help="RM server HTTP endpoint",
    )
    parser.add_argument(
        "--rm-artifact-id",
        required=True,
        help="Expected RM artifact ID",
    )
    parser.add_argument(
        "--rm-deploy-id",
        default="",
        help="RM deploy ID (optional)",
    )
    parser.add_argument(
        "--search-session-id",
        required=True,
        help="Source search session ID",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier",
    )
    parser.add_argument(
        "--dataset-version",
        required=True,
        help="Dataset version",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--trainer-project",
        required=True,
        help="Trainer project name",
    )
    parser.add_argument(
        "--trainer-repo-url",
        default="",
        help="Trainer repository URL (optional)",
    )
    parser.add_argument(
        "--trainer-code-version",
        required=True,
        help="Trainer code version (git commit or tag)",
    )
    parser.add_argument(
        "--trainer-entrypoint",
        required=True,
        help="Trainer entrypoint command",
    )
    parser.add_argument(
        "--training-run-id",
        default="",
        help="Training run ID (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local output directory for this training run",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/training_runs",
        help="Base directory for the experiment registry",
    )
    parser.add_argument(
        "--skip-healthz",
        action="store_true",
        help="Skip RM server healthz check",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags (optional)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form notes (optional)",
    )
    parser.add_argument(
        "--trainer-config-json",
        default="{}",
        help="JSON string with trainer config overrides",
    )
    parser.add_argument(
        "--execution-json",
        default='{"launcher": "local", "host": "localhost", "accelerator": "cuda", "num_workers": 1}',
        help="JSON string with execution context",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Generate run ID
    # ------------------------------------------------------------------
    training_run_id = args.training_run_id or generate_training_run_id()
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Create local output directories (reference layout)
    # ------------------------------------------------------------------
    run_dir = output_dir / training_run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "eval"
    manifests_dir = run_dir / "manifests"

    for d in (checkpoints_dir, logs_dir, eval_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # RM server handshake
    # ------------------------------------------------------------------
    if not args.skip_healthz:
        client = RMScoringClient(
            endpoint=args.rm_endpoint,
            expected_artifact_id=args.rm_artifact_id,
            expected_source_session_id=args.search_session_id,
        )
        try:
            healthz = client.healthz_check()
        except RMHealthzError as exc:
            parser.error(f"RM server healthz failed: {exc}")
            return
        print(f"RM server healthz ok: artifact_id={healthz.get('artifact_id')}")
    else:
        print("Skipped RM server healthz check")

    # ------------------------------------------------------------------
    # Build TrainingManifest
    # ------------------------------------------------------------------
    try:
        trainer_config: dict[str, Any] = json.loads(args.trainer_config_json)
        execution: dict[str, Any] = json.loads(args.execution_json)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON: {exc}")
        return

    manifest = TrainingManifest(
        training_run_id=training_run_id,
        rm_artifact_id=args.rm_artifact_id,
        rm_deploy_id=args.rm_deploy_id,
        search_session_id=args.search_session_id,
        rm_endpoint=args.rm_endpoint,
        dataset={
            "dataset_id": args.dataset_id,
            "dataset_version": args.dataset_version,
            "split": args.dataset_split,
        },
        trainer={
            "project": args.trainer_project,
            "repo_url": args.trainer_repo_url,
            "code_version": args.trainer_code_version,
            "entrypoint": args.trainer_entrypoint,
        },
        trainer_config=trainer_config,
        execution=execution,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        notes=args.notes,
    )

    # ------------------------------------------------------------------
    # Record manifest locally and in registry
    # ------------------------------------------------------------------
    local_manifest_path = manifests_dir / f"{training_run_id}.training.json"
    local_manifest_path.write_text(manifest.to_json(indent=2), encoding="utf-8")
    print(f"Wrote local manifest: {local_manifest_path}")

    registry = ExperimentRegistry(base_dir=args.registry_dir)
    try:
        validate_payload_before_record(manifest.to_dict(), kind="manifest")
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    try:
        recorded_path = registry.record_manifest(manifest)
    except DuplicateEntryError as exc:
        parser.error(str(exc))
        return
    except RegistryError as exc:
        parser.error(str(exc))
        return

    print(f"Recorded manifest in registry: {recorded_path}")
    print(f"training_run_id: {training_run_id}")
    print(f"output_dir: {run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autosr.rl.data_models import TrainingManifest, TrainingValidationError
from autosr.rl.registry import DuplicateEntryError, ExperimentRegistry, RegistryError
from autosr.rl.validation import validate_payload_before_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a TrainingManifest into the experiment registry",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to TrainingManifest JSON file",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/training_runs",
        help="Base directory for the experiment registry",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow idempotent replay for identical payload with same training_run_id",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"Manifest file not found: {manifest_path}")
        return

    try:
        raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON: {exc}")
        return

    try:
        validate_payload_before_record(raw, kind="manifest")
        manifest = TrainingManifest.from_dict(raw)
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    registry = ExperimentRegistry(base_dir=args.registry_dir)
    try:
        recorded_path = registry.record_manifest(
            manifest,
            allow_identical_overwrite=args.force,
        )
    except DuplicateEntryError as exc:
        parser.error(str(exc))
        return
    except RegistryError as exc:
        parser.error(str(exc))
        return

    print(f"Recorded manifest: {recorded_path}")
    print(f"training_run_id: {manifest.training_run_id}")
    print(f"rm_artifact_id: {manifest.rm_artifact_id}")
    print(f"search_session_id: {manifest.search_session_id}")


if __name__ == "__main__":
    main()

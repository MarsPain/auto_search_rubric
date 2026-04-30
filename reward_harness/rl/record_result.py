from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reward_harness.rl.data_models import (
    TrainingResultManifest,
    TrainingValidationError,
)
from reward_harness.rl.registry import (
    DuplicateEntryError,
    ExperimentRegistry,
    MissingManifestError,
    RegistryError,
)
from reward_harness.rl.validation import validate_payload_before_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a TrainingResultManifest into the experiment registry",
    )
    parser.add_argument(
        "--result",
        required=True,
        help="Path to TrainingResultManifest JSON file",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/training_runs",
        help="Base directory for the experiment registry",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        parser.error(f"Result file not found: {result_path}")
        return

    try:
        raw: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON: {exc}")
        return

    try:
        validate_payload_before_record(raw, kind="result")
        result = TrainingResultManifest.from_dict(raw)
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    registry = ExperimentRegistry(base_dir=args.registry_dir)
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

    print(f"Recorded result: {recorded_path}")
    print(f"training_run_id: {result.training_run_id}")
    print(f"status: {result.status}")
    print(f"duration_seconds: {result.duration_seconds}")


if __name__ == "__main__":
    main()

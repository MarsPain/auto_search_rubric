from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autosr.rl.data_models import EvalReport, TrainingValidationError
from autosr.rl.registry import (
    DuplicateEntryError,
    ExperimentRegistry,
    MissingManifestError,
    RegistryError,
)
from autosr.rl.validation import validate_payload_before_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record an EvalReport into the experiment registry",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to EvalReport JSON file",
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

    report_path = Path(args.report)
    if not report_path.exists():
        parser.error(f"Report file not found: {report_path}")
        return

    try:
        raw: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON: {exc}")
        return

    try:
        validate_payload_before_record(raw, kind="eval")
        report = EvalReport.from_dict(raw)
    except TrainingValidationError as exc:
        parser.error(str(exc))
        return

    registry = ExperimentRegistry(base_dir=args.registry_dir)
    try:
        recorded_path = registry.record_eval(report)
    except MissingManifestError as exc:
        parser.error(str(exc))
        return
    except DuplicateEntryError as exc:
        parser.error(str(exc))
        return
    except RegistryError as exc:
        parser.error(str(exc))
        return

    print(f"Recorded eval: {recorded_path}")
    print(f"eval_run_id: {report.eval_run_id}")
    print(f"training_run_id: {report.training_run_id}")
    print(f"benchmark: {report.benchmark.get('name')} v{report.benchmark.get('version')}")


if __name__ == "__main__":
    main()

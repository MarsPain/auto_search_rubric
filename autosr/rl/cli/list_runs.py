from __future__ import annotations

import argparse
import json

from autosr.rl.comparison import detect_anomalies
from autosr.rl.lineage import build_lineage_view
from autosr.rl.registry import ExperimentRegistry, RegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List training runs with optional filtering",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="Filter by RM artifact ID",
    )
    parser.add_argument(
        "--status",
        default=None,
        choices=["succeeded", "failed", "canceled"],
        help="Filter by training result status",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Filter by dataset version",
    )
    parser.add_argument(
        "--anomalies-only",
        action="store_true",
        help="Only show anomalous runs",
    )
    parser.add_argument(
        "--registry-dir",
        default="artifacts/training_runs",
        help="Base directory for the experiment registry",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        registry = ExperimentRegistry(base_dir=args.registry_dir)
    except RegistryError as exc:
        parser.error(str(exc))
        return

    # Determine candidate run IDs
    if args.artifact:
        run_ids = registry.list_runs_by_artifact(args.artifact)
    elif args.status:
        run_ids = registry.list_runs_by_status(args.status)
    elif args.dataset_version:
        run_ids = registry.list_runs_by_dataset_version(args.dataset_version)
    else:
        run_ids = registry.list_training_run_ids()

    if args.anomalies_only:
        anomaly_ids = set(detect_anomalies(registry))
        run_ids = [rid for rid in run_ids if rid in anomaly_ids]

    if not run_ids:
        print("No matching training runs found.")
        return

    views = []
    for rid in run_ids:
        view = build_lineage_view(registry, rid)
        if view is not None:
            views.append(view)

    if args.json:
        data = [v.to_dict() for v in views]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for view in views:
            status_marker = ""
            if view.status == "failed":
                status_marker = " [FAILED]"
            elif view.status == "canceled":
                status_marker = " [CANCELED]"
            print(f"{view.training_run_id}{status_marker}")
            print(
                f"  artifact={view.rm_artifact_id}"
                f" dataset={view.dataset_version} code={view.code_version}"
            )
            print(f"  duration={view.duration_seconds:.1f}s evals={view.eval_count}")
            if view.eval_benchmarks:
                print(f"  benchmarks={', '.join(view.eval_benchmarks)}")
            print()


if __name__ == "__main__":
    main()

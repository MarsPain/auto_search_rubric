from __future__ import annotations

import argparse
import json

from reward_harness.rl.comparison import compare_artifacts
from reward_harness.rl.registry import ExperimentRegistry, RegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across RM artifacts",
    )
    parser.add_argument(
        "--artifact-ids",
        required=True,
        nargs="+",
        help="List of artifact IDs to compare",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Filter to a specific benchmark name",
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

    tables = compare_artifacts(registry, args.artifact_ids, args.benchmark)
    if not tables:
        print("No comparable evals found for the given artifacts.")
        return

    if args.json:
        data = []
        for t in tables:
            data.append(
                {
                    "benchmark": t.benchmark_name,
                    "metric": t.metric_name,
                    "values": {aid: val for aid, val in t.rows},
                }
            )
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for t in tables:
            print(f"Benchmark: {t.benchmark_name} | Metric: {t.metric_name}")
            for artifact_id, value in t.rows:
                print(f"  {artifact_id}: {value}")
            print()


if __name__ == "__main__":
    main()

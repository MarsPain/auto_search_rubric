from __future__ import annotations

import argparse
import json

from autosr.rl.comparison import compare_runs
from autosr.rl.registry import ExperimentRegistry, RegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics between two training runs",
    )
    parser.add_argument(
        "--run-a",
        required=True,
        help="First training run ID",
    )
    parser.add_argument(
        "--run-b",
        required=True,
        help="Second training run ID",
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

    comparisons = compare_runs(registry, args.run_a, args.run_b, args.benchmark)
    if not comparisons:
        print("No comparable evals found between the two runs.")
        return

    if args.json:
        data = []
        for comp in comparisons:
            data.append(
                {
                    "benchmark": comp.benchmark_name,
                    "run_a": comp.run_a_id,
                    "run_b": comp.run_b_id,
                    "metrics": [
                        {
                            "name": m.name,
                            "value_a": m.value_a,
                            "value_b": m.value_b,
                            "delta": m.delta,
                            "delta_pct": m.delta_pct,
                            "direction": m.direction,
                        }
                        for m in comp.metric_deltas
                    ],
                }
            )
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for comp in comparisons:
            print(f"Benchmark: {comp.benchmark_name}")
            print(f"  {comp.run_a_id} vs {comp.run_b_id}")
            for m in comp.metric_deltas:
                pct_str = f"{m.delta_pct:+.1f}%" if m.delta_pct is not None else "N/A"
                print(
                    f"    {m.name}: {m.value_a} -> {m.value_b}"
                    f" ({m.delta:+.4f}, {pct_str}) [{m.direction}]"
                )
            print()


if __name__ == "__main__":
    main()

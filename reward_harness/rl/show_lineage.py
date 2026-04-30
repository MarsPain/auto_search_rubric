from __future__ import annotations

import argparse
import json
from typing import Any

from reward_harness.rl.comparison import compare_runs
from reward_harness.rl.lineage import (
    build_lineage_view,
    format_lineage_text,
    list_all_training_runs,
)
from reward_harness.rl.registry import ExperimentRegistry, RegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show lineage for training runs",
    )
    parser.add_argument(
        "--training-run-id",
        default=None,
        help="Specific training run ID to query (default: list all)",
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
    parser.add_argument(
        "--with-baseline-delta",
        action="store_true",
        help="Show metric deltas against comparison_baseline evals",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    registry = ExperimentRegistry(base_dir=args.registry_dir)

    if args.training_run_id:
        try:
            view = build_lineage_view(registry, args.training_run_id)
        except RegistryError as exc:
            parser.error(str(exc))
            return
        if view is None:
            parser.error(f"Training run not found: {args.training_run_id}")
            return
        if args.json:
            data = view.to_dict()
            if args.with_baseline_delta:
                data["baseline_deltas"] = _build_baseline_deltas(
                    registry, args.training_run_id
                )
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            text = format_lineage_text(view)
            if args.with_baseline_delta:
                delta_text = _format_baseline_deltas(registry, args.training_run_id)
                if delta_text:
                    text = text + "\n" + delta_text
            print(text)
    else:
        try:
            views = list_all_training_runs(registry)
        except RegistryError as exc:
            parser.error(str(exc))
            return
        if args.json:
            data = [v.to_dict() for v in views]
            if args.with_baseline_delta:
                for item in data:
                    item["baseline_deltas"] = _build_baseline_deltas(
                        registry, item["training_run_id"]
                    )
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            if not views:
                print("No training runs recorded.")
                return
            for view in views:
                text = format_lineage_text(view)
                if args.with_baseline_delta:
                    delta_text = _format_baseline_deltas(registry, view.training_run_id)
                    if delta_text:
                        text = text + "\n" + delta_text
                print(text)
                print()


def _build_baseline_deltas(
    registry: ExperimentRegistry, training_run_id: str
) -> list[dict[str, Any]]:
    """Build baseline delta data for a training run's evals."""
    eval_ids = registry.list_evals_for_training_run(training_run_id)
    deltas: list[dict[str, Any]] = []
    for eid in eval_ids:
        ev = registry.get_eval(eid)
        if ev is None or not ev.comparison_baseline:
            continue
        comparisons = compare_runs(
            registry, ev.comparison_baseline, training_run_id, ev.benchmark.get("name")
        )
        for comp in comparisons:
            for m in comp.metric_deltas:
                deltas.append(
                    {
                        "baseline_run_id": ev.comparison_baseline,
                        "benchmark": comp.benchmark_name,
                        "metric": m.name,
                        "baseline_value": m.value_a,
                        "current_value": m.value_b,
                        "delta": m.delta,
                        "delta_pct": m.delta_pct,
                        "direction": m.direction,
                    }
                )
    return deltas


def _format_baseline_deltas(registry: ExperimentRegistry, training_run_id: str) -> str:
    """Format baseline deltas as human-readable text."""
    deltas = _build_baseline_deltas(registry, training_run_id)
    if not deltas:
        return ""
    lines = ["  Baseline Deltas:"]
    for d in deltas:
        pct_str = f"{d['delta_pct']:+.1f}%" if d["delta_pct"] is not None else "N/A"
        lines.append(
            f"    {d['benchmark']}/{d['metric']}:"
            f" {d['baseline_value']} -> {d['current_value']}"
            f" ({d['delta']:+.4f}, {pct_str})"
            f" [baseline={d['baseline_run_id']}]"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

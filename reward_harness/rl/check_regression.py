from __future__ import annotations

import argparse
import json

from reward_harness.rl.comparison import detect_regression
from reward_harness.rl.registry import ExperimentRegistry, RegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check for regression signals in a training run",
    )
    parser.add_argument(
        "--training-run-id",
        required=True,
        help="Training run ID to check",
    )
    parser.add_argument(
        "--baseline-run-id",
        default=None,
        help="Explicit baseline run ID (auto-inferred if omitted)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Regression threshold in percent (default: 5.0)",
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

    signals = detect_regression(
        registry,
        args.training_run_id,
        baseline_run_id=args.baseline_run_id,
        threshold_pct=args.threshold_pct,
    )
    if not signals:
        print("No regression signals detected.")
        return

    if args.json:
        data = [
            {
                "run_id": s.run_id,
                "baseline_run_id": s.baseline_run_id,
                "benchmark": s.benchmark,
                "metric": s.metric,
                "severity": s.severity,
                "current_value": s.current_value,
                "baseline_value": s.baseline_value,
                "delta_pct": s.delta_pct,
            }
            for s in signals
        ]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Detected {len(signals)} regression signal(s):")
        for s in signals:
            direction = "↓" if s.delta_pct < 0 else "↑"
            print(
                f"  [{s.severity.upper()}] {s.benchmark}/{s.metric}: "
                f"{s.baseline_value} -> {s.current_value} "
                f"({direction}{abs(s.delta_pct):.1f}%) "
                f"baseline={s.baseline_run_id}"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autosr.rl.lineage import build_lineage_view, format_lineage_text, list_all_training_runs
from autosr.rl.registry import ExperimentRegistry, RegistryError


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
            print(json.dumps(view.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_lineage_text(view))
    else:
        try:
            views = list_all_training_runs(registry)
        except RegistryError as exc:
            parser.error(str(exc))
            return
        if args.json:
            print(json.dumps([v.to_dict() for v in views], indent=2, ensure_ascii=False))
        else:
            if not views:
                print("No training runs recorded.")
                return
            for view in views:
                print(format_lineage_text(view))
                print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .data_models import ArtifactValidationError
from .io import load_rm_artifact
from .use_cases import export_rm_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deployable RM artifact from reward_harness search output",
    )
    parser.add_argument(
        "--search-output",
        required=True,
        help="Path to reward_harness search output JSON",
    )
    parser.add_argument(
        "--out-artifact",
        required=True,
        help="Path to write RM artifact JSON",
    )
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="Optional explicit artifact id (default: auto-generated)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output_path = export_rm_artifact(
            search_output_path=args.search_output,
            out_artifact_path=args.out_artifact,
            artifact_id=args.artifact_id,
        )
    except ArtifactValidationError as exc:
        parser.error(str(exc))
        return
    artifact = load_rm_artifact(output_path)
    print(f"Exported RM artifact: {Path(output_path)}")
    print(f"artifact_id: {artifact.artifact_id}")
    print(f"source_run_id: {artifact.source_run_id}")
    print(f"source_session_id: {artifact.source_session_id}")


if __name__ == "__main__":
    main()

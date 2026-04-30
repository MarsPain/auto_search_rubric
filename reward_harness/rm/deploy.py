from __future__ import annotations

import argparse
from pathlib import Path

from .data_models import ArtifactValidationError
from .io import load_deploy_manifest
from .use_cases import record_deploy_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record RM artifact deployment manifest",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to RM artifact JSON",
    )
    parser.add_argument(
        "--deployment-target",
        required=True,
        choices=["dev", "staging", "prod"],
        help="Deployment target environment",
    )
    parser.add_argument(
        "--deployed-by",
        default=None,
        help="Deployer identity (defaults to $USER or 'unknown')",
    )
    parser.add_argument(
        "--previous-artifact-id",
        default=None,
        help="Override previous artifact id (default: auto-resolve by target)",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/rm_deployments",
        help="Directory to store deploy manifests",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output_path = record_deploy_manifest(
            artifact_path=args.artifact,
            deployment_target=args.deployment_target,
            deployed_by=args.deployed_by,
            previous_artifact_id=args.previous_artifact_id,
            out_dir=args.out_dir,
        )
    except (ArtifactValidationError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return

    manifest = load_deploy_manifest(output_path)
    print(f"Recorded deploy manifest: {Path(output_path)}")
    print(f"deploy_id: {manifest.deploy_id}")
    print(f"artifact_id: {manifest.artifact_id}")
    print(f"previous_artifact_id: {manifest.previous_artifact_id}")
    print(f"deployment_target: {manifest.deployment_target}")


if __name__ == "__main__":
    main()

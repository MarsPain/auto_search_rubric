from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io_utils import atomic_write_text
from .data_models import DeployManifest, RMArtifact


def load_search_output(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_rm_artifact(path: str | Path) -> RMArtifact:
    file_path = Path(path)
    return RMArtifact.from_json(file_path.read_text(encoding="utf-8"))


def load_deploy_manifest(path: str | Path) -> DeployManifest:
    file_path = Path(path)
    return DeployManifest.from_json(file_path.read_text(encoding="utf-8"))


def _save_json_payload(path: str | Path, payload: str) -> Path:
    return atomic_write_text(path, payload)


def save_rm_artifact(path: str | Path, artifact: RMArtifact) -> Path:
    return _save_json_payload(path, artifact.to_json(indent=2))


def save_deploy_manifest(path: str | Path, manifest: DeployManifest) -> Path:
    return _save_json_payload(path, manifest.to_json(indent=2))

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .data_models import RMArtifact


def load_search_output(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_rm_artifact(path: str | Path) -> RMArtifact:
    file_path = Path(path)
    return RMArtifact.from_json(file_path.read_text(encoding="utf-8"))


def save_rm_artifact(path: str | Path, artifact: RMArtifact) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = artifact.to_json(indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f"{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        temp_path.replace(file_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return file_path

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .data_models import EvalReport, LineageIndex, TrainingManifest, TrainingResultManifest


def load_training_manifest(path: str | Path) -> TrainingManifest:
    file_path = Path(path)
    return TrainingManifest.from_json(file_path.read_text(encoding="utf-8"))


def load_training_result(path: str | Path) -> TrainingResultManifest:
    file_path = Path(path)
    return TrainingResultManifest.from_json(file_path.read_text(encoding="utf-8"))


def load_eval_report(path: str | Path) -> EvalReport:
    file_path = Path(path)
    return EvalReport.from_json(file_path.read_text(encoding="utf-8"))


def load_lineage_index(path: str | Path) -> LineageIndex:
    file_path = Path(path)
    return LineageIndex.from_json(file_path.read_text(encoding="utf-8"))


def _save_json_payload(path: str | Path, payload: str) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

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


def save_training_manifest(path: str | Path, manifest: TrainingManifest) -> Path:
    return _save_json_payload(path, manifest.to_json(indent=2))


def save_training_result(path: str | Path, result: TrainingResultManifest) -> Path:
    return _save_json_payload(path, result.to_json(indent=2))


def save_eval_report(path: str | Path, report: EvalReport) -> Path:
    return _save_json_payload(path, report.to_json(indent=2))


def save_lineage_index(path: str | Path, index: LineageIndex) -> Path:
    return _save_json_payload(path, index.to_json(indent=2))

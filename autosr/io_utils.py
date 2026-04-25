from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .data_models import PromptExample, Rubric


def load_dataset(path: str | Path) -> list[PromptExample]:
    """Load dataset from JSON file.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of PromptExample objects.
    """
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    prompts_raw = raw.get("prompts", [])
    return [PromptExample.from_dict(item) for item in prompts_raw]


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write text via a same-directory temp file and atomic replace."""
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
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        temp_path.replace(file_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return file_path


def atomic_write_json(path: str | Path, data: Any, *, indent: int | None = 2) -> Path:
    """Serialize JSON using the repository's stable UTF-8 output format."""
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    return atomic_write_text(path, payload)


def load_initial_rubrics(path: str | Path) -> dict[str, Rubric]:
    """Load initial rubrics from JSON file.

    Supports multiple formats:
    - best_rubrics format: {"best_rubrics": [{"prompt_id": "p1", "rubric": {...}}, ...]}
    - rubrics format: {"rubrics": [{"prompt_id": "p1", "rubric": {...}}, ...]}
    - Direct format: {"p1": {...rubric...}, "p2": {...rubric...}}

    Args:
        path: Path to the JSON file containing rubrics.

    Returns:
        Dict mapping prompt_id to Rubric objects.
    """
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))

    result: dict[str, Rubric] = {}

    # Try best_rubrics format (output format from save_rubrics)
    if "best_rubrics" in raw:
        for item in raw["best_rubrics"]:
            prompt_id = item.get("prompt_id")
            rubric_data = item.get("rubric")
            if prompt_id and rubric_data:
                result[prompt_id] = Rubric.from_dict(rubric_data)
        return result

    # Try rubrics format (array of prompt_id + rubric pairs)
    if "rubrics" in raw:
        for item in raw["rubrics"]:
            prompt_id = item.get("prompt_id")
            rubric_data = item.get("rubric")
            if prompt_id and rubric_data:
                result[prompt_id] = Rubric.from_dict(rubric_data)
        return result

    # Try direct format: {prompt_id: rubric_dict, ...}
    for key, value in raw.items():
        if isinstance(value, dict) and "criteria" in value:
            result[key] = Rubric.from_dict(value)

    return result


def save_rubrics(
    path: str | Path,
    best_rubrics: dict[str, Rubric],
    best_scores: dict[str, float],
    *,
    best_candidates: dict[str, str] | None = None,
    candidate_scores: dict[str, dict[str, float]] | None = None,
    run_manifest: dict[str, Any] | None = None,
    search_diagnostics: dict[str, Any] | None = None,
) -> None:
    """Save rubrics and scores to JSON file.

    Args:
        path: Output file path.
        best_rubrics: Dict mapping prompt_id to best Rubric.
        best_scores: Dict mapping prompt_id to objective score.
        best_candidates: Optional dict mapping prompt_id to best candidate_id.
        candidate_scores: Optional dict mapping prompt_id to dict of candidate_id -> score.
        run_manifest: Optional reproducibility metadata for this run.
        search_diagnostics: Optional search diagnostics emitted by the searcher.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    best_rubrics_list = []
    best_objective_scores: dict[str, float] = {}
    for prompt_id in sorted(best_rubrics.keys()):
        rubric = best_rubrics[prompt_id]
        objective_score = best_scores.get(prompt_id, 0.0)
        best_objective_scores[prompt_id] = objective_score
        entry: dict[str, Any] = {
            "prompt_id": prompt_id,
            "rubric": rubric.to_dict(),
            "score": objective_score,
        }
        if best_candidates and prompt_id in best_candidates:
            entry["best_candidate_id"] = best_candidates[prompt_id]
        if candidate_scores and prompt_id in candidate_scores:
            entry["candidate_scores"] = candidate_scores[prompt_id]
        best_rubrics_list.append(entry)

    output = {
        "best_rubrics": best_rubrics_list,
        "best_objective_scores": best_objective_scores,
        # Legacy alias kept for compatibility with existing downstream parsers.
        "best_scores": dict(best_objective_scores),
    }
    if run_manifest is not None:
        output["run_manifest"] = run_manifest
    if search_diagnostics is not None:
        output["search_diagnostics"] = search_diagnostics
    atomic_write_json(file_path, output)


def save_run_record_files(
    path: str | Path,
    *,
    run_manifest: dict[str, Any],
    reproducible_script: str,
) -> tuple[Path, Path]:
    """Archive per-run reproducibility files.

    Files are written under `<output_parent>/run_records/` and include:
    - a manifest JSON snapshot (`*.manifest.json`)
    - an executable shell script (`*.reproduce.sh`)

    Returns:
        Tuple of (manifest_path, script_path).
    """
    file_path = Path(path)
    run_records_dir = file_path.parent / "run_records"
    run_records_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(run_manifest.get("run_id", "unknown_run"))
    base_name = file_path.stem
    manifest_path = run_records_dir / f"{base_name}_{run_id}.manifest.json"
    script_path = run_records_dir / f"{base_name}_{run_id}.reproduce.sh"

    atomic_write_json(manifest_path, run_manifest)
    atomic_write_text(script_path, reproducible_script)
    script_path.chmod(0o755)
    return manifest_path, script_path

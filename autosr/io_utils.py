from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PromptExample, Rubric


def load_dataset(path: str | Path) -> list[PromptExample]:
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    prompts_raw = raw.get("prompts", [])
    return [PromptExample.from_dict(item) for item in prompts_raw]




def load_initial_rubrics(path: str | Path) -> dict[str, Rubric]:
    """Load preset initial rubrics from JSON.

    Supported formats:
    - {"best_rubrics": {"prompt_id": {rubric}}}
    - {"rubrics": {"prompt_id": {rubric}}}
    - {"rubric": {rubric}} or a direct rubric object with `criteria`
      (mapped to special key "__default__").
    """
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "best_rubrics" in raw:
        source = raw["best_rubrics"]
    elif isinstance(raw, dict) and "rubrics" in raw:
        source = raw["rubrics"]
    elif isinstance(raw, dict) and "rubric" in raw:
        source = {"__default__": raw["rubric"]}
    elif isinstance(raw, dict) and "criteria" in raw:
        source = {"__default__": raw}
    else:
        raise ValueError(
            "unsupported preset rubric format; expected best_rubrics/rubrics/rubric or direct rubric object"
        )

    if not isinstance(source, dict):
        raise ValueError("preset rubric payload must be an object mapping")

    return {str(prompt_id): Rubric.from_dict(rubric_raw) for prompt_id, rubric_raw in source.items()}

def save_rubrics(
    path: str | Path,
    rubrics: dict[str, Rubric],
    scores: dict[str, float],
    best_candidates: dict[str, str] | None = None,
    candidate_scores: dict[str, dict[str, float]] | None = None,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "best_rubrics": {
            prompt_id: rubric.to_dict() for prompt_id, rubric in rubrics.items()
        },
        "best_objective_scores": scores,
        "best_scores": scores,
    }
    if best_candidates is not None:
        payload["best_candidates"] = best_candidates
    if candidate_scores is not None:
        payload["candidate_scores"] = candidate_scores
        payload["best_candidate_scores"] = {
            prompt_id: max(scores_for_prompt.values()) if scores_for_prompt else 0.0
            for prompt_id, scores_for_prompt in candidate_scores.items()
        }
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

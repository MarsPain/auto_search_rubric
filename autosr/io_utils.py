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


def save_rubrics(path: str | Path, rubrics: dict[str, Rubric], scores: dict[str, float]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "best_rubrics": {
            prompt_id: rubric.to_dict() for prompt_id, rubric in rubrics.items()
        },
        "best_scores": scores,
    }
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


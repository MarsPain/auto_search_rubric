#!/usr/bin/env python3
"""
Convert benchmark_all.jsonl to the project's standard initial data format.

Input: data/writing/benchmark_all.jsonl
  Each line is a JSON object with fields:
    - index, domain1, domain2, lang, query, checklist
  Each checklist contains 5 criteria with fields:
    - name, criteria_description, 1-2, 3-4, 5-6, 7-8, 9-10

Outputs:
  1. writing_prompts.json — standard prompts format (compatible with
     examples/call_summary_dataset_with_rank.json).
     NOTE: candidates are empty because the raw data does not include
     response candidates. You must populate candidates (at least 2 per
     prompt) before feeding this dataset into training or search.
  2. writing_rubrics.json — initial rubrics derived from checklists
     (compatible with examples/call_summary_initial_rubric.json).

Usage:
    uv run python scripts/data_process/convert_benchmark.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _slugify(text: str) -> str:
    """Convert criterion name to a valid snake_case id.

    For English text: lowercase + underscore separation.
    For CJK text: keeps CJK characters as-is (they are valid Unicode
    word characters), strips punctuation and whitespace.
    """
    # Remove punctuation, keep letters/numbers/CJK
    cleaned = re.sub(r"[^\w\s]", "", text)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned.lower()[:40]


def _build_rubric_text(name: str, description: str, scores: dict[str, str]) -> str:
    """Compose criterion text from name, description and score-level details."""
    lines = [f"{name}: {description}", ""]
    for band in ("1-2", "3-4", "5-6", "7-8", "9-10"):
        lines.append(f"{band}分: {scores.get(band, '')}")
    return "\n".join(lines)


def convert_checklist_to_criteria(checklist: list[dict]) -> list[dict]:
    """Transform raw checklist items into Rubric-compatible criteria."""
    criteria = []
    for item in checklist:
        name = item["name"]
        description = item["criteria_description"]
        scores = {k: item[k] for k in ("1-2", "3-4", "5-6", "7-8", "9-10") if k in item}

        criterion = {
            "criterion_id": _slugify(name),
            "text": _build_rubric_text(name, description, scores),
            "weight": 1.0,
            "criterion_type": "score",
            "check_points": [
                description,
                *[f"{band}分: {scores[band]}" for band in ("1-2", "3-4", "5-6", "7-8", "9-10") if band in scores],
            ],
        }
        criteria.append(criterion)
    return criteria


def convert_to_prompts(input_path: Path) -> dict:
    """Convert raw benchmark lines to the standard {'prompts': [...]} format."""
    prompts = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            index = raw["index"]
            prompt_id = f"writing_{index:04d}"

            prompt_entry = {
                "prompt_id": prompt_id,
                "prompt": raw["query"],
                "candidates": [],  # To be populated later by LLM generation or manual curation
                "metadata": {
                    "domain1": raw.get("domain1", ""),
                    "domain2": raw.get("domain2", ""),
                    "lang": raw.get("lang", ""),
                    "checklist": raw.get("checklist", []),
                },
            }
            prompts.append(prompt_entry)

    return {"prompts": prompts}


def convert_to_rubrics(input_path: Path) -> dict:
    """Convert raw benchmark lines to initial rubrics keyed by prompt_id."""
    rubrics = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            index = raw["index"]
            prompt_id = f"writing_{index:04d}"
            checklist = raw.get("checklist", [])

            criteria = convert_checklist_to_criteria(checklist)

            rubric = {
                "rubric_id": f"{prompt_id}_default",
                "criteria": criteria,
                "grading_protocol": {
                    "output_format": "json",
                    "allow_na": False,
                    "num_votes": 1,
                    "vote_method": "majority",
                    "strict_json": True,
                },
                "metadata": {
                    "description": f"Initial rubric for {raw.get('domain2', 'writing task')}",
                    "version": "1.0.0",
                    "task_type": "writing",
                    "domain1": raw.get("domain1", ""),
                    "domain2": raw.get("domain2", ""),
                    "lang": raw.get("lang", ""),
                },
            }
            rubrics[prompt_id] = rubric

    return rubrics


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "data" / "writing" / "benchmark_all.jsonl"
    output_dir = repo_root / "data" / "writing" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1. Generate prompts dataset
    prompts_data = convert_to_prompts(input_path)
    prompts_path = output_dir / "writing_prompts.json"
    prompts_path.write_text(
        json.dumps(prompts_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Prompts dataset written to {prompts_path}")
    print(f"     Total prompts: {len(prompts_data['prompts'])}")

    # 2. Generate initial rubrics
    rubrics_data = convert_to_rubrics(input_path)
    rubrics_path = output_dir / "writing_rubrics.json"
    rubrics_path.write_text(
        json.dumps(rubrics_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Initial rubrics written to {rubrics_path}")
    print(f"     Total rubrics: {len(rubrics_data)}")

    # Quick sanity check
    sample_prompt = prompts_data["prompts"][0]
    sample_rubric = rubrics_data[sample_prompt["prompt_id"]]
    print(f"\n[Sample] prompt_id: {sample_prompt['prompt_id']}")
    print(f"         prompt length: {len(sample_prompt['prompt'])}")
    print(f"         candidates: {len(sample_prompt['candidates'])}")
    print(f"         rubric criteria: {len(sample_rubric['criteria'])}")
    print(
        "\n[NOTE] The generated prompts file has empty candidates lists. "
        "You must add at least 2 candidates per prompt before using it with "
        "reward_harness.io_utils.load_dataset()."
    )


if __name__ == "__main__":
    main()

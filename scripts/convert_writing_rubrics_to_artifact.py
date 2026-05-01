#!/usr/bin/env python3
"""Convert best_rubrics_formal_writing_260501.json to deployable RM artifact.

.. deprecated::
    本脚本为历史兼容的 workaround，已废弃，仅保留用于处理旧 CLI 生成的
    非标准格式 rubric 文件（dict-style best_rubrics + 缺失 run_manifest）。

    标准流程应直接使用::

        uv run python -m reward_harness.cli ...
        uv run python -m reward_harness.rm.export --search-output <output> --out-artifact <artifact>

    标准 ``reward_harness.cli`` 通过 ``save_rubrics()`` 输出的 JSON 已经是
    ``rm.export`` 可直接消费的格式（list-style best_rubrics + 真实 run_manifest），
    无需再经过此脚本转换。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from reward_harness.rm.use_cases import export_rm_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert writing rubrics JSON to RM artifact"
    )
    parser.add_argument(
        "--input",
        default="artifacts/best_rubrics_formal_writing_260501.json",
        help="Path to raw best_rubrics JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rm_artifact_writing_260501.json",
        help="Path to write RM artifact JSON",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="LLM API base URL for runtime snapshot",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="Default LLM model for runtime snapshot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for runtime snapshot",
    )
    return parser


def main() -> None:
    warnings.warn(
        "convert_writing_rubrics_to_artifact.py is deprecated. "
        "Use reward_harness.cli -> rm.export directly for standard workflows.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = build_parser()
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Convert dict-style best_rubrics to list-style expected by export_rm_artifact
    best_rubrics_list = []
    for prompt_id, rubric in data["best_rubrics"].items():
        best_rubrics_list.append({"prompt_id": prompt_id, "rubric": rubric})

    run_manifest = {
        "seed": args.seed,
        "llm_snapshot": {
            "base_url": args.base_url,
            "default_model": args.model,
            "verifier_model": args.model,
            "timeout": 300,
            "max_retries": 0,
            "retry_backoff_base": 0.5,
            "retry_backoff_max": 8.0,
            "retry_jitter": 0.2,
            "fail_soft": True,
            "prompt_language": None,
        },
        "config_snapshot": {
            "extraction": {
                "strategy": "identity",
                "tag_name": "content",
                "pattern": None,
                "join_separator": "\n\n",
            },
            "candidate_extraction": {
                "strategy": "answer",
                "join_separator": "\n\n",
            },
        },
        "dataset": {"dataset_sha256": "unknown_writing_dataset"},
        "harness": {"session_id": "writing_260501_session"},
    }

    intermediate = {
        "best_rubrics": best_rubrics_list,
        "run_manifest": run_manifest,
    }

    intermediate_path = Path(args.output).with_suffix(".intermediate.json")
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(intermediate, f, ensure_ascii=False, indent=2)

    artifact_path = export_rm_artifact(
        search_output_path=intermediate_path,
        out_artifact_path=args.output,
    )
    print(f"Exported RM artifact: {artifact_path}")


if __name__ == "__main__":
    main()

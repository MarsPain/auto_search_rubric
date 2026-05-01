#!/usr/bin/env python3
"""Generate good and bad candidates for writing prompts using configured LLM."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path so we can import reward_harness
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward_harness.llm_client import LLMClient
from reward_harness.llm_config import LLMConfig

# ---------------------------------------------------------------------------
# Configuration – mirrors scripts/run_formal_search.sh defaults
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
API_KEY = os.getenv("LLM_API_KEY", "")
TIMEOUT = 120.0
MAX_RETRIES = 1
LLM_THINKING_TYPE = os.getenv("LLM_THINKING_TYPE", "disabled")

INPUT_PATH = PROJECT_ROOT / "data" / "writing" / "writing_prompts_tiny.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "writing" / "writing_prompts_tiny_with_candidates.json"

# Concurrency limit – keep conservative for free-tier endpoints
MAX_WORKERS = 3

GOOD_SYSTEM_TEMPLATE = """You are a professional academic writing assistant with deep expertise in the relevant field. 
Please provide a high-quality, well-structured, comprehensive, and accurate response to the user's request.
Your answer should demonstrate strong domain knowledge, clear organization, and professional language.
Respond directly with the requested content (outline, abstract, introduction, etc.) without extra commentary."""

BAD_SYSTEM_TEMPLATE = """You are a careless and unqualified writing assistant. 
Please produce a LOW-QUALITY response to the user's request. 
Your answer should be incomplete, poorly structured, vague, contain factual errors or inaccuracies, lack technical depth, and use inappropriate or sloppy language.
Do NOT mention that this is a bad example; just output the low-quality response directly.
Respond directly with the requested content (outline, abstract, introduction, etc.) without extra commentary."""


def _build_user_prompt(prompt_text: str, domain2: str) -> str:
    """Wrap the raw prompt so the model knows exactly what to generate."""
    task_type = domain2.lower().strip()
    return f"""Please help me with the following writing task.

Task type: {task_type}

---

{prompt_text}
"""


def call_llm(client: LLMClient, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call LLM and return raw text."""
    return client._request_text(
        client.request_json.__func__.__self__.__class__.__mro__[0](
            # We use the internal _request_text directly to avoid forcing JSON mode
        )
    )


def call_llm_text(client: LLMClient, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call LLM and return raw text via the internal helper."""
    from reward_harness.llm_client import ChatRequest
    req = ChatRequest(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    return client._request_text(req)


def generate_pair(client: LLMClient, model: str, item: dict, idx: int) -> dict:
    """Generate (good, bad) candidates for a single prompt item."""
    prompt_id = item["prompt_id"]
    prompt_text = item["prompt"]
    domain2 = item.get("metadata", {}).get("domain2", "Writing")

    user_prompt = _build_user_prompt(prompt_text, domain2)

    print(f"[{idx+1}] Generating GOOD candidate for {prompt_id} ...", flush=True)
    try:
        good_text = call_llm_text(client, model, GOOD_SYSTEM_TEMPLATE, user_prompt)
    except Exception as exc:
        print(f"[{idx+1}] GOOD candidate FAILED for {prompt_id}: {exc}", flush=True)
        good_text = f"[ERROR generating good candidate: {exc}]"

    print(f"[{idx+1}] Generating BAD  candidate for {prompt_id} ...", flush=True)
    try:
        bad_text = call_llm_text(client, model, BAD_SYSTEM_TEMPLATE, user_prompt)
    except Exception as exc:
        print(f"[{idx+1}] BAD candidate FAILED for {prompt_id}: {exc}", flush=True)
        bad_text = f"[ERROR generating bad candidate: {exc}]"

    candidates = [
        {
            "candidate_id": f"{prompt_id}_c1",
            "source": "chosen",
            "text": good_text.strip(),
            "metadata": {"rank": 1},
        },
        {
            "candidate_id": f"{prompt_id}_c2",
            "source": "rejected",
            "text": bad_text.strip(),
            "metadata": {"rank": 2},
        },
    ]

    return {
        "prompt_id": prompt_id,
        "candidates": candidates,
    }


def main() -> None:
    if not API_KEY:
        print("LLM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts", [])
    if not prompts:
        print("No prompts found in input file.", file=sys.stderr)
        sys.exit(1)

    # Resume support: load existing output and skip already-processed prompt_ids
    existing: dict[str, list] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        for item in old_data.get("prompts", []):
            existing[item["prompt_id"]] = item.get("candidates", [])
        print(f"Resuming from existing output ({len(existing)} already done).")

    extra_body = None
    if LLM_THINKING_TYPE:
        extra_body = {"thinking": {"type": LLM_THINKING_TYPE}}

    config = LLMConfig(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
        temperature=1.0,
        extra_body=extra_body,
    )
    client = LLMClient(config)

    results_map: dict[str, dict] = {}

    # Identify items to process
    to_process = [(i, p) for i, p in enumerate(prompts) if p["prompt_id"] not in existing]

    if not to_process:
        print("All prompts already have candidates. Nothing to do.")
        sys.exit(0)

    print(f"Processing {len(to_process)} prompts with model={LLM_MODEL}, base_url={BASE_URL}")

    # Process with limited concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(generate_pair, client, LLM_MODEL, p, i): (i, p)
            for i, p in to_process
        }

        for future in as_completed(future_to_idx):
            i, p = future_to_idx[future]
            try:
                result = future.result()
                results_map[result["prompt_id"]] = result["candidates"]
                print(f"[DONE] {result['prompt_id']} ({len(results_map)}/{len(to_process)})", flush=True)
            except Exception as exc:
                print(f"[FATAL] Failed for {p['prompt_id']}: {exc}", flush=True)
                # Leave candidates empty on fatal error so the user can retry
                results_map[p["prompt_id"]] = []

    # Merge with existing / original data
    out_prompts = []
    for item in prompts:
        pid = item["prompt_id"]
        new_item = dict(item)
        if pid in results_map:
            new_item["candidates"] = results_map[pid]
        elif pid in existing:
            new_item["candidates"] = existing[pid]
        out_prompts.append(new_item)

    output_data = {"prompts": out_prompts}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"Total prompts: {len(out_prompts)}")
    ok_count = sum(1 for p in out_prompts if p.get("candidates"))
    print(f"Prompts with candidates: {ok_count}")


if __name__ == "__main__":
    main()

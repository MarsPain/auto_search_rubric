# auto_search_rubric

English | [中文](README.zh.md)

An automated search framework for **rubric-based reward modeling**, inspired by the paper:

[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](https://arxiv.org/abs/2509.21500)

This repo keeps a baseline `Iterative RTD` for comparison, and defaults to a more extensible `Evolutionary RTD` implementation (`--mode evolutionary`).

> Original paper Iterative RTD open-source project: [Jun-Kai-Zhang/rubrics](https://github.com/Jun-Kai-Zhang/rubrics)

## Highlights

- Structured rubric schema: `criteria`, `weights`, `grading_protocol`, positive/negative examples, checkpoints
- Multi-vote verification aggregation: majority vote + vote-level variance tracking
- Tail-focused objective:
  - `TailAcc`
  - `TailVar`
  - `DiverseTailAcc`
- Dataset-supported preference ranks (`metadata.rank`) with automatic `RankPreferenceJudge`

## How This Repo Differs: Evolutionary RTD vs. Iterative RTD (Paper)

> Note: this repo implements `iterative` as a baseline and provides `evolutionary` as the default/stronger search mode.

### 1) A more extensible architecture

- Protocol-based interfaces: `RubricInitializer`, `RubricProposer`, `Verifier`, `PreferenceJudge`
- The same search flow can swap `mock` / `llm` components
- Role-based model configuration (initializer/proposer/verifier/judge can use different models)

### 2) Evolutionary search (not a single iterative trajectory)

- **Mutation strategies + elitism**
  - Maintain a rubric candidate **population**
  - Generate multiple mutated rubrics per generation (`mutations_per_round`)
  - Keep elite rubrics (`elitism_count`) and add winners to reduce single-path local optima risks

- **Budget-aware filtering (successive halving)**
  - Coarse evaluation for many candidates (small pair budget)
  - Fine evaluation for a few survivors (medium to full budget)
  - Implemented via `pair_budget_small / medium / full` for more "budget-efficient" search

- **Hard-prompt focus**
  - Each generation prioritizes prompts that are harder to separate (top margin + population disagreement)
  - Concentrates limited budget on the "hard cases"

### 3) Dataset-defined candidate preference ranks

- If **all candidates** in the dataset provide `metadata.rank` (lower is better):
  - the system automatically uses `RankPreferenceJudge`
  - even with `llm` backend, it will prefer rank-based ground truth (no LLM judge calls needed)
- If any candidate misses `rank`, it falls back to the default judge (heuristic under `mock`, LLM judge under `llm`)

## Project Layout

- `autosr/`: core package (CLI, search, evaluation, LLM/mock components)
- `tests/`: `unittest` tests
- `scripts/`: helper scripts for tests and "formal" runs
- `examples/`: demo datasets (with/without rank)
- `artifacts/`: default output directory

## Install

Requires: Python `>=3.11`

```bash
python3 -m pip install -e .
```

After installation, you can run via:

- `python3 -m autosr.cli ...`
- `autosr ...`

## Quick Start

Recommended default: evolutionary search

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json
```

Iterative baseline:

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode iterative \
  --output artifacts/best_rubrics_iterative.json
```

## LLM Backend

CLI supports `--backend {auto,mock,llm}`:

- `auto` (default): use `llm` if `LLM_API_KEY` is present; otherwise `mock`
- `llm`: require API key and fail fast if missing
- `mock`: always use local components

The default configuration uses OpenRouter-compatible endpoints. You can override `--base-url` to use any OpenAI-compatible API provider.
Default model is `stepfun/step-3.5-flash:free`.

Run the "formal" flow (requires API key):

```bash
export LLM_API_KEY="<YOUR_API_KEY>"
./scripts/run_formal_search.sh examples/single_case.json evolutionary artifacts/best_rubrics_formal.json
```

Direct CLI example with role-specific models:

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode evolutionary \
  --output artifacts/best_rubrics_llm.json \
  --backend llm \
  --base-url https://openrouter.ai/api/v1 \
  --model-default stepfun/step-3.5-flash:free \
  --model-initializer stepfun/step-3.5-flash:free \
  --model-proposer stepfun/step-3.5-flash:free \
  --model-verifier stepfun/step-3.5-flash:free \
  --model-judge stepfun/step-3.5-flash:free \
  --llm-timeout 30 \
  --llm-max-retries 2
```

## Search and Objective (Implementation Notes)

- Objective:
  - `total = TailAcc - lambda_var * TailVar + mu_diverse * DiverseTailAcc`
- Key defaults (CLI):
  - `--generations 12`
  - `--population-size 8`
  - `--mutations-per-round 6`
  - `--batch-size 3`
  - `--tail-fraction 0.25`
  - `--lambda-var 0.2`
  - `--mu-diverse 0.25`
- Code-level (in `EvolutionaryConfig`) settings not currently exposed as CLI flags:
  - `survival_fraction` (stage survival ratio)
  - `elitism_count` (number of elites to keep)
  - `stagnation_generations` (early-stop threshold)

## Dataset Format

Input must be JSON with top-level `prompts`:

```json
{
  "prompts": [
    {
      "prompt_id": "p1",
      "prompt": "Write ...",
      "candidates": [
        {
          "candidate_id": "c1",
          "text": "response text",
          "source": "strong",
          "metadata": { "quality": 0.91, "rank": 1 }
        }
      ]
    }
  ]
}
```

Fields:

- `prompt_id`, `prompt`: required
- each prompt needs at least 2 `candidates`
- `metadata.quality`: optional, used by the heuristic judge
- `metadata.rank`: optional preference label (`1` is the best)

Examples:

- `examples/single_case.json` (quality-based)
- `examples/single_case_with_rank.json` (rank-based)

## Output Format

The result is written to `--output`:

- `best_rubrics`: best rubric per prompt
- `best_objective_scores`: objective score per prompt
- `best_scores`: legacy alias of `best_objective_scores` (kept for compatibility)
- `best_candidates`: top candidate id under the best rubric
- `candidate_scores`: all candidate scores under the best rubric
- `best_candidate_scores`: max value from `candidate_scores` per prompt

## Tests

Run unit tests:

```bash
./scripts/run_tests_unit.sh
```

Run integration tests:

```bash
export LLM_API_KEY="<YOUR_API_KEY>"
./scripts/run_tests_integration.sh
```

Run the aggregate entrypoint:

```bash
./scripts/run_tests.sh
```

Or run full test discovery directly:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Notes:

- `run_tests_unit.sh` forces integration tests to skip
- `run_tests_integration.sh` requires `LLM_API_KEY`
- `run_tests.sh` always runs unit tests first, then runs integration tests only when `LLM_API_KEY` is set
- Test scripts pick Python in this order: `VIRTUAL_ENV/bin/python` -> `PYTHON_BIN` -> `python3`
- Integration test endpoint/model can be overridden with `LLM_BASE_URL` and `LLM_MODEL`

## Notes

- This repo emphasizes reproducibility, component replaceability, and budget-aware search.
- For strict comparisons, run both `iterative` and `evolutionary` and compare `best_scores` and generated rubrics.

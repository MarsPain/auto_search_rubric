# AutoSearchRubric

This repository provides an implementation framework for rubric search inspired by:

`Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training`

Implemented pieces:

- Structured rubric schema (`criteria`, `weights`, `grading_protocol`, examples/checkpoints)
- Multi-vote verifier aggregation (majority vote + variance tracking)
- Baseline iterative RTD workflow
- Evolutionary RTD search with:
  - active tail sampling
  - mutation-based proposer calls
  - successive halving for budgeted evaluation
  - tail-focused objective:
    - `TailAcc`
    - `TailVar`
    - `DiverseTailAcc`
- Conservative/aggressive blended reward schedule helper

## Quick Start

```bash
python3 -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json
```

Run iterative baseline:

```bash
python3 -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode iterative \
  --output artifacts/best_rubrics_iterative.json
```

Run tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## LLM Backend (OpenRouter via openai-python)

The CLI now supports `mock` and real `llm` backends.

- `--backend auto` (default): use LLM when `OPENROUTER_API_KEY` exists, otherwise mock.
- `--backend llm`: require API key and fail fast if missing.
- `--backend mock`: always use local heuristic components.

Install dependencies:

```bash
python3 -m pip install -e .
```

Run formal flow with real LLM:

```bash
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_API_KEY>"
./scripts/run_formal_search.sh examples/demo_dataset.json evolutionary artifacts/best_rubrics_formal.json
```

Direct CLI example with role-specific models:

```bash
python3 -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode iterative \
  --output artifacts/best_rubrics_llm.json \
  --backend auto \
  --base-url https://openrouter.ai/api/v1 \
  --model-default openai/gpt-4o-mini \
  --model-initializer openai/gpt-4o-mini \
  --model-proposer anthropic/claude-3.5-sonnet \
  --model-verifier openai/gpt-4o-mini \
  --model-judge openai/gpt-4o-mini \
  --llm-timeout 30 \
  --llm-max-retries 2
```

Run all tests with automatic integration skip/run behavior:

```bash
./scripts/run_tests.sh
```

## Dataset Format

Input dataset must be JSON:

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
          "metadata": { "quality": 0.91 }
        }
      ]
    }
  ]
}
```

`metadata.quality` is optional but useful for the demo `PreferenceJudge`.

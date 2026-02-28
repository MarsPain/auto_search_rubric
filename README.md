# auto_search_rubric

English | [中文](README.zh.md)

Automated search framework for rubric-based reward modeling, inspired by:
[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](https://arxiv.org/abs/2509.21500)

This repository keeps `iterative` as a baseline and uses `evolutionary` as the default search mode.

## Highlights

- Unified runtime configuration with typed enums (`autosr.types`) and layered config dataclasses (`autosr.config`)
- Composition-root factory (`ComponentFactory`) for backend-aware dependency wiring
- Canonical domain models in `autosr.data_models` with compatibility re-export in `autosr.models`
- Search extensibility:
  - Parent selection: `rank`, `tournament`, `top_k`
  - Adaptive mutation: `fixed`, `success_feedback`, `exploration_decay`, `diversity_driven`
- LLM architecture split into transport config (`autosr.llm_config`) and runtime config (`autosr.config`)
- Reproducibility outputs:
  - `run_manifest` embedded in output JSON
  - archived manifest and replay script under `<output_parent>/run_records/`

## Architecture (Current)

### Entry and Composition

- `autosr/cli.py`
  - Parses CLI args only
  - Builds `RuntimeConfig`
  - Delegates runtime wiring to `ComponentFactory`
- `autosr/factory.py`
  - Single composition root for backend selection and component assembly
  - Auto-resolves rank-based judge when all candidates provide `metadata.rank`

### Config and Types

- `autosr/config.py`
  - Runtime-level configuration:
    - `RuntimeConfig`
    - `LLMBackendConfig`
    - `SearchAlgorithmConfig`
    - `ObjectiveConfig` (compat alias: `ObjectiveFunctionConfig`)
    - `InitializerStrategyConfig`, `ContentExtractionConfig`, `VerifierConfig`
- `autosr/llm_config.py`
  - Low-level LLM transport/model config (`LLMConfig`, `RoleModelConfig`)
- `autosr/types.py`
  - Shared enums:
    - `BackendType`, `SearchMode`, `SelectionStrategy`
    - `AdaptiveMutationSchedule`, `InitializerStrategy`, `ExtractionStrategy`, `LLMRole`

### Domain and Shared Modules

- `autosr/data_models.py`: canonical domain entities (`Rubric`, `Criterion`, `PromptExample`, ...)
- `autosr/models.py`: compatibility import layer
- `autosr/exceptions.py`: shared LLM exceptions (`LLMCallError`, `LLMParseError`)
- `autosr/io_utils.py`: dataset/rubric I/O and run-record persistence
- `autosr/run_records/use_cases.py`: run manifest + reproducible shell script generation

### Search Domain

- `autosr/search/config.py`: `IterativeConfig`, `EvolutionaryConfig`, `SearchResult`
- `autosr/search/iterative.py`: iterative baseline implementation
- `autosr/search/evolutionary.py`: evolutionary algorithm implementation
- `autosr/search/strategies.py`: reusable search helpers
- `autosr/search/selection_strategies.py`: parent selection policies
- `autosr/search/adaptive_mutation.py`: mutation scheduler and diversity metrics
- `autosr/search/use_cases.py`: searcher entrypoints exports

### LLM + Extraction Domain

- `autosr/llm_components/base.py`: request/retry base + prompt rendering fallback
- `autosr/llm_components/parsers.py`: response normalization/validation
- `autosr/llm_components/use_cases.py`: initializer/proposer/verifier/judge implementations
- `autosr/llm_components/factory.py`: legacy helper kept for compatibility
- `autosr/content_extraction/strategies.py`: `tag` / `regex` / `identity` extraction
- `autosr/content_extraction/use_cases.py`: extraction-decorated verifier
- `autosr/prompts/loader.py` + `autosr/prompts/constants.py`: file templates and constant fallback

## Project Layout

- `autosr/`: core package
- `prompts/`: prompt templates (supports locale folders such as `prompts/zh/` and `prompts/en/`)
- `tests/`: `unittest` test suite
- `scripts/`: unit/integration/formal run scripts
- `examples/`: demo datasets and examples
- `artifacts/`: default output directory

## Environment Setup

Requires Python `>=3.11` and `uv`.

```bash
uv sync
```

Run commands with `uv run`:

```bash
uv run python -m autosr.cli --help
```

## Quick Start

Default (evolutionary):

```bash
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json
```

Iterative baseline:

```bash
uv run python -m autosr.cli \
  --dataset examples/single_case.json \
  --mode iterative \
  --output artifacts/best_rubrics_iterative.json
```

Evolutionary with custom strategy and prompt locale:

```bash
uv run python -m autosr.cli \
  --dataset examples/single_case_with_rank.json \
  --mode evolutionary \
  --output artifacts/best_rubrics_rank.json \
  --selection-strategy top_k \
  --adaptive-mutation diversity_driven \
  --prompt-language zh
```

## Backend and LLM Configuration

`--backend {auto,mock,llm}`:

- `auto` (default): resolve to `llm` when API key exists, else `mock`
- `llm`: requires API key (`LLM_API_KEY` by default, configurable via `--api-key-env`)
- `mock`: local heuristic components only

Default endpoint/model:

- `--base-url https://openrouter.ai/api/v1`
- `--model-default stepfun/step-3.5-flash:free`

Role-specific model override is supported:

- `--model-initializer`
- `--model-proposer`
- `--model-verifier`
- `--model-judge`

Prompt locale loading order:

1. `prompts/<language>/` (when `--prompt-language` is set)
2. `prompts/`
3. built-in constants in code

Formal LLM-backed flow:

```bash
export LLM_API_KEY="..."
./scripts/run_formal_search.sh \
  examples/call_summary_dataset_with_rank_single.json \
  evolutionary \
  artifacts/best_rubrics_formal_call_summary.json
```

## Search Objective and Controls

Objective:

`score = TailAcc - lambda_var * TailVar + mu_diverse * DiverseTailAcc`

Common flags:

- `--generations`, `--population-size`, `--mutations-per-round`, `--batch-size`
- `--tail-fraction`, `--lambda-var`, `--mu-diverse`
- `--pair-confidence-prior` (pairwise confidence shrinkage; set `0` to disable)
- `--selection-strategy {rank,tournament,top_k}`
- `--adaptive-mutation {fixed,success_feedback,exploration_decay,diversity_driven}`

## Dataset Format

Input JSON must contain top-level `prompts`:

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

Notes:

- `prompt_id` and `prompt` are required
- each prompt must provide at least 2 candidates
- `metadata.rank` is optional (`1` is best); if present for all candidates, rank-based judge is auto-selected

## Output and Reproducibility

Main output JSON (`--output`) includes:

- `best_rubrics` (array; each item may include `best_candidate_id` and `candidate_scores`)
- `best_objective_scores`
- `best_scores` (legacy alias of `best_objective_scores`)
- optional `run_manifest`

Per-run reproducibility files are written to:

- `<output_parent>/run_records/<output_stem>_<run_id>.manifest.json`
- `<output_parent>/run_records/<output_stem>_<run_id>.reproduce.sh`

## Tests

Unit tests:

```bash
./scripts/run_tests_unit.sh
```

Integration tests (requires API key):

```bash
export LLM_API_KEY="..."
./scripts/run_tests_integration.sh
```

Aggregate entrypoint:

```bash
./scripts/run_tests.sh
```

Run all tests directly:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Architecture-focused regression set:

```bash
uv run python -m unittest \
  tests.test_architecture_refactor \
  tests.test_cli_backend_selection \
  tests.test_cli_best_candidates \
  tests.test_io_utils \
  tests.test_search_config_enum_unification \
  tests.test_data_models_compat \
  tests.test_exceptions_module \
  tests.test_evolutionary_decoupling
```

## Notes

- Import domain entities from `autosr.data_models` in new code.
- Prefer `ComponentFactory(RuntimeConfig(...))` over manual runtime wiring.
- Keep secrets in environment variables only (`LLM_API_KEY`, optional `LLM_BASE_URL`, `LLM_MODEL`).

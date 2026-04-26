# auto_search_rubric

English | [中文](README.zh.md)

Automated search framework for rubric-based reward modeling, inspired by:
[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](https://arxiv.org/abs/2509.21500)

This repository keeps `iterative` as a baseline and uses `evolutionary` as the default search mode.
It now covers the path from rubric search to deployable RM artifacts and an RM server MVP for online scoring. RL experiment lineage tracking and external RL training run referencing are under active design (Stage D/E).

## Highlights

- Unified runtime configuration with typed enums (`autosr.types`) and layered config dataclasses (`autosr.config`)
- Composition-root factory (`ComponentFactory`) for backend-aware dependency wiring
- Canonical domain models in `autosr.data_models` with compatibility re-export in `autosr.models`
- Search extensibility:
  - Parent selection: `rank`, `tournament`, `top_k`
  - Adaptive mutation: `fixed`, `success_feedback`, `exploration_decay`, `diversity_driven`
  - Iteration scope: `global_batch` (dataset-level) and `prompt_local` (prompt-level independent evolution)
- LLM architecture split into transport config (`autosr.llm_config`) and runtime config (`autosr.config`)
- Deployable RM artifacts with validated schema and embedded runtime snapshot for server startup
- Deployment tracking via `autosr.rm.deploy` manifests with per-target `previous_artifact_id` resolution
- RM Server MVP (`autosr.rm.server`) exposing `/healthz`, `/score`, and `/batch_score` with closed-loop LLM scoring
- Reproducibility outputs:
  - `run_manifest` embedded in output JSON
  - archived manifest and replay script under `<output_parent>/run_records/`
  - RM deployment records under `artifacts/rm_deployments/`
  - optional RM server request logs under `artifacts/rm_server_logs/`

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
    - `BackendType`, `SearchMode`, `EvolutionIterationScope`, `SelectionStrategy`
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

### RL Domain (Stage D/E — active design)

- `autosr/rl/`: experiment registry, lineage tracking, and external RL training-run reference scaffolding
  - `data_models.py`, `registry.py`, `lineage.py`, `validation.py`, `io.py`
  - `cli/`: `record_manifest`, `record_eval`, `record_result`, `show_lineage`
  - `verl/`: `prepare_training_run`, `run_verl_training`, `finalize_training_run`, `reward_client`

### RM Artifact + Serving Domain

- `autosr/rm/data_models.py`: deployable RM artifact schema and deploy manifest schema
- `autosr/rm/use_cases.py`: artifact export and deployment-record use cases
- `autosr/rm/export.py`: CLI for exporting search output into a deployable RM artifact
- `autosr/rm/deploy.py`: CLI for recording per-environment deployment manifests
- `autosr/rm/server.py`: FastAPI RM server that loads artifact runtime snapshot and serves scoring APIs

## Project Layout

- `autosr/`: core package
- `autosr/rm/`: RM artifact/export/deploy/server modules
- `autosr/rl/`: RL experiment lineage and external training-run reference modules
- `prompts/`: prompt templates (supports locale folders such as `prompts/zh/` and `prompts/en/`)
- `tests/`: `unittest` test suite
- `scripts/`: unit/integration/formal run scripts
- `examples/`: demo datasets and examples
- `artifacts/`: default output directory for search outputs, RM artifacts, deployment manifests, and server logs

## Environment Setup

Requires Python `>=3.11` and `uv`.

```bash
uv sync
```

Run commands with `uv run`:

```bash
uv run python -m autosr.cli --help
```

`uv sync` installs both the search stack and RM server dependencies (`fastapi`, `uvicorn`).

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

End-to-end RM flow:

```bash
# 1) Search for the best rubric
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json

# 2) Export a deployable RM artifact
uv run python -m autosr.rm.export \
  --search-output artifacts/best_rubrics.json \
  --out-artifact artifacts/rm_artifacts/rm_v1.json

# 3) Record deployment metadata
uv run python -m autosr.rm.deploy \
  --artifact artifacts/rm_artifacts/rm_v1.json \
  --deployment-target dev

# 4) Start the RM server
export LLM_API_KEY="..."
uv run python -m autosr.rm.server \
  --artifact artifacts/rm_artifacts/rm_v1.json \
  --host 0.0.0.0 \
  --port 8080 \
  --request-log-path artifacts/rm_server_logs/requests.jsonl
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

Note:
- `scripts/run_formal_search.sh` now defaults to `--evolution-iteration-scope prompt_local`
- Override with environment variable if needed:
  - `EVOLUTION_ITERATION_SCOPE=global_batch ./scripts/run_formal_search.sh`

## Search Objective and Controls

Objective:

`score = TailAcc - lambda_var * TailVar + mu_diverse * DiverseTailAcc`

Common flags:

- `--generations`, `--population-size`, `--mutations-per-round`, `--batch-size`
- `--mutation-parent-count` (number of parent rubrics used for mutation each generation)
- `--tail-fraction`, `--lambda-var`, `--mu-diverse`
- `--pair-confidence-prior` (pairwise confidence shrinkage; set `0` to disable)
- `--selection-strategy {rank,tournament,top_k}`
- `--adaptive-mutation {fixed,success_feedback,exploration_decay,diversity_driven}`
- `--evolution-iteration-scope {global_batch,prompt_local}`
- `--stop-when-distinguished` / `--no-stop-when-distinguished` (prompt-local early stop)
- `--distinguish-margin` (override top-margin threshold; default uses objective tie tolerance)

Verifier grading scale:

- Supports continuous criterion scores in `0-5` (preferred) and `0-1` (compatible).
- Final rubric score is normalized to `0-1` before objective computation.

Iteration behavior:

- `global_batch`:
  - Original dataset-level generations
  - each generation evolves only the selected hard prompts (`batch_size`)
- `prompt_local`:
  - each prompt evolves independently for up to `generations`
  - no cross-prompt batching dependency
  - can stop early per prompt when top candidates are already distinguished

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

RM artifact and deployment outputs:

- `artifacts/rm_artifacts/*.json`: deployable RM artifacts exported from search results
- `artifacts/rm_deployments/*.json`: deployment records with `deployment_target`, `deployed_by`, and `previous_artifact_id`
- `artifacts/rm_server_logs/requests.jsonl`: request logs emitted by `autosr.rm.server`

RM server notes:

- The server requires an artifact with the embedded runtime snapshot produced by `autosr.rm.export`.
- Stable endpoints: `GET /healthz`, `POST /score`, `POST /batch_score`.

## Tests

Recommended local quality gate before handing off changes:

```bash
./scripts/run_tests_unit.sh
uv run python scripts/validate_docs.py
./scripts/run_quality_checks.sh
```

`run_quality_checks.sh` enforces ruff lint and the current staged mypy scope.
Ruff format remains a visible report rather than a hard gate until the repository
is batch-formatted.

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

RL lineage regression set:

```bash
uv run python -m unittest \
  tests.test_rl_lineage \
  tests.test_rl_verl_reference_flow
```

RM artifact/server regression set:

```bash
uv run python -m unittest \
  tests.test_rm_artifact \
  tests.test_rm_deploy_manifest \
  tests.test_rm_server
```

## Notes

- Import domain entities from `autosr.data_models` in new code.
- `autosr.models` is a long-term compatibility re-export shim for historical
  import paths; keep it working, but do not use it in new code.
- Prefer `ComponentFactory(RuntimeConfig(...))` over manual runtime wiring.
- Keep secrets in environment variables only (`LLM_API_KEY`, optional `LLM_BASE_URL`, `LLM_MODEL`).

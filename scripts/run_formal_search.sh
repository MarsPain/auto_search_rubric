#!/usr/bin/env bash
set -euo pipefail

# DATASET_PATH="${1:-examples/single_case_with_rank.json}"
# DATASET_PATH="${1:-examples/call_summary_dataset_with_rank.json}"
DATASET_PATH="${1:-examples/call_summary_dataset_with_rank_single.json}"
MODE="${2:-evolutionary}"
OUTPUT_PATH="${3:-artifacts/best_rubrics_formal_call_summary.json}"

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY is not set. Export it before running formal search." >&2
  exit 1
fi

BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}"
# LLM_MODEL="${LLM_MODEL:-deepseek/deepseek-v3.2}"
LLM_MODEL="${LLM_MODEL:-stepfun/step-3.5-flash:free}"
MODEL_INITIALIZER="${MODEL_INITIALIZER:-}"
MODEL_PROPOSER="${MODEL_PROPOSER:-}"
MODEL_VERIFIER="${MODEL_VERIFIER:-}"
MODEL_JUDGE="${MODEL_JUDGE:-}"
LLM_TIMEOUT="${LLM_TIMEOUT:-30}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-0}"

# Initial rubric configuration
INITIALIZER_STRATEGY="${INITIALIZER_STRATEGY:-preset}"
PRESET_RUBRICS="${PRESET_RUBRICS:-examples/call_summary_initial_rubric.json}"
PRESET_STRICT="${PRESET_STRICT:-}"

# Content extraction configuration for call summary dataset
# The call_summary dataset wraps conversation content in <通话内容> tags
EXTRACT_STRATEGY="${EXTRACT_STRATEGY:-tag}"
EXTRACT_TAG="${EXTRACT_TAG:-通话内容}"

cmd=(
  python3 -u -m autosr.cli
  --dataset "${DATASET_PATH}"
  --mode "${MODE}"
  --output "${OUTPUT_PATH}"
  --backend auto
  --base-url "${BASE_URL}"
  --model-default "${LLM_MODEL}"
  --llm-timeout "${LLM_TIMEOUT}"
  --llm-max-retries "${LLM_MAX_RETRIES}"
  --initializer-strategy "${INITIALIZER_STRATEGY}"
  --extract-strategy "${EXTRACT_STRATEGY}"
  --generations 4
  --population-size 4
  --mutations-per-round 2
  --batch-size 2
)

if [[ -n "${MODEL_INITIALIZER}" ]]; then
  cmd+=(--model-initializer "${MODEL_INITIALIZER}")
fi
if [[ -n "${MODEL_PROPOSER}" ]]; then
  cmd+=(--model-proposer "${MODEL_PROPOSER}")
fi
if [[ -n "${MODEL_VERIFIER}" ]]; then
  cmd+=(--model-verifier "${MODEL_VERIFIER}")
fi
if [[ -n "${MODEL_JUDGE}" ]]; then
  cmd+=(--model-judge "${MODEL_JUDGE}")
fi
if [[ -n "${PRESET_RUBRICS}" ]]; then
  cmd+=(--preset-rubrics "${PRESET_RUBRICS}")
fi
if [[ "${PRESET_STRICT:-}" == "true" ]]; then
  cmd+=(--preset-strict)
fi
if [[ -n "${EXTRACT_TAG:-}" ]]; then
  cmd+=(--extract-tag "${EXTRACT_TAG}")
fi

echo "Running command: ${cmd[*]}"
"${cmd[@]}"

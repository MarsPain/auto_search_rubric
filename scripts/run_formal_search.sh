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

# =============================================================================
# SELECTION STRATEGY & ADAPTIVE MUTATION CONFIGURATION
# =============================================================================
# Uncomment one of the following strategy blocks to test different configurations.
# Only one SELECTION_STRATEGY should be active at a time.

# -----------------------------------------------------------------------------
# OPTION 1: Original (Rank-based) - Default behavior
# -----------------------------------------------------------------------------
SELECTION_STRATEGY="${SELECTION_STRATEGY:-rank}"
# ADAPTIVE_MUTATION="${ADAPTIVE_MUTATION:-fixed}"

# -----------------------------------------------------------------------------
# OPTION 2: Tournament Selection + Success Feedback
# Description: Balanced selection pressure with adaptive mutation learning
# Best for: Stable convergence with automatic mutation optimization
# -----------------------------------------------------------------------------
# SELECTION_STRATEGY="${SELECTION_STRATEGY:-tournament}"
# TOURNAMENT_SIZE="${TOURNAMENT_SIZE:-3}"
# TOURNAMENT_P="${TOURNAMENT_P:-0.8}"
# ADAPTIVE_MUTATION="${ADAPTIVE_MUTATION:-success_feedback}"
# MUTATION_WINDOW_SIZE="${MUTATION_WINDOW_SIZE:-10}"

# -----------------------------------------------------------------------------
# OPTION 3: Top-K with Diversity Protection + Diversity Driven
# Description: Explicit diversity control in both selection and mutation
# Best for: Maintaining population diversity, avoiding local optima
# -----------------------------------------------------------------------------
SELECTION_STRATEGY="${SELECTION_STRATEGY:-top_k}"
TOP_K_RATIO="${TOP_K_RATIO:-0.3}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.4}"
ADAPTIVE_MUTATION="${ADAPTIVE_MUTATION:-diversity_driven}"
DIVERSITY_THRESHOLD="${DIVERSITY_THRESHOLD:-0.05}"

# -----------------------------------------------------------------------------
# OPTION 4: Tournament + Exploration Decay (Two-phase search)
# Description: Early exploration with later exploitation
# Best for: Complex search spaces requiring broad initial exploration
# -----------------------------------------------------------------------------
# SELECTION_STRATEGY="${SELECTION_STRATEGY:-tournament}"
# TOURNAMENT_SIZE="${TOURNAMENT_SIZE:-2}"
# TOURNAMENT_P="${TOURNAMENT_P:-0.7}"
# ADAPTIVE_MUTATION="${ADAPTIVE_MUTATION:-exploration_decay}"
# EXPLORATION_PHASE_RATIO="${EXPLORATION_PHASE_RATIO:-0.3}"

# =============================================================================

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

# Add selection strategy parameters if specified
if [[ -n "${SELECTION_STRATEGY:-}" ]]; then
  cmd+=(--selection-strategy "${SELECTION_STRATEGY}")
fi
if [[ -n "${TOURNAMENT_SIZE:-}" ]]; then
  cmd+=(--tournament-size "${TOURNAMENT_SIZE}")
fi
if [[ -n "${TOURNAMENT_P:-}" ]]; then
  cmd+=(--tournament-p "${TOURNAMENT_P}")
fi
if [[ -n "${TOP_K_RATIO:-}" ]]; then
  cmd+=(--top-k-ratio "${TOP_K_RATIO}")
fi
if [[ -n "${DIVERSITY_WEIGHT:-}" ]]; then
  cmd+=(--diversity-weight "${DIVERSITY_WEIGHT}")
fi

# Add adaptive mutation parameters if specified
if [[ -n "${ADAPTIVE_MUTATION:-}" ]]; then
  cmd+=(--adaptive-mutation "${ADAPTIVE_MUTATION}")
fi
if [[ -n "${MUTATION_WINDOW_SIZE:-}" ]]; then
  cmd+=(--mutation-window-size "${MUTATION_WINDOW_SIZE}")
fi
if [[ -n "${DIVERSITY_THRESHOLD:-}" ]]; then
  cmd+=(--diversity-threshold "${DIVERSITY_THRESHOLD}")
fi
if [[ -n "${EXPLORATION_PHASE_RATIO:-}" ]]; then
  cmd+=(--exploration-phase-ratio "${EXPLORATION_PHASE_RATIO}")
fi

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

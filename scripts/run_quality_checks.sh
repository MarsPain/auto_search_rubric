#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  RUFF_CMD=(uv run --with ruff ruff)
  MYPY_CMD=(uv run --with mypy mypy)
elif command -v ruff >/dev/null 2>&1; then
  RUFF_CMD=(ruff)
  if command -v mypy >/dev/null 2>&1; then
    MYPY_CMD=(mypy)
  else
    echo "mypy is not available. Install uv or mypy to run quality checks." >&2
    exit 1
  fi
else
  echo "ruff is not available. Install uv or ruff to run quality checks." >&2
  exit 1
fi

echo "Running ruff lint..."
"${RUFF_CMD[@]}" check autosr tests scripts

echo "Running mypy type checks..."
"${MYPY_CMD[@]}" autosr/mix_reward.py

echo "Checking ruff formatting..."
if ! "${RUFF_CMD[@]}" format --check autosr tests scripts; then
  echo "Ruff formatting differences detected; formatter enforcement is staged." >&2
fi

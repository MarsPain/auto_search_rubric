#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
  PYTHON_DESC="uv run python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_CMD=("${VIRTUAL_ENV}/bin/python")
  PYTHON_DESC="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_CMD=("${PYTHON_BIN:-python3}")
  PYTHON_DESC="${PYTHON_CMD[0]}"
fi

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY is not set. Export it before running integration tests." >&2
  exit 1
fi

echo "Running integration tests with ${PYTHON_DESC}..."
"${PYTHON_CMD[@]}" -m unittest tests.test_llm_integration

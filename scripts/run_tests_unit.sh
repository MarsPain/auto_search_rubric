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

echo "Running unit tests with ${PYTHON_DESC}..."
# Force integration tests to skip, even if caller exported LLM_API_KEY.
LLM_API_KEY="" "${PYTHON_CMD[@]}" -m unittest discover -s tests -p "test_*.py"

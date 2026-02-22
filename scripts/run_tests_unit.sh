#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "Running unit tests with ${PYTHON_BIN}..."
# Force integration tests to skip, even if caller exported LLM_API_KEY.
LLM_API_KEY="" "${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py"

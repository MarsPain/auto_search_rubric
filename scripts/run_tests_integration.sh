#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY is not set. Export it before running integration tests." >&2
  exit 1
fi

echo "Running integration tests with ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m unittest tests.test_llm_integration

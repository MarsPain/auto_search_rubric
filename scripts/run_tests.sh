#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_tests_unit.sh"

if [[ -n "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY detected: running integration tests."
  "${SCRIPT_DIR}/run_tests_integration.sh"
else
  echo "LLM_API_KEY not set: integration tests skipped."
fi

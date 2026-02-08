#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY detected: integration tests will run."
else
  echo "LLM_API_KEY not set: integration tests will be skipped."
fi

python3 -m unittest discover -s tests -p "test_*.py"

#!/usr/bin/env bash
# Deploy RM server from writing rubrics artifact.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARTIFACT="${1:-artifacts/rm_artifact_writing_260501.json}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8080}"

if [[ -z "${LLM_API_KEY:-}" ]]; then
    echo "ERROR: LLM_API_KEY is not set. Export it before deploying." >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    PYTHON_PREFIX=(uv run)
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_PREFIX=("${VIRTUAL_ENV}/bin/python")
else
    PYTHON_PREFIX=("${PYTHON_BIN:-python3}")
fi

# Generate artifact if missing
if [[ ! -f "$ARTIFACT" ]]; then
    echo "Artifact not found at $ARTIFACT. Generating from raw rubrics..."
    "${PYTHON_PREFIX[@]}" python scripts/convert_writing_rubrics_to_artifact.py --output "$ARTIFACT"
fi

# Record deployment manifest
echo "Recording deployment manifest..."
"${PYTHON_PREFIX[@]}" python -m reward_harness.rm.deploy \
    --artifact "$ARTIFACT" \
    --deployment-target dev \
    --out-dir artifacts/rm_deployments

echo ""
echo "Starting RM server on ${HOST}:${PORT} ..."
echo "Artifact : $ARTIFACT"
echo "Press Ctrl+C to stop."
echo ""

"${PYTHON_PREFIX[@]}" python -m reward_harness.rm.server \
    --artifact "$ARTIFACT" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key-env LLM_API_KEY \
    --request-log-path artifacts/rm_server_logs/requests.jsonl

#!/usr/bin/env bash
# Test RM server health and scoring endpoints.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SERVER_URL="${1:-http://localhost:8080}"
ARTIFACT="${2:-artifacts/rm_artifact_writing_260501.json}"
HOST="${3:-127.0.0.1}"
PORT="${4:-8080}"
PID=""

cleanup() {
    if [[ -n "${PID:-}" ]]; then
        echo ""
        echo "Stopping background server (PID $PID)..."
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if command -v uv >/dev/null 2>&1; then
    PYTHON_PREFIX=(uv run)
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_PREFIX=("${VIRTUAL_ENV}/bin/python")
else
    PYTHON_PREFIX=("${PYTHON_BIN:-python3}")
fi

# ---------------------------------------------------------------------------
# Start server if not already running
# ---------------------------------------------------------------------------
if ! curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/healthz" 2>/dev/null | grep -q '^2'; then
    echo "Server not running. Starting background server..."

    if [[ -z "${LLM_API_KEY:-}" ]]; then
        echo "ERROR: LLM_API_KEY is not set." >&2
        exit 1
    fi

    if [[ ! -f "$ARTIFACT" ]]; then
        echo "Artifact not found. Generating..."
        "${PYTHON_PREFIX[@]}" python scripts/convert_writing_rubrics_to_artifact.py --output "$ARTIFACT"
    fi

    "${PYTHON_PREFIX[@]}" python -m reward_harness.rm.server \
        --artifact "$ARTIFACT" \
        --host "$HOST" \
        --port "$PORT" \
        --api-key-env LLM_API_KEY \
        --request-log-path artifacts/rm_server_logs/requests.jsonl \
        --log-level warning &
    PID=$!

    echo "Waiting for server to be ready (PID $PID)..."
    for i in {1..30}; do
        if curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/healthz" 2>/dev/null | grep -q '^2'; then
            echo "Server is ready."
            break
        fi
        sleep 1
    done

    if ! curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/healthz" 2>/dev/null | grep -q '^2'; then
        echo "ERROR: Server failed to start within 30 seconds." >&2
        exit 1
    fi
else
    echo "Server is already running at $SERVER_URL"
fi

# ---------------------------------------------------------------------------
# Test payloads
# ---------------------------------------------------------------------------
SCORE_PAYLOAD=$(cat <<'EOF'
{
  "prompt_id": "writing_0001",
  "prompt": "请为以下研究主题撰写一份符合《智能制造》期刊格式要求的完整论文大纲：基于机器学习的智能制造质量控制系统研究。",
  "candidate": {
    "candidate_id": "test_candidate_001",
    "text": "标题：基于深度学习的智能制造质量控制系统研究\n\n摘要：本文针对智能制造过程中的质量控制问题，提出了一种基于深度学习的智能质量控制方法。通过分析生产过程中的多源数据，构建了深度神经网络模型，实现了对产品质量的实时预测和异常检测。实验结果表明，所提方法在准确率、召回率等指标上均优于传统统计过程控制方法。\n\n关键词：智能制造；质量控制；深度学习；异常检测\n\n1. 引言\n1.1 研究背景与意义\n1.2 国内外研究现状\n1.3 研究内容与论文结构\n\n2. 相关理论与技术基础\n2.1 智能制造概述\n2.2 质量控制理论\n2.3 深度学习基础\n\n3. 智能制造质量控制问题分析\n3.1 生产过程数据特征分析\n3.2 质量控制难点与挑战\n\n4. 基于深度学习的质量控制模型\n4.1 模型架构设计\n4.2 数据预处理与特征工程\n4.3 模型训练与优化策略\n\n5. 实验与结果分析\n5.1 实验环境与数据集描述\n5.2 评价指标定义\n5.3 实验结果对比分析\n\n6. 结论与展望\n6.1 主要工作总结\n6.2 未来研究方向\n\n参考文献",
    "source": "test"
  }
}
EOF
)

BATCH_SCORE_PAYLOAD=$(cat <<'EOF'
{
  "items": [
    {
      "prompt_id": "writing_0001",
      "prompt": "请为以下研究主题撰写一份符合《智能制造》期刊格式要求的完整论文大纲：基于机器学习的智能制造质量控制系统研究。",
      "candidate": {
        "candidate_id": "test_candidate_001",
        "text": "标题：基于深度学习的智能制造质量控制系统研究\n\n摘要：本文针对智能制造过程中的质量控制问题，提出了一种基于深度学习的智能质量控制方法。\n\n关键词：智能制造；质量控制；深度学习\n\n1. 引言\n2. 相关理论与技术基础\n3. 智能制造质量控制问题分析\n4. 基于深度学习的质量控制模型\n5. 实验与结果分析\n6. 结论与展望\n\n参考文献",
        "source": "test"
      }
    },
    {
      "prompt_id": "writing_0002",
      "prompt": "Write a comprehensive paper outline for a study on intelligent building energy-saving control using deep reinforcement learning, targeting the Building and Environment journal.",
      "candidate": {
        "candidate_id": "test_candidate_002",
        "text": "Title: Deep Reinforcement Learning for Intelligent Building Energy-Saving Control\n\nAbstract: This paper presents a DRL-based HVAC control system for office buildings. We analyze six months of operational data and demonstrate significant energy savings.\n\nKeywords: Deep reinforcement learning, HVAC, energy saving, building control\n\n1. Introduction\n2. Literature Review\n3. Methodology\n4. Experimental Setup\n5. Results and Discussion\n6. Conclusion\n\nReferences",
        "source": "test"
      }
    }
  ]
}
EOF
)

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Test 1: GET /healthz"
echo "========================================"
curl -s -m 10 "${SERVER_URL}/healthz" | python3 -m json.tool

echo ""
echo "========================================"
echo "Test 2: POST /score (single candidate)"
echo "========================================"
echo "Prompt ID: writing_0001 (Chinese paper outline)"
echo "This may take 30-120s depending on LLM latency..."
curl -s -m 300 -X POST "${SERVER_URL}/score" \
    -H "Content-Type: application/json" \
    -d "$SCORE_PAYLOAD" | python3 -m json.tool

echo ""
echo "========================================"
echo "Test 3: POST /batch_score (2 candidates)"
echo "========================================"
echo "Prompt IDs: writing_0001, writing_0002"
echo "This may take 60-240s depending on LLM latency..."
curl -s -m 300 -X POST "${SERVER_URL}/batch_score" \
    -H "Content-Type: application/json" \
    -d "$BATCH_SCORE_PAYLOAD" | python3 -m json.tool

echo ""
echo "========================================"
echo "All tests completed."
echo "========================================"

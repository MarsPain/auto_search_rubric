# auto_search_rubric

[English](README.md) | 中文

面向 **Rubric-based Reward Modeling** 的自动搜索框架，灵感来自论文：

[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](https://arxiv.org/abs/2509.21500)

本仓库保留了对照基线 `Iterative RTD`，并默认采用更可扩展的 `Evolutionary RTD` 搜索实现（`--mode evolutionary`）。

> 原论文 Iterative RTD 开源项目：[Jun-Kai-Zhang/rubrics](https://github.com/Jun-Kai-Zhang/rubrics)

## 重点

- 结构化 Rubric Schema：`criteria`、`weights`、`grading_protocol`、正/反例与检查点
- 多票验证聚合：多数投票 + vote-level 方差统计
- Tail-focused 目标函数：
  - `TailAcc`
  - `TailVar`
  - `DiverseTailAcc`
- 支持在数据集中预定义候选答案偏好排序（`metadata.rank`），自动启用 `RankPreferenceJudge`

## Evolutionary RTD 与论文 Iterative RTD 的区别

> 说明：本仓库实现了 `iterative` 模式作为基线，同时提供默认 `evolutionary` 模式用于更强搜索。

### 1) 更易拓展的架构设计

- 基于协议接口解耦：`RubricInitializer`、`RubricProposer`、`Verifier`、`PreferenceJudge`
- 同一套搜索流程可替换 mock / LLM 组件
- 支持 role-based 模型配置（initializer/proposer/verifier/judge 可分别指定模型）

### 2) Evolutionary 搜索方式（非单路径迭代）

- **变异策略 + 精英保留（elitism）**
  - 维护 rubric 候选群体（population），不是只沿单条轨迹迭代
  - 每轮生成多个变异 rubric（`mutations_per_round`）
  - 保留精英 rubric（`elitism_count`）并补充优胜变体，降低陷入局部路径的风险

- **预算筛选机制（successive halving）**
  - 先粗评大量候选（小预算 pair 采样）
  - 再精评少量候选（中预算到全预算）
  - 通过 `pair_budget_small / medium / full` 实现"省预算"的候选筛选

- **难例聚焦（hard prompt selection）**
  - 每代优先优化"难区分"的 prompt（综合 top margin + 群体分歧度）
  - 用有限预算集中攻克区分难题样本

### 3) 支持预定义 candidates 的 Preference Rank

- 当数据集中 **所有 candidate** 都带有 `metadata.rank`（数值越小越好）时：
  - 自动启用 `RankPreferenceJudge`
  - 即使在 `llm` backend 下，也会优先使用 rank 作为偏好真值，不必调用 LLM judge
- 若任一 candidate 缺失 `rank`，则回退到默认 judge（mock 下为启发式，llm 下为 LLM judge）

## 项目结构

- `autosr/`：核心包（CLI、搜索、评估、LLM/mock 组件）
- `tests/`：`unittest` 测试
- `scripts/`：测试与 formal 运行脚本
- `examples/`：示例数据集（含带/不带 rank 的版本）
- `artifacts/`：默认输出目录

## 安装

要求：Python `>=3.11`

```bash
python3 -m pip install -e .
```

安装后可使用以下两种方式运行：

- `python3 -m autosr.cli ...`
- `autosr ...`

## 快速开始

默认推荐：Evolutionary 搜索

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json
```

Iterative 基线对照：

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode iterative \
  --output artifacts/best_rubrics_iterative.json
```

## LLM 后端

CLI 支持 `--backend {auto,mock,llm}`：

- `auto`（默认）：检测到 `LLM_API_KEY` 则用 `llm`，否则 `mock`
- `llm`：强制使用 LLM，缺少 key 时直接报错
- `mock`：强制使用本地组件

默认配置使用 OpenRouter 兼容的端点。你可以通过 `--base-url` 参数覆盖为任何 OpenAI 兼容的 API 提供商。

运行 formal 流程（需要 API Key）：

```bash
export LLM_API_KEY="<YOUR_API_KEY>"
./scripts/run_formal_search.sh examples/single_case.json evolutionary artifacts/best_rubrics_formal.json
```

直接通过 CLI 指定角色模型：

```bash
python3 -m autosr.cli \
  --dataset examples/single_case.json \
  --mode evolutionary \
  --output artifacts/best_rubrics_llm.json \
  --backend llm \
  --base-url https://openrouter.ai/api/v1 \
  --model-default deepseek/deepseek-v3.2 \
  --model-initializer deepseek/deepseek-v3.2 \
  --model-proposer deepseek/deepseek-v3.2 \
  --model-verifier deepseek/deepseek-v3.2 \
  --model-judge deepseek/deepseek-v3.2 \
  --llm-timeout 30 \
  --llm-max-retries 2
```

## 搜索与目标函数（实现细节）

- 目标函数：
  - `total = TailAcc - lambda_var * TailVar + mu_diverse * DiverseTailAcc`
- 默认关键参数（CLI）：
  - `--generations 12`
  - `--population-size 8`
  - `--mutations-per-round 6`
  - `--batch-size 3`
  - `--tail-fraction 0.25`
  - `--lambda-var 0.2`
  - `--mu-diverse 0.25`
- 代码层面（`EvolutionaryConfig`）还包含但目前未暴露为 CLI 参数的配置项：
  - `survival_fraction`（每阶段保留比例）
  - `elitism_count`（精英保留数量）
  - `stagnation_generations`（停滞早停阈值）

## 数据格式

输入必须是 JSON，最外层含 `prompts`：

```json
{
  "prompts": [
    {
      "prompt_id": "p1",
      "prompt": "Write ...",
      "candidates": [
        {
          "candidate_id": "c1",
          "text": "response text",
          "source": "strong",
          "metadata": { "quality": 0.91, "rank": 1 }
        }
      ]
    }
  ]
}
```

字段说明：

- `prompt_id`、`prompt` 必填
- 每个 prompt 至少需要 2 个 `candidates`
- `metadata.quality`：用于启发式 judge（可选）
- `metadata.rank`：偏好排序标签（可选，`1` 表示最佳）

可参考：

- `examples/single_case.json`（quality 版本）
- `examples/single_case_with_rank.json`（rank 版本）

## 输出格式

结果会写入 `--output` 指定文件，结构如下：

- `best_rubrics`：每个 prompt 的最优 rubric
- `best_objective_scores`：每个 prompt 的目标分
- `best_scores`：`best_objective_scores` 的兼容别名
- `best_candidates`：每个 prompt 在最优 rubric 下的 top candidate id
- `candidate_scores`：最优 rubric 下所有 candidate 的分数
- `best_candidate_scores`：每个 prompt 的最高 candidate 分数

## 测试

运行全部单测：

```bash
./scripts/run_tests.sh
```

或：

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

说明：

- 设置了 `LLM_API_KEY` 时，会执行集成测试
- 未设置时，集成测试会自动跳过

## 备注

- 本仓库强调"可复现实验 + 可替换组件 + 预算感知搜索"。
- 若你想严格复现实验对照，可同时运行 `iterative` 与 `evolutionary`，比较 `best_scores` 与输出 rubric 差异。

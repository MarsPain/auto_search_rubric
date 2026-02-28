# auto_search_rubric

[English](README.md) | 中文

面向 Rubric-based Reward Modeling 的自动搜索框架，灵感来自论文：
[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](https://arxiv.org/abs/2509.21500)

本仓库保留 `iterative` 作为基线，并将 `evolutionary` 作为默认搜索模式。

## 亮点

- 使用类型化枚举（`autosr.types`）和分层配置 dataclass（`autosr.config`）统一运行配置
- 通过组合根 `ComponentFactory` 进行后端感知的依赖装配
- 领域模型以 `autosr.data_models` 为主，`autosr.models` 仅保留兼容导出
- 搜索策略可扩展：
  - 父代选择：`rank`、`tournament`、`top_k`
  - 自适应变异：`fixed`、`success_feedback`、`exploration_decay`、`diversity_driven`
- LLM 架构分离为传输配置（`autosr.llm_config`）与运行时配置（`autosr.config`）
- 可复现实验产物：
  - 输出 JSON 内嵌 `run_manifest`
  - `<output_parent>/run_records/` 下归档 manifest 与复现脚本

## 当前架构

### 入口与组合

- `autosr/cli.py`
  - 仅负责 CLI 参数解析
  - 构建 `RuntimeConfig`
  - 将运行时装配委托给 `ComponentFactory`
- `autosr/factory.py`
  - 统一组合根：后端选择与组件组装
  - 当所有候选都带 `metadata.rank` 时自动启用 rank judge

### 配置与类型

- `autosr/config.py`
  - 运行时配置：
    - `RuntimeConfig`
    - `LLMBackendConfig`
    - `SearchAlgorithmConfig`
    - `ObjectiveConfig`（兼容别名：`ObjectiveFunctionConfig`）
    - `InitializerStrategyConfig`、`ContentExtractionConfig`、`VerifierConfig`
- `autosr/llm_config.py`
  - LLM 传输/模型底层配置（`LLMConfig`、`RoleModelConfig`）
- `autosr/types.py`
  - 共享枚举：
    - `BackendType`、`SearchMode`、`SelectionStrategy`
    - `AdaptiveMutationSchedule`、`InitializerStrategy`、`ExtractionStrategy`、`LLMRole`

### 领域与共享模块

- `autosr/data_models.py`：规范领域实体（`Rubric`、`Criterion`、`PromptExample` 等）
- `autosr/models.py`：兼容导入层
- `autosr/exceptions.py`：共享 LLM 异常（`LLMCallError`、`LLMParseError`）
- `autosr/io_utils.py`：数据集/结果 I/O 与 run-record 持久化
- `autosr/run_records/use_cases.py`：run manifest 与可复现实验脚本生成

### 搜索域

- `autosr/search/config.py`：`IterativeConfig`、`EvolutionaryConfig`、`SearchResult`
- `autosr/search/iterative.py`：迭代基线实现
- `autosr/search/evolutionary.py`：进化搜索实现
- `autosr/search/strategies.py`：可复用搜索辅助策略
- `autosr/search/selection_strategies.py`：父代选择策略
- `autosr/search/adaptive_mutation.py`：变异调度与多样性指标
- `autosr/search/use_cases.py`：searcher 对外入口导出

### LLM 与提取域

- `autosr/llm_components/base.py`：请求/重试基类与 prompt 回退渲染
- `autosr/llm_components/parsers.py`：响应归一化与校验
- `autosr/llm_components/use_cases.py`：initializer/proposer/verifier/judge 实现
- `autosr/llm_components/factory.py`：保留的兼容工厂
- `autosr/content_extraction/strategies.py`：`tag` / `regex` / `identity` 提取策略
- `autosr/content_extraction/use_cases.py`：带提取装饰器的 verifier
- `autosr/prompts/loader.py` + `autosr/prompts/constants.py`：模板加载与常量回退

## 项目结构

- `autosr/`：核心包
- `prompts/`：提示词模板（支持 `prompts/zh/`、`prompts/en/` 等 locale 子目录）
- `tests/`：`unittest` 测试集
- `scripts/`：单测/集测/formal 运行脚本
- `examples/`：示例数据与示例脚本
- `artifacts/`：默认输出目录

## 环境准备

要求：Python `>=3.11` 与 `uv`。

```bash
uv sync
```

建议统一使用 `uv run` 执行命令：

```bash
uv run python -m autosr.cli --help
```

## 快速开始

默认（evolutionary）：

```bash
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --output artifacts/best_rubrics.json
```

Iterative 基线：

```bash
uv run python -m autosr.cli \
  --dataset examples/single_case.json \
  --mode iterative \
  --output artifacts/best_rubrics_iterative.json
```

自定义选择策略与提示词语言：

```bash
uv run python -m autosr.cli \
  --dataset examples/single_case_with_rank.json \
  --mode evolutionary \
  --output artifacts/best_rubrics_rank.json \
  --selection-strategy top_k \
  --adaptive-mutation diversity_driven \
  --prompt-language zh
```

## 后端与 LLM 配置

`--backend {auto,mock,llm}`：

- `auto`（默认）：检测到 API key 用 `llm`，否则 `mock`
- `llm`：必须提供 API key（默认读取 `LLM_API_KEY`，可用 `--api-key-env` 改名）
- `mock`：仅本地启发式组件

默认端点与模型：

- `--base-url https://openrouter.ai/api/v1`
- `--model-default stepfun/step-3.5-flash:free`

支持按角色覆写模型：

- `--model-initializer`
- `--model-proposer`
- `--model-verifier`
- `--model-judge`

提示词模板加载顺序：

1. `prompts/<language>/`（设置 `--prompt-language` 时）
2. `prompts/`
3. 代码内置常量模板

LLM formal 流程示例：

```bash
export LLM_API_KEY="..."
./scripts/run_formal_search.sh \
  examples/call_summary_dataset_with_rank_single.json \
  evolutionary \
  artifacts/best_rubrics_formal_call_summary.json
```

## 搜索目标与常用控制参数

目标函数：

`score = TailAcc - lambda_var * TailVar + mu_diverse * DiverseTailAcc`

常用参数：

- `--generations`、`--population-size`、`--mutations-per-round`、`--batch-size`
- `--tail-fraction`、`--lambda-var`、`--mu-diverse`
- `--pair-confidence-prior`（pairwise 置信收缩，设为 `0` 可关闭）
- `--selection-strategy {rank,tournament,top_k}`
- `--adaptive-mutation {fixed,success_feedback,exploration_decay,diversity_driven}`

## 数据格式

输入 JSON 顶层需要包含 `prompts`：

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

说明：

- `prompt_id` 与 `prompt` 必填
- 每个 prompt 至少 2 个 candidates
- `metadata.rank` 可选（`1` 最优）；若所有候选都提供，则自动切换 rank judge

## 输出与可复现

主输出 JSON（`--output`）包含：

- `best_rubrics`（数组；每项可包含 `best_candidate_id` 与 `candidate_scores`）
- `best_objective_scores`
- `best_scores`（`best_objective_scores` 的兼容别名）
- 可选 `run_manifest`

每次运行还会在如下目录生成可复现文件：

- `<output_parent>/run_records/<output_stem>_<run_id>.manifest.json`
- `<output_parent>/run_records/<output_stem>_<run_id>.reproduce.sh`

## 测试

单元测试：

```bash
./scripts/run_tests_unit.sh
```

集成测试（需要 API key）：

```bash
export LLM_API_KEY="..."
./scripts/run_tests_integration.sh
```

聚合入口：

```bash
./scripts/run_tests.sh
```

直接执行全量测试：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

架构回归测试集合：

```bash
uv run python -m unittest \
  tests.test_architecture_refactor \
  tests.test_cli_backend_selection \
  tests.test_cli_best_candidates \
  tests.test_io_utils \
  tests.test_search_config_enum_unification \
  tests.test_data_models_compat \
  tests.test_exceptions_module \
  tests.test_evolutionary_decoupling
```

## 备注

- 新代码优先从 `autosr.data_models` 导入领域实体。
- 运行时装配优先使用 `ComponentFactory(RuntimeConfig(...))`，避免手工拼装依赖。
- 密钥只通过环境变量管理（`LLM_API_KEY`，以及可选 `LLM_BASE_URL`、`LLM_MODEL`）。

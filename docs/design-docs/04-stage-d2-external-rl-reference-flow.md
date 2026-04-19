# AutoSR Stage D2: External RL Repo Reference Flow

> **版本**: v1.0 | **状态**: 已实现 | **最后更新**: 2026-04-19
>
> 本文档描述外部 RL 仓库（以 verl 为例）如何接入 AutoSR RM Server 与实验注册表。

---

## 1. 概述

Stage D2 在 Stage D1（Contract & Registry）基础上，提供**参考接入链实现**。核心目标：

- 外部 trainer 能稳定消费 RM endpoint
- 训练前自动生成 `TrainingManifest` 并预检握手
- 训练后自动回填 `TrainingResultManifest` / `EvalReport`

**非目标**：

- 不在 `autosr` 内实现 trainer 或 optimizer
- 不约束外部 RL repo 的内部框架选型
- 提供的脚本为参考实现，外部项目可重新实现

---

## 2. 参考目录约定

### 2.1 AutoSR 侧（已 D1 落地）

```text
artifacts/
  training_runs/
    manifests/   -> TrainingManifest 文件
    results/     -> TrainingResultManifest 文件
    evals/       -> EvalReport 文件
    index/       -> LineageIndex 派生文件
```

### 2.2 外部 RL Repo 侧（推荐结构）

```text
outputs/
  <training_run_id>/
    checkpoints/     -> 训练 checkpoint
    logs/            -> 训练日志
    eval/            -> 评测结果
    manifests/       -> 本地 manifest 副本
```

约定：

- RL repo 对 checkpoint、日志、模型文件等大件产物负责
- `autosr` 只保存结构化 JSON 记录和路径引用
- `autosr` 不把 RL repo 的目录结构当作硬契约

---

## 3. 参考交互时序

### 3.1 三阶段握手

```text
Phase 1: Prepare
  -> generate training_run_id
  -> create local output dirs
  -> RM server healthz check (artifact_id + source_session_id 校验)
  -> build & record TrainingManifest

Phase 2: Train
  -> RL trainer consumes RM endpoint via reward_client
  -> all outputs go to outputs/<training_run_id>/

Phase 3: Finalize
  -> build TrainingResultManifest
  -> optional: build EvalReport(s)
  -> record both into autosr registry
```

### 3.2 时序图

```text
External RL Repo                          AutoSR
    |                                        |
    |  1) GET /healthz                       |
    |--------------------------------------->|
    |     {artifact_id, source_session_id,   |
    |      rm_api_version, status: ok}       |
    |<---------------------------------------|
    |  2) validate consistency               |
    |                                        |
    |  3) POST /score (rollout step)         |
    |--------------------------------------->|
    |     {request_id, score, ...}           |
    |<---------------------------------------|
    |  4) POST /batch_score (batch step)     |
    |--------------------------------------->|
    |     {request_id, results, ...}         |
    |<---------------------------------------|
    |                                        |
    |  5) write TrainingResultManifest       |
    |  6) record_result (CLI / API)          |
    |--------------------------------------->|
    |     saved to artifacts/training_runs/  |
    |<---------------------------------------|
    |  7) write EvalReport                   |
    |  8) record_eval (CLI / API)            |
    |--------------------------------------->|
    |     saved to artifacts/training_runs/  |
    |<---------------------------------------|
```

---

## 4. RM Server 预检握手

### 4.1 Healthz 响应格式

`GET /healthz` 返回：

```json
{
  "status": "ok",
  "artifact_id": "artifact_001",
  "source_session_id": "session_001",
  "schema_version": "1.0",
  "rm_api_version": "1.0"
}
```

### 4.2 校验逻辑

外部 trainer 在启动前应：

1. 调用 `GET /healthz`
2. 确认 `status == "ok"`
3. 将返回的 `artifact_id` 与本地 `TrainingManifest` 中的 `rm_artifact_id` 比对
4. 将返回的 `source_session_id` 与本地 `TrainingManifest` 中的 `search_session_id` 比对
5. 任一不匹配则拒绝启动，记录 `failure.stage = preflight`

参考实现见 `autosr/rl/verl/reward_client.py` 中的 `RMScoringClient.healthz_check()`。

---

## 5. 参考脚本职责

### 5.1 prepare_training_run

**路径**: `python -m autosr.rl.verl.prepare_training_run`

**职责**:
- 生成或接收 `training_run_id`
- 创建本地输出目录结构
- 执行 RM server healthz 预检
- 构建并校验 `TrainingManifest`
- 将 manifest 记录到 registry

**关键参数**:
- `--rm-endpoint`, `--rm-artifact-id`, `--search-session-id`
- `--dataset-id`, `--dataset-version`
- `--trainer-project`, `--trainer-code-version`, `--trainer-entrypoint`
- `--output-dir`

### 5.2 finalize_training_run

**路径**: `python -m autosr.rl.verl.finalize_training_run`

**职责**:
- 读取训练状态（成功/失败/取消）
- 构建 `TrainingResultManifest`
- 可选读取并记录 `EvalReport`
- 回填到 registry

**关键参数**:
- `--training-run-id`, `--status`, `--started-at`, `--finished-at`, `--duration-seconds`
- `--trainer-code-version`
- `--checkpoint-path`, `--log-path`, `--model-artifact-path`
- `--training-summary-json`, `--reward-summary-json`
- `--failure-type`, `--failure-message`, `--failure-stage`（失败时必填）
- `--eval-report-json`（可选）

### 5.3 run_verl_training（编排脚本）

**路径**: `python -m autosr.rl.verl.run_verl_training`

**职责**:
- 串联 prepare → train → finalize 三阶段
- 向 trainer 子进程注入环境变量（`RM_ENDPOINT`, `TRAINING_RUN_ID` 等）
- 如果 prepare 失败，自动记录 preflight failure
- 如果 trainer 失败，自动记录 training failure

**用法**:

```bash
python -m autosr.rl.verl.run_verl_training \
  --rm-endpoint http://127.0.0.1:8080 \
  --rm-artifact-id artifact_001 \
  --search-session-id session_001 \
  --dataset-id gsm8k \
  --dataset-version v1.0 \
  --trainer-project verl_grpo \
  --trainer-code-version abc123 \
  --output-dir outputs \
  -- \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    ...
```

### 5.4 reward_client（可复用客户端）

**路径**: `autosr.rl.verl.reward_client.RMScoringClient`

**职责**:
- 封装与 RM server 的 HTTP 通信
- 仅依赖 Python 标准库（`urllib`），便于 vendoring
- 支持 healthz、score、batch_score
- 支持 artifact_id / source_session_id 一致性校验

**使用示例**:

```python
from autosr.rl.verl.reward_client import RMScoringClient

client = RMScoringClient(
    endpoint="http://127.0.0.1:8080",
    expected_artifact_id="artifact_001",
    expected_source_session_id="session_001",
)

# 预检
healthz = client.healthz_check()

# 单条打分
result = client.score(
    prompt_id="prompt_001",
    prompt="What is 2+2?",
    candidate_id="resp_001",
    text="The answer is 4.",
)
score = result["score"]
```

---

## 6. verl 框架 Reward 集成建议

verl 的 reward function 通常通过自定义模块或数据管道注入。参考集成方式：

### 6.1 方案 A：数据预处理阶段注入

在数据加载或预处理阶段，对每条 prompt/response 调用 RM server 预计算 reward，写入 parquet 的 `reward` 列。训练时直接读取。

**优点**: 训练稳定，不依赖在线服务
**缺点**: 无法动态采样，不支持 PPO 类 on-policy 算法

### 6.2 方案 B：自定义 Reward Function（推荐）

在 verl 的 trainer 配置中指定自定义 reward function，该函数内部调用 `RMScoringClient`。

参考代码片段：

```python
# In your verl project: my_project/reward_fn.py
from autosr.rl.verl.reward_client import RMScoringClient

_client = None

def _get_client():
    global _client
    if _client is None:
        import os
        _client = RMScoringClient(
            endpoint=os.environ["RM_ENDPOINT"],
            expected_artifact_id=os.environ.get("RM_ARTIFACT_ID", ""),
        )
    return _client

def compute_reward(prompt: str, response: str) -> float:
    client = _get_client()
    result = client.score(
        prompt_id="auto",  # or map from dataset
        prompt=prompt,
        candidate_id="auto",
        text=response,
    )
    return result["score"]
```

然后在 verl 配置中引用此函数。

### 6.3 方案 C：Ray Actor Reward Model

对于大规模分布式训练，可将 `RMScoringClient` 包装为 Ray Actor，在 rollout 后异步批量调用 `/batch_score`。

---

## 7. 失败恢复

### 7.1 启动前失败（preflight）

- RM server 不可达
- manifest 校验失败
- artifact_id / session_id 不匹配

处理：
- `prepare_training_run` 会自动报错并退出
- `run_verl_training` 编排脚本会捕获失败并记录 `TrainingResultManifest(status=failed, failure.stage=preflight)`

### 7.2 训练中失败（training）

- GPU OOM
- trainer crash
- reward timeout

处理：
- trainer 进程崩溃后，`run_verl_training` 会捕获非零退出码
- 自动调用 `finalize_training_run` 记录 `failure.stage=training`
- 建议 RL repo 自行实现 checkpoint resume 策略

### 7.3 评测中失败（eval）

- benchmark runner 崩溃
- 评测数据损坏

处理：
- 训练结果可独立成功
- 评测失败不污染训练主记录
- 可重新运行评测并追加新的 `EvalReport`

---

## 8. 环境变量约定

`run_verl_training` 向 trainer 子进程注入以下环境变量：

| 变量名 | 说明 |
|--------|------|
| `RM_ENDPOINT` | RM server HTTP endpoint |
| `RM_ARTIFACT_ID` | 当前使用的 artifact ID |
| `TRAINING_RUN_ID` | 当前训练 run ID |
| `TRAINING_RUN_DIR` | 本地输出目录 |

外部 RL repo 的 reward function 可通过 `os.environ` 读取这些变量。

---

## 9. 与现有文档的关系

- [02-stage-d-rl-lineage.md](02-stage-d-rl-lineage.md): Stage D 总体设计，定义 TrainingManifest / TrainingResultManifest / EvalReport schema
- [../ROADMAP.md](../ROADMAP.md): 路线图与阶段划分
- [../ARCHITECTURE.md](../ARCHITECTURE.md): 顶层域地图

---

*本文档为 Stage D2 实现完成后的参考文档，随项目进展持续更新。*

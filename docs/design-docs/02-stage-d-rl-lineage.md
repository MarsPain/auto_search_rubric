# Reward Harness Stage D 设计：RL Handshake 与 Experiment Lineage

> **版本**: v1.1 | **状态**: 草案 | **最后更新**: 2026-05-01
>
> 本文档细化 Stage D 的设计边界：`reward_harness` 不实现 trainer，而是定义并维护 RL 训练接入所需的契约、registry 与 lineage 查询能力。

---

## 1. 设计目标

Stage C 已完成 `RM Artifact -> RM Server` 主链路。Stage D 的目标不是把外部 RL 项目并入 `reward_harness`，而是补齐以下能力：

1. 定义稳定的训练前后契约，使外部 RL repo 可以一致接入 RM server。
2. 建立 append-only 的实验记录平面，使训练结果可追溯、可比较、可审计。
3. 让任一训练或评测结果都能反查到 reward 来源链：
   `Training/Eval -> RM Deploy -> RM Artifact -> Search Session`

非目标：

- 不在 `reward_harness` 内实现 trainer 或 optimizer。
- 不在 `reward_harness` 内托管训练 checkpoint、原始日志、模型大文件。
- 不引入在线 experiment tracking service。
- 不约束外部 RL repo 的内部训练框架选型。

---

## 2. 方案选择

### 2.1 备选方案

#### 方案 A：Manifest-Only Handshake

- `reward_harness` 只定义 schema。
- 外部 RL repo 负责生成、保存、校验、查询所有训练记录。

优点：
- 边界最窄。
- 本仓库改动最少。

缺点：
- 记录格式和目录约定容易漂移。
- lineage 查询分散在多个项目中，难以统一。

#### 方案 B：Contract + Registry + Reference Flow

- `reward_harness` 定义训练前后契约。
- `reward_harness` 维护 append-only registry 与 lineage 查询能力。
- 文档化外部 RL repo 的参考交互时序、目录约定和回填流程。

优点：
- 保持清晰架构边界。
- 让追溯链在 `reward_harness` 中闭合。
- 与现有 `run_manifest` / `DeployManifest` 风格一致。

缺点：
- 需要新增一层 integration metadata plane。

#### 方案 C：Orchestrated Training Adapter

- `reward_harness` 内新增 orchestration 层，负责启动 trainer、注入 endpoint、收集结果。

优点：
- 端到端体验更完整。

缺点：
- 很容易把 trainer 生命周期和框架细节带入本仓库。
- 现阶段会过早扩大 `reward_harness` 的职责边界。

### 2.2 选定方案

本阶段采用 **方案 B：Contract + Registry + Reference Flow**。

结论：

- `reward_harness` 是 reward harness 与 lineage registry 的供给方。
- 外部 RL repo 是训练执行方。
- 双方通过版本化 manifest/result/report 契约握手，而不是通过代码耦合。

---

## 3. 系统边界与职责

Stage D 在 `reward_harness` 中新增的不是“训练子系统”，而是 **RL integration metadata plane**。

### 3.1 `reward_harness` 负责

- 定义 `TrainingManifest` 契约。
- 定义 `TrainingResultManifest` 契约。
- 定义 `EvalReport` 契约。
- 维护 append-only experiment registry。
- 提供 lineage 查询能力：
  `training_run -> rm_deploy -> rm_artifact -> search_session`
- 作为 RM server 的供给方，暴露稳定 reward endpoint。

### 3.2 外部 RL repo 负责

- 训练实现、checkpoint、resume、retry。
- 调用 RM server 获取 reward。
- 产出训练结果与评测指标。
- 依据 `reward_harness` 契约回填 manifest/result/report。

### 3.3 主数据流

```text
SearchSession
  -> RMArtifact
    -> DeployManifest
      -> RM Server
        -> TrainingManifest
          -> TrainingResultManifest
            -> EvalReport
```

关键约束：

- `TrainingManifest` 表示训练开始前的声明。
- `TrainingResultManifest` 表示训练结束后的事实。
- `EvalReport` 表示可比较的评测结果。

三者不能合并成一个对象，否则 planned state、observed state 与 comparison state 会混在一起。

---

## 4. 核心对象与目录约定

### 4.1 `TrainingManifest`

用途：训练开始前登记“这次实验准备怎么跑”。

建议字段（v1）：

- `schema_version`
- `training_run_id`
- `created_at_utc`
- `rm_artifact_id`
- `rm_deploy_id`
- `search_session_id`
- `rm_endpoint`
- `dataset`
  - `dataset_id`
  - `dataset_version`
  - `split`
  - `dataset_hash` optional
- `trainer`
  - `project`
  - `repo_url` optional
  - `code_version`
  - `entrypoint`
- `trainer_config`
- `execution`
  - `launcher`
  - `host`
  - `accelerator`
  - `num_workers`
- `tags`
- `notes` optional

设计说明：

- `rm_artifact_id` 表示训练所绑定的 reward 版本。
- `rm_deploy_id` 表示训练所连接的实际服务部署态。
- `search_session_id` 用于向上游搜索会话回溯。

### 4.2 `TrainingResultManifest`

用途：训练结束后登记“这次实验实际发生了什么”。

建议字段（v1）：

- `schema_version`
- `training_run_id`
- `status`
- `started_at_utc`
- `finished_at_utc`
- `duration_seconds`
- `trainer_code_version`
- `output`
  - `checkpoint_path` optional
  - `model_artifact_path` optional
  - `log_path` optional
- `reward_summary`
- `training_summary`
- `failure` optional
  - `type`
  - `message`
  - `stage`

设计说明：

- 必须支持 `succeeded`、`failed`、`canceled` 三类终态。
- 失败训练也必须回填，不允许“无结果即失败”。

### 4.3 `EvalReport`

用途：对训练产物的评测结果进行标准化记录，支持横向比较。

建议字段（v1）：

- `schema_version`
- `eval_run_id`
- `training_run_id`
- `evaluated_at_utc`
- `benchmark`
  - `name`
  - `version`
  - `split`
- `metrics`
- `artifacts`
  - `report_path` optional
  - `raw_predictions_path` optional
- `summary`
- `comparison_baseline` optional

设计说明：

- 一个 training run 可以关联多个 `EvalReport`。
- `EvalReport` 不等于训练结果；评测可能延后执行，也可能针对多个 benchmark 重复执行。

### 4.4 可选对象：`LineageIndex`

用途：加速查询；不是 source of truth。

建议关系展开：

- `training_run_id -> rm_artifact_id`
- `training_run_id -> rm_deploy_id`
- `training_run_id -> search_session_id`
- `training_run_id -> eval_run_ids`

### 4.5 `reward_harness` 侧目录约定

建议在现有 `artifacts/` 下新增：

```text
artifacts/
  training_runs/
    manifests/
      <training_run_id>.json
    results/
      <training_run_id>.json
    evals/
      <eval_run_id>.json
    index/                      # optional
```

约定：

- `manifests/<training_run_id>.json` 存 `TrainingManifest`
- `results/<training_run_id>.json` 存 `TrainingResultManifest`
- `evals/<eval_run_id>.json` 存 `EvalReport`

### 4.6 外部 RL repo 参考目录约定

推荐结构：

```text
configs/training/
scripts/
outputs/
  <training_run_id>/
    checkpoints/
    logs/
    eval/
manifests/
  <training_run_id>.training.json   # optional local copy
```

约束：

- RL repo 对 checkpoint、日志、模型文件等大件产物负责。
- `reward_harness` 只保存结构化 JSON 记录和路径引用。
- `reward_harness` 不把 RL repo 的目录结构当作硬契约。

---

## 5. 参考交互时序与回填流程

### 5.1 阶段 1：准备训练上下文

1. RL repo 选定 `rm_artifact_id` 或 `rm_deploy_id`。
2. 通过 `reward_harness` 侧流程记录一份 `TrainingManifest`。
3. RL repo 将该 manifest 作为本次训练的只读上下文使用。

关键约束：

- 训练开始前必须先存在 `TrainingManifest`。
- `training_run_id` 一旦生成，后续 checkpoint、日志、评测都必须挂在该 ID 下。

### 5.2 阶段 2：连接 RM server

1. RL repo 从 `TrainingManifest` 读取 `rm_endpoint`。
2. 训练进程启动时先调用 `/healthz`。
3. trainer 将服务返回的 `artifact_id` 与本地 manifest 中的记录进行一致性校验。
4. 校验通过后开始 rollout / reward 调用。

建议 RM server 在健康检查或 metadata 响应中返回：

- `artifact_id`
- `source_session_id`
- `rm_api_version`

### 5.3 阶段 3：训练执行

1. trainer 正常训练并记录本地日志、checkpoint、异常。
2. 所有输出写入 RL repo 本地 `outputs/<training_run_id>/...`。
3. 中途失败也必须保留最后已知状态，便于回填失败结果。

设计原则：

- `reward_harness` 不负责训练恢复。
- RL repo 自行实现 retry / resume 策略。

### 5.4 阶段 4：训练结果回填

训练结束后，无论成功还是失败，都回填一份 `TrainingResultManifest`。

- 成功态：写训练摘要、输出路径、reward summary、训练统计。
- 失败态：写失败阶段、异常类型、错误消息和最后可用状态 optional。

约束：

- 回填是 append-only，不覆盖 `TrainingManifest`。
- `TrainingResultManifest.training_run_id` 必须与 manifest 同 ID。

### 5.5 阶段 5：评测与回填

1. RL repo 或独立 eval job 对训练产物做评测。
2. 每次评测生成单独的 `EvalReport`。
3. `EvalReport` 通过 `training_run_id` 关联训练，通过 baseline 字段支持横向比较。

设计说明：

- 不要求训练结束立即评测。
- 同一训练 run 允许挂多个 benchmark、多次评测。

### 5.6 推荐时序图

```text
RMArtifact
  -> DeployManifest
    -> RM Server
      -> TrainingManifest
        -> RL Trainer
          -> TrainingResultManifest
            -> EvalReport
```

---

## 6. 契约细化与关键校验规则

### 6.1 `TrainingManifest` 校验

结构完整性：

- `training_run_id` 非空且全局唯一
- `rm_artifact_id` 非空
- `search_session_id` 非空
- `rm_endpoint` 为合法 URL
- `trainer.project`、`trainer.code_version`、`trainer.entrypoint` 非空
- `dataset.dataset_version` 非空

引用完整性：

- `rm_artifact_id` 必须能关联到已知 `RMArtifact`
- 若提供 `rm_deploy_id`，必须能关联到对应 `DeployManifest`
- `search_session_id` 必须与 `RMArtifact.source_session_id` 一致

冻结约束：

一旦 manifest 创建，以下字段不可修改：

- `training_run_id`
- `rm_artifact_id`
- `rm_deploy_id`
- `search_session_id`
- `dataset.dataset_version`
- `trainer.code_version`

### 6.2 `TrainingResultManifest` 校验

结构完整性：

- `training_run_id` 非空
- `status` 必须属于 `succeeded | failed | canceled`
- `started_at_utc`、`finished_at_utc` 必填
- `duration_seconds >= 0`

引用一致性：

- 必须能找到同 ID 的 `TrainingManifest`
- `trainer_code_version` 默认应与 manifest 中一致
- 若运行时发生代码热修复，需要显式记录 override 原因

状态约束：

- `succeeded` 时：
  - `training_summary` 必填
  - `output` 中至少存在一个可定位产物
- `failed` 时：
  - `failure.type`、`failure.message`、`failure.stage` 必填
- `canceled` 时：
  - 必须记录取消原因或取消发起者

### 6.3 `EvalReport` 校验

- `eval_run_id` 全局唯一
- `training_run_id` 必须关联到已存在 training run
- `benchmark.name`、`benchmark.version`、`metrics` 非空
- `evaluated_at_utc` 必填
- 若声明 `comparison_baseline`，baseline 必须可解析

指标建模原则：

- `metrics` 优先采用扁平 key-value 结构
- 避免在 v1 中引入深层嵌套指标对象

### 6.4 跨对象一致性规则

- `TrainingManifest.search_session_id == RMArtifact.source_session_id`
- `TrainingManifest.rm_artifact_id == DeployManifest.artifact_id`（若提供 deploy）
- `TrainingResultManifest.training_run_id == TrainingManifest.training_run_id`
- `EvalReport.training_run_id == TrainingResultManifest.training_run_id`
- 同一 `training_run_id` 可对应多个 `EvalReport`
- 同一 `rm_artifact_id` 可对应多个 `training_run_id`

### 6.5 Schema 演进策略

- v1 起始版本统一为 `schema_version = "1.0"`
- 新增 optional 字段：视为向后兼容演进
- 修改必填字段语义：必须升级 schema 版本
- 历史记录 append-only 保存，不做原地覆盖迁移

---

## 7. 接口与命令约定

### 7.1 运行时服务接口

由 RM server 提供：

- `GET /healthz`
- `POST /score`
- `POST /batch_score`

建议在 metadata 响应中返回：

- `artifact_id`
- `source_session_id`
- `rm_api_version`

用途：让 trainer 在开跑前确认所连接的 reward 服务就是预期部署态。

### 7.2 元数据写入接口

Stage D 起步阶段不新增 HTTP 写接口，采用文件回填：

- 外部 RL repo 生成结构化 JSON
- 通过 `reward_harness` CLI 或脚本落入约定目录
- `reward_harness` 负责 schema 与引用校验

原因：

- 避免过早引入 experiment-tracking service
- 保持单机闭环可跑

### 7.3 查询接口

建议提供最小 CLI 面：

```bash
# 记录训练前 manifest
uv run python -m reward_harness.rl.record_manifest \
  --manifest /path/to/training_manifest.json

# 回填训练结果
uv run python -m reward_harness.rl.record_result \
  --result /path/to/training_result.json

# 回填评测报告
uv run python -m reward_harness.rl.record_eval \
  --report /path/to/eval_report.json

# 查询 lineage
uv run python -m reward_harness.rl.show_lineage \
  --training-run-id train_20260417_001
```

说明：

- 这些命令属于 integration metadata plane，不负责启动训练。
- 命令命名可在实现阶段微调，但职责边界应保持不变。

### 7.4 外部 RL repo 参考接入脚本职责

建议拆分为三个职责清晰的脚本：

- `prepare_training_run.py`
  - 创建本地输出目录
  - 生成或接收 `TrainingManifest`
  - 校验 RM server 可用性
- `run_training.py`
  - 执行 trainer 主流程
- `finalize_training_run.py`
  - 生成 `TrainingResultManifest`
  - 可选生成 `EvalReport`
  - 调用 `reward_harness` CLI 完成回填

这样可将训练逻辑与协议接入逻辑解耦。

---

## 8. 失败恢复与可观测性

### 8.1 失败恢复职责划分

- `reward_harness` 负责“记录不丢、事实可追溯”
- 外部 RL repo 负责“训练怎么恢复、怎么续跑”

### 8.2 失败场景

#### 启动前失败

例子：

- RM server 不可达
- manifest 校验失败
- 配置缺失

处理：

- 允许生成 `TrainingResultManifest(status=failed)`
- `failure.stage = preflight`

#### 训练中失败

例子：

- GPU OOM
- trainer crash
- reward timeout

处理：

- 写入 `failure.stage = training`
- 记录最后已知 checkpoint 路径 optional

#### 评测中失败

例子：

- benchmark runner 崩溃
- 评测数据损坏

处理：

- 训练结果可以成功
- 评测单独失败或缺失，不污染训练主记录

### 8.3 幂等与重试语义

`record_manifest`：

- 同一 `training_run_id` 默认拒绝覆盖
- 若 payload 完全一致，可视为幂等重放

`record_result`：

- 同一 `training_run_id` 默认只允许写一次终态
- 若未来支持修正，应采用显式 revision 或 supersede 机制

`record_eval`：

- 允许多个 `eval_run_id` 指向同一 `training_run_id`
- `eval_run_id` 自身必须唯一

### 8.4 最小可观测信息

训练上下文：

- `training_run_id`
- `rm_artifact_id`
- `rm_deploy_id`
- `search_session_id`
- `trainer.code_version`
- `dataset.dataset_version`

训练结果摘要：

- `status`
- `duration_seconds`
- `reward_summary`
- `training_summary`

评测摘要：

- benchmark 名称与版本
- 主指标
- baseline 对比 optional

---

## 9. Stage D 落地切分

### Stage D1：Contract & Registry

目标：建立训练记录体系。

交付：

- `TrainingManifest` schema
- `TrainingResultManifest` schema
- `EvalReport` schema
- append-only 落盘目录
- 基础校验与 lineage 查询

验收：

- 能从任一 `training_run_id` 反查到 `rm_artifact_id` 与 `search_session_id`
- 失败 run 可被记录和查询

### Stage D2：External RL Reference Flow

目标：打通外部 RL repo 的参考接入链。

交付：

- 参考交互时序
- 参考目录约定
- 参考回填脚本职责
- RM server 预检握手建议

验收：

- 外部 trainer 能稳定消费 RM endpoint
- 能自动回填 manifest/result/eval

### Stage D3：Comparative Experiment View

目标：具备基础实验编排与对比能力。

交付：

- lineage 查询 CLI 增强
- artifact / training / eval 维度比较视图
- baseline 对比约定

验收：

- 能横向比较不同 `rm_artifact_id` 的训练与评测结果
- 能快速识别回归或异常 run

---

## 10. 成功标准

Stage D 完成后，系统应能回答以下问题：

1. 这次训练使用了哪个 `RMArtifact`？
2. 该 artifact 来自哪个 `SearchSession`？
3. 它通过哪个 deploy 进入服务态？
4. 训练使用了哪个 dataset version 和 code version？
5. 训练最终成功、失败还是取消？
6. 训练后的评测结果如何？
7. 相比 baseline 或上一个 artifact，表现是变好还是变差？

如果这些问题仍无法稳定回答，则说明 Stage D 还未真正闭环。

---

## 11. 与现有文档的关系

- [docs/ARCHITECTURE.md](../ARCHITECTURE.md)：提供稳定的顶层域地图。
- [docs/DESIGN.md](../DESIGN.md)：提供系统级设计目标与核心契约摘要。
- [docs/ROADMAP.md](../ROADMAP.md)：提供阶段性路线图与优先级。
- [docs/design-docs/01-architecture.md](01-architecture.md)：提供从 Harness 到 RM/RL 闭环的总体演进图。

本文件补足 Stage D 的详细设计，是后续执行计划与实现工作的直接输入。

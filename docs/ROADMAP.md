# AutoSR Roadmap: 从 Rubric Search 到 RM+RL 闭环

> **版本**: v1.1 | **最后更新**: 2026-04-17
> 
> 将 `autosr` 从"单次运行的 rubric 搜索器"演进为"可用于 RL 训练与评测的 Reward Harness"。

---

## 当前状态（截至 2026-04-17）

### 已经完成且建议保留

- ✅ `SearchSession` 生命周期管理（创建、执行、恢复）
- ✅ `SearchCheckpoint` v1 schema（含 config/dataset hash）
- ✅ `StateManager` 原子持久化
- ✅ CLI 的 `--resume-from` / `--checkpoint-every-generation`
- ✅ evolutionary + `global_batch` scope 的 step-wise 执行
- ✅ Harness 阶段 A 收尾：RNG 恢复、interval checkpoint、生效 resume 语义、scheduler state 恢复
- ✅ RMArtifact 阶段 B 核心落地：schema v1、导出命令、校验器（含 hash 一致性）
- ✅ RM deploy manifest 落地：独立 deploy CLI、部署记录 schema、按目标环境回填 previous artifact

### 这批改动的定位

- 它们是"**工程底座**"，不是"RM/RL 主业务闭环"。
- 结论：**保留并继续演进**，但后续路线必须切向 RM server、RL 驱动、classifier RM 蒸馏、训练/评测监控。

---

## 北极星目标（最终形态）

形成可重复的闭环：

1. 自动搜索与迭代 rubric（已有能力 + 持续增强）
2. 将当前最优 rubric 产出为可部署的 RM artifact（版本化）
3. 自动部署/更新 RM server（稳定评分 API）
4. RL 训练任务消费 RM server 进行 reward 计算
5. 基于 RL 采样自动蒸馏 classifier RM，降低 reward 成本并沉淀偏好数据
6. 监控 RL 训练、classifier RM 与评测表现，触发告警与回归分析
7. 根据评测结果决定是否再次触发 rubric search

---

## 设计原则（重定向后的硬约束）

1. **闭环优先**：任何新增能力必须服务 `Search -> RM -> RL -> Classifier RM -> Eval` 链路。
2. **契约先行**：先定义 artifact/API/指标契约，再做实现。
3. **可回滚**：每次 RM 发布、每次训练实验都可追溯到明确版本。
4. **兼容现有 CLI**：`uv run python -m autosr.cli` 继续可用。
5. **先单机后分布式**：未证明价值前不提前引入复杂基础设施。

---

## 阶段划分

### 阶段 A：Harness 底座稳定化 ✅ 已完成

目标：让搜索阶段可长时运行、可恢复、可复盘，作为上层 RM/RL 的可信输入。

#### 必做收尾（全部完成）
- [x] 修复 RNG 状态恢复逻辑（避免"恢复后轨迹漂移"）
- [x] 实现 `checkpoint_interval_seconds` 的真实生效逻辑
- [x] 明确 resume 行为契约（`continue_from_checkpoint` / `reseed_from_checkpoint`）
- [x] 完善 scheduler 可恢复状态（不仅 diagnostics）

---

### 阶段 B：RM Artifact 契约与部署 ✅ 已完成

目标：把"best rubric JSON"升级为可部署、可追溯的 RM artifact。

#### 关键任务
- [x] 定义 `RMArtifact` schema（v1）
  - `artifact_id`, `created_at`, `source_session_id`, `dataset_hash`, `config_hash`
  - `rubric`, `scoring_policy`, `normalization`, `compatibility`
- [x] 新增 artifact 导出能力（由 search 输出生成 RM artifact）
- [x] 增加 artifact 校验器（schema + 必填字段 + hash 一致性）
- [x] 定义 RM 发布记录（deploy manifest）：谁在何时发布了哪个 artifact

#### 交付物
- `artifacts/rm_artifacts/*.json`
- `artifacts/rm_deployments/*.json`
- `run_records/*` 与 RM artifact 的关联字段
- 文档化的 artifact 契约

#### Go/No-Go
- [ ] 给定同一 artifact，RM 打分结果可重复
- [x] 每次 RM 部署都可追溯到搜索会话和数据集版本

---

### 阶段 C：RM Server MVP ✅ 已完成

目标：提供稳定的 reward 打分服务，供 RL 训练调用。

#### 关键任务
- [x] 实现 RM server 进程（本地部署优先）
- [x] 最小 API：
  - `GET /healthz`
  - `POST /score`（单样本）
  - `POST /batch_score`（批量）
- [x] 运行时加载指定 `RMArtifact`
- [x] 记录请求日志（request_id、artifact_id、latency、异常）
- [x] 提供重启切换 artifact 的安全机制
- [x] server 内部 LLM 按 criterion 闭环打分（不接受外部传分）
- [x] 复用搜索评分内核，保持在线/离线评分同构

#### Go/No-Go
- [x] 正确性：与离线评分实现一致（单测验证同构）
- [ ] 稳定性：训练负载下无明显崩溃/泄漏
- [x] 可观测：请求日志可导出（stdout + JSONL）

---

### 阶段 D：RL 训练接入与实验编排 🚧 设计完成，待实现

目标：让 RL 训练可直接消费 RM server，并将训练结果与 RM/Search 版本绑定。

#### 关键任务
- [ ] 定义 `TrainingManifest`（训练前声明）：
  - `training_run_id`, `rm_artifact_id`, `rm_deploy_id`, `search_session_id`
  - `rm_endpoint`, 数据版本、trainer 代码版本、执行上下文
- [ ] 定义 `TrainingResultManifest`（训练后事实）：
  - 成功/失败/取消终态
  - 训练摘要、reward summary、输出路径、失败信息
- [ ] 定义 `EvalReport`（标准化评测报告）：
  - benchmark 元信息、扁平 metrics、baseline 对比
- [ ] 建立 append-only training registry：
  - `artifacts/training_runs/manifests/`
  - `artifacts/training_runs/results/`
  - `artifacts/training_runs/evals/`
- [ ] 提供 lineage 查询能力：
  - `training_run -> rm_deploy -> rm_artifact -> search_session`
- [ ] 文档化外部 RL repo 参考交互时序、目录约定与回填流程

#### Go/No-Go
- [ ] 任一训练 run 可回放关键上下文与 reward 来源链
- [ ] 失败 run 可被记录、查询并诊断
- [ ] 可横向比较不同 RM artifact 对训练结果与评测结果的影响

---

### 阶段 E：基于 RL 采样自动训练 Classifier RM 📋 规划中

目标：复用 RL 训练中的采样数据，自动构建去噪后的打分/偏好数据，并交由外部项目训练 classifier RM。

#### 关键任务
- [ ] 数据模型统一：
  - RL 采样批次元信息（来源 run、checkpoint、采样策略）
  - 原始打分数据（`query`、`response`、`score`）
  - 偏好数据（`query`、`chosen_response`、`rejected_response`、`margin`）
- [ ] 多次重复打分降噪：
  - 对同一 `(query, response)` 做多次评分
  - 记录 criterion 级与总分级均值/方差
  - 基于方差与 margin 做 pair 过滤
- [ ] 建立 append-only classifier RM 数据 registry：
  - `artifacts/classifier_rm/sample_batches/`
  - `artifacts/classifier_rm/score_datasets/`
  - `artifacts/classifier_rm/preference_datasets/`
  - `artifacts/classifier_rm/training_runs/`
- [ ] 定义 classifier RM 训练契约：
  - `ClassifierRMTrainingManifest`
  - `ClassifierRMTrainingResult`
  - `ClassifierRMArtifact`
- [ ] 文档化外部 classifier RM repo 参考交互时序与回填流程

#### Go/No-Go
- [ ] 任一 classifier RM artifact 可回查到 teacher RM artifact / deploy
- [ ] 任一 preference pair 可回查到原始采样与重复打分记录
- [ ] 不同 RL run 的样本可被统一构造成同口径训练集

---

### 阶段 F：训练、Classifier RM 与评测监控 📋 规划中

目标：建立面向运营的训练/评测监控，支持告警与回归定位。

#### 关键任务
- [ ] 指标模型统一：
  - RL 训练指标（reward trend、stability、KL/entropy 等）
  - classifier RM 训练指标（pair accuracy、val loss、calibration 等）
  - 评测指标（win-rate、任务成功率、拒答率等）
  - RM 服务指标（QPS、p95、error rate）
- [ ] 时间序列存储与看板
- [ ] 告警规则（退化、异常波动、服务不可用）
- [ ] 自动生成阶段性报告（按 run 或按天）

---

### 阶段 G：闭环调度 📋 规划中

目标：让系统在评测退化时自动触发新一轮 rubric search，并进入灰度发布。

#### 关键任务
- [ ] 定义触发规则（何时 rerun search）
- [ ] 定义发布策略（shadow / canary / full rollout）
- [ ] 定义回滚策略（评测跌破阈值自动回滚）

---

## 优先级矩阵

| 任务 | 用户价值 | 技术难度 | 优先级 | 状态 |
|------|----------|----------|--------|------|
| Harness 收尾修缮（RNG/interval/resume 契约） | 高 | 中 | P0 | ✅ 已完成 |
| RMArtifact 契约与导出 | 高 | 中 | P0 | ✅ 已完成 |
| RM Server MVP | 高 | 中高 | P0 | ✅ 已完成 |
| RL 训练接入与实验编排 | 高 | 中高 | P0 | 🚧 设计完成，待实现 |
| Classifier RM 自动蒸馏 | 高 | 中高 | P0 | 📋 规划中 |
| 训练/评测监控与告警 | 中高 | 中 | P1 | 📋 规划中 |
| 闭环自动调度与灰度发布 | 中高 | 高 | P2 | 📋 规划中 |
| Benchmark/分布式扩展 | 中 | 高 | P3 | ⏸️ 延后 |

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| RM 与离线评分不一致 | 高 | 建立一致性测试集，发布前强制比对 |
| 训练结果不可追溯 | 高 | 强制 run manifest 记录 artifact/data/code 版本 |
| Reward hacking / 指标虚高 | 高 | 引入独立 holdout 评测与多指标约束 |
| RM 服务抖动影响训练稳定性 | 高 | 设定超时/重试/降级策略与容量基线 |
| 过早追求分布式导致复杂度失控 | 中高 | 单机闭环先跑通，再评估扩展 |

---

## API 演进策略

保持现有入口：

```bash
uv run python -m autosr.cli --dataset ... --output ...
```

新能力通过新增命令或可选参数引入：

```bash
# 1) 搜索并得到 best rubrics（已有）
uv run python -m autosr.cli --dataset ... --mode evolutionary --output ...

# 2) 导出可部署 RM artifact（已实现）
uv run python -m autosr.rm.export --search-output ... --out-artifact ...

# 3) 记录 RM 部署 manifest（已实现）
uv run python -m autosr.rm.deploy --artifact ... --deployment-target prod

# 4) 启动 RM server（已实现）
uv run python -m autosr.rm.server --artifact ... --host 0.0.0.0 --port 8080 --request-log-path artifacts/rm_server_logs/requests.jsonl

# 5) RL 训练消费 RM endpoint（规划）
uv run python -m autosr.rl.train --rm-endpoint http://127.0.0.1:8080 --run-manifest ...

# 6) 登记 RL 采样批次（规划）
uv run python -m autosr.classifier_rm.record_sample_batch --manifest ...

# 7) 构建重复打分数据集（规划）
uv run python -m autosr.classifier_rm.build_score_dataset --sample-batch sample_batch_001 --repeat-count 5

# 8) 构建偏好数据集（规划）
uv run python -m autosr.classifier_rm.build_preference_dataset --score-dataset score_dataset_001 --pairing-policy hybrid

# 9) 准备 classifier RM 训练 manifest（规划）
uv run python -m autosr.classifier_rm.prepare_training --preference-dataset preference_dataset_001 --trainer-project external-classifier-rm
```

---

## 变更日志

### 2026-04-16
- 阶段 B 收尾完成：新增 `DeployManifest` schema、`record_deploy_manifest` 用例、CLI 命令 `autosr.rm.deploy`。
- deploy manifest 默认写入 `artifacts/rm_deployments/*.json`（一部署一文件），支持按 `deployment_target` 自动推断 `previous_artifact_id`。
- 发布记录补齐 `artifact_id/source_session_id/dataset_hash/config_hash` 链路，满足部署追溯要求。

### 2026-04-17
- 明确 Stage D 方向采用 “Contract + Registry + Reference Flow”：
  `autosr` 负责训练契约、append-only registry 与 lineage 查询；外部 RL repo 负责 trainer 执行与结果回填。
- 新增 Stage D 详细设计文档 `docs/design-docs/02-stage-d-rl-lineage.md`，补齐 TrainingManifest / TrainingResultManifest / EvalReport、参考交互时序、目录约定与失败恢复策略。
- 新增 Stage E 方向：基于 RL 采样自动蒸馏 classifier RM，位于 Stage D 之后、监控之前。
- 新增 Stage E 详细设计文档 `docs/design-docs/03-stage-e-classifier-rm.md`，补齐 sample batch / repeated score dataset / preference dataset / classifier training handshake。
- 原路线图中的“训练与评测监控 / 闭环调度”顺延为 Stage F / Stage G，以保持阶段语义清晰。

### 2026-04-04
- 阶段 A 收尾完成：RNG state 恢复修复、`checkpoint_interval_seconds` 生效、resume 语义落地、scheduler state 可恢复。
- 阶段 B 核心能力落地：新增 `autosr.rm` 子包，包含 `RMArtifact` schema v1、导出命令 `autosr.rm.export`、artifact 校验器。
- `run_manifest` 新增 harness 会话信息回写（用于 artifact 的 `source_session_id` 追溯）。

### 2026-04-03
- 路线图主线从"通用 Harness 扩展"重定向为"RM+RL 闭环"。
- 明确阶段 B/C/D/E/F 对应：artifact、服务、训练、监控、闭环调度。
- 保留阶段 A（当前 harness）作为底座，不再作为最终目标本身。

---

*本路线图为活文档，随项目进展持续更新。*

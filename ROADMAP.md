# AutoSR Roadmap: 从 Rubric Search 到 RM+RL 闭环（重定向版）

将 `autosr` 从“单次运行的 rubric 搜索器”演进为“可用于 RL 训练与评测的 Reward Harness”。

## 当前状态（截至 2026-04-03）

### 已经完成且建议保留
- ✅ `SearchSession` 生命周期管理（创建、执行、恢复）
- ✅ `SearchCheckpoint` v1 schema（含 config/dataset hash）
- ✅ `StateManager` 原子持久化
- ✅ CLI 的 `--resume-from` / `--checkpoint-every-generation`
- ✅ evolutionary + `global_batch` scope 的 step-wise 执行

### 这批改动的定位
- 它们是“**工程底座**”，不是“RM/RL 主业务闭环”。
- 结论：**保留并继续演进**，但后续路线必须切向 RM server、RL 驱动、训练/评测监控。

---

## 北极星目标（最终形态）

形成可重复的闭环：

1. 自动搜索与迭代 rubric（已有能力 + 持续增强）
2. 将当前最优 rubric 产出为可部署的 RM artifact（版本化）
3. 自动部署/更新 RM server（稳定评分 API）
4. RL 训练任务消费 RM server 进行 reward 计算
5. 监控 RL 训练与评测，触发告警与回归分析
6. 根据评测结果决定是否再次触发 rubric search

---

## 设计原则（重定向后的硬约束）

1. **闭环优先**：任何新增能力必须服务 `Search -> RM -> RL -> Eval` 链路。
2. **契约先行**：先定义 artifact/API/指标契约，再做实现。
3. **可回滚**：每次 RM 发布、每次训练实验都可追溯到明确版本。
4. **兼容现有 CLI**：`uv run python -m autosr.cli` 继续可用。
5. **先单机后分布式**：未证明价值前不提前引入复杂基础设施。

---

## 阶段划分（面向目标重排）

## 阶段 A：Harness 底座稳定化（已完成，持续修缮）

目标：让搜索阶段可长时运行、可恢复、可复盘，作为上层 RM/RL 的可信输入。

### 必做收尾
- [ ] 修复 RNG 状态恢复逻辑（避免“恢复后轨迹漂移”）
- [ ] 实现 `checkpoint_interval_seconds` 的真实生效逻辑
- [ ] 明确 resume 行为契约（什么场景是真正续跑，什么场景退化为重跑）
- [ ] 完善 scheduler 可恢复状态（不仅 diagnostics）

### Go/No-Go
- [ ] 同一 checkpoint 多次恢复结果偏差可解释且在阈值内
- [ ] 长时运行（>2h）无状态损坏

---

## 阶段 B：RM Artifact 契约与部署（P0）

目标：把“best rubric JSON”升级为可部署、可追溯的 RM artifact。

### 关键任务
- [ ] 定义 `RMArtifact` schema（建议 v1）
  - `artifact_id`, `created_at`, `source_session_id`, `dataset_hash`, `config_hash`
  - `rubric`, `scoring_policy`, `normalization`, `compatibility`
- [ ] 新增 artifact 导出能力（由 search 输出生成 RM artifact）
- [ ] 增加 artifact 校验器（schema + 必填字段 + hash 一致性）
- [ ] 定义 RM 发布记录（deploy manifest）：谁在何时发布了哪个 artifact

### 交付物
- `artifacts/rm_artifacts/*.json`
- `run_records/*` 与 RM artifact 的关联字段
- 文档化的 artifact 契约

### Go/No-Go
- [ ] 给定同一 artifact，RM 打分结果可重复
- [ ] 每次 RM 部署都可追溯到搜索会话和数据集版本

---

## 阶段 C：RM Server MVP（P0）

目标：提供稳定的 reward 打分服务，供 RL 训练调用。

### 关键任务
- [ ] 实现 RM server 进程（本地部署优先）
- [ ] 最小 API：
  - `GET /healthz`
  - `POST /score`（单样本）
  - `POST /batch_score`（批量）
- [ ] 运行时加载指定 `RMArtifact`
- [ ] 记录请求日志（request_id、artifact_id、latency、异常）
- [ ] 提供热切换或重启切换 artifact 的安全机制

### Go/No-Go
- [ ] 正确性：与离线评分实现一致
- [ ] 稳定性：训练负载下无明显崩溃/泄漏
- [ ] 可观测：请求成功率、延迟分布可导出

---

## 阶段 D：RL 训练接入与实验编排（P0）

目标：让 RL 训练可直接消费 RM server，并将训练结果与 RM/Search 版本绑定。

### 关键任务
- [ ] 训练入口支持注入 RM endpoint + artifact_id
- [ ] 训练实验 manifest 记录：
  - `training_run_id`, `rm_artifact_id`, `search_session_id`
  - 训练超参数、数据版本、代码版本
- [ ] 训练失败恢复策略（重试、跳过、安全终止）
- [ ] 输出标准化评测报告（train/eval 指标）

### Go/No-Go
- [ ] 任一训练 run 可回放关键上下文
- [ ] 可横向比较不同 RM artifact 对训练结果的影响

---

## 阶段 E：训练与评测监控（P1）

目标：建立面向运营的训练/评测监控，支持告警与回归定位。

### 关键任务
- [ ] 指标模型统一：
  - 训练指标（reward trend、stability、KL/entropy 等）
  - 评测指标（win-rate、任务成功率、拒答率等）
  - RM 服务指标（QPS、p95、error rate）
- [ ] 时间序列存储与看板
- [ ] 告警规则（退化、异常波动、服务不可用）
- [ ] 自动生成阶段性报告（按 run 或按天）

### Go/No-Go
- [ ] 指标异常可在分钟级发现
- [ ] 回归定位可定位到 artifact / 训练版本 / 数据版本

---

## 阶段 F：闭环调度（P2）

目标：让系统在评测退化时自动触发新一轮 rubric search，并进入灰度发布。

### 关键任务
- [ ] 定义触发规则（何时 rerun search）
- [ ] 定义发布策略（shadow / canary / full rollout）
- [ ] 定义回滚策略（评测跌破阈值自动回滚）

---

## 优先级矩阵（重排后）

| 任务 | 用户价值 | 技术难度 | 优先级 | 状态 |
|------|----------|----------|--------|------|
| Harness 收尾修缮（RNG/interval/resume 契约） | 高 | 中 | P0 | 🚧 进行中 |
| RMArtifact 契约与导出 | 高 | 中 | P0 | 📋 待开始 |
| RM Server MVP | 高 | 中高 | P0 | 📋 待开始 |
| RL 训练接入与实验编排 | 高 | 中高 | P0 | 📋 待开始 |
| 训练/评测监控与告警 | 中高 | 中 | P1 | 📋 待开始 |
| 闭环自动调度与灰度发布 | 中高 | 高 | P2 | 📋 规划中 |
| Benchmark/分布式扩展 | 中 | 高 | P3 | ⏸ 延后 |

---

## 风险与缓解（面向 RM+RL）

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| RM 与离线评分不一致 | 高 | 建立一致性测试集，发布前强制比对 |
| 训练结果不可追溯 | 高 | 强制 run manifest 记录 artifact/data/code 版本 |
| Reward hacking / 指标虚高 | 高 | 引入独立 holdout 评测与多指标约束 |
| RM 服务抖动影响训练稳定性 | 高 | 设定超时/重试/降级策略与容量基线 |
| 过早追求分布式导致复杂度失控 | 中高 | 单机闭环先跑通，再评估扩展 |

---

## API 演进策略（兼容优先）

- 保持现有入口：`uv run python -m autosr.cli --dataset ... --output ...`
- 新能力通过新增命令或可选参数引入，不破坏既有搜索流程。

### 目标 API 形态（规划）

```bash
# 1) 搜索并得到 best rubrics（已有）
uv run python -m autosr.cli --dataset ... --mode evolutionary --output ...

# 2) 导出可部署 RM artifact（规划）
uv run python -m autosr.rm.export --search-output ... --out-artifact ...

# 3) 启动 RM server（规划）
uv run python -m autosr.rm.server --artifact ... --host 0.0.0.0 --port 8080

# 4) RL 训练消费 RM endpoint（规划）
uv run python -m autosr.rl.train --rm-endpoint http://127.0.0.1:8080 --run-manifest ...
```

---

## 建议的近期执行顺序（4-8 周）

1. 完成阶段 A 收尾（先修复可恢复一致性问题）
2. 落地阶段 B（RM artifact 契约 + 导出 + 校验）
3. 落地阶段 C（RM server MVP + 基础监控）
4. 落地阶段 D（RL 训练接入 + 实验 manifest）
5. 在真实训练中验证阶段 E 指标体系

---

## 变更日志

### 2026-04-03
- 路线图主线从“通用 Harness 扩展”重定向为“RM+RL 闭环”。
- 明确阶段 B/C/D/E/F 对应：artifact、服务、训练、监控、闭环调度。
- 保留阶段 A（当前 harness）作为底座，不再作为最终目标本身。

### 2025-04-02
- 阶段 0/1 首次落地：SearchSession + Checkpoint/Resume。

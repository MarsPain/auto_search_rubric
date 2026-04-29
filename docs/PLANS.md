# AutoSR 执行计划管理

> **版本**: v1.4 | **最后更新**: 2026-04-29
> 
> 计划作为一等工件，活跃/已完成/技术债统一版本化管理。

---

## 计划状态说明

| 状态 | 目录 | 含义 |
|------|------|------|
| 🟡 活跃 | `exec-plans/active/` | 当前正在执行的计划 |
| ✅ 已完成 | `exec-plans/completed/` | 已交付并验收的计划 |
| 🔴 技术债 | `exec-plans/tech-debt/` | 已知但未解决的技术债务 |

---

## 活跃计划 (Active)

| 计划 | 描述 | 创建日期 | 文档 |
|------|------|----------|------|
| Rename to Reward Harness | 将项目主命名从 `auto_search_rubric` / `autosr` 平滑迁移到 `Reward Harness` / `reward_harness`，保留旧入口兼容 | 2026-04-29 | [rename-to-reward-harness.md](exec-plans/active/rename-to-reward-harness.md) |

### 计划模板

创建新计划时请参考以下模板：

```markdown
# 计划标题

> **状态**: 活跃 | **优先级**: P0/P1/P2 | **负责人**: TBD | **创建日期**: YYYY-MM-DD

## 目标

一句话描述计划目标。

## 任务清单

- [ ] 任务1
- [ ] 任务2
- [ ] 任务3

## 验收标准

- [ ] 验收项1
- [ ] 验收项2

## 关联文档

- [设计文档](../design-docs/xxx.md)
- [API契约](../DESIGN.md)

## 变更日志

- YYYY-MM-DD: 创建计划
```

---

## 已完成计划 (Completed)

| 计划 | 描述 | 完成日期 | 文档 |
|------|------|----------|------|
| Stage 0 - Harness底座 | 会话化、Checkpoint schema、Resume验证 | 2026-04-03 | [stage0-harness.md](exec-plans/completed/stage0-harness.md) |
| Stage 1 - 可恢复执行 | StateManager、单步执行、Resume能力 | 2026-04-04 | [stage1-resume.md](exec-plans/completed/stage1-resume.md) |
| RMArtifact 阶段B核心 | RMArtifact schema、导出、校验器、deploy manifest | 2026-04-16 | [stage2-rm-artifact.md](exec-plans/completed/stage2-rm-artifact.md) |
| Stage D3 - Comparative Experiment View | 横向比较、baseline对比、回归检测 | 2026-04-22 | [stage-d3-comparative-view.md](exec-plans/completed/stage-d3-comparative-view.md) |
| Stage E Readiness - Tech Debt Sprint | 清偿进入 Stage E 前的阻塞债务 | 2026-04-25 | [stage-e-readiness-tech-debt-sprint.md](exec-plans/completed/stage-e-readiness-tech-debt-sprint.md) |
| Tech Debt Follow-up - Local Quality Gates | 继续清理非阻塞技术债，并收敛本地质量门禁；中短期不引入 GitHub CI/CD | 2026-04-26 | [tech-debt-followup-local-quality-gates.md](exec-plans/completed/tech-debt-followup-local-quality-gates.md) |

---

## 技术债 (Tech Debt)

| 债务 | 描述 | 影响 | 计划解决 |
|------|------|------|----------|
| prompt_local scope checkpoint | prompt_local作用域已支持step-wise checkpoint/resume | 低 | ✅ 已完成：`evolutionary + prompt_local` 可按 prompt/generation 级别恢复 |
| Iterative模式step执行 | Iterative模式暂不支持step-wise执行 | 低 | 按需 |
| 本地质量门禁收敛 | 中短期不引入 GitHub CI/CD；本地统一单元测试、文档校验、ruff、mypy 的执行入口与验收标准 | 中 | ✅ 已完成：[Tech Debt Follow-up](exec-plans/completed/tech-debt-followup-local-quality-gates.md) |
| **全面技术债务审计** | 详见 [tech-debt-audit-2026-04.md](exec-plans/tech-debt/tech-debt-audit-2026-04.md)，Stage E 阻塞项与本地质量门禁后续计划已完成，剩余低优先级项按需清偿 | **中** | 分阶段清偿 |

---

## 路线图阶段

### 阶段 A: Harness底座稳定化 ✅ 已完成

目标：让搜索阶段可长时运行、可恢复、可复盘。

- [x] RNG状态恢复逻辑修复
- [x] `checkpoint_interval_seconds` 生效
- [x] resume行为契约明确
- [x] scheduler可恢复状态

### 阶段 B: RM Artifact契约与部署 ✅ 已完成

目标：把"best rubric JSON"升级为可部署、可追溯的RM artifact。

- [x] `RMArtifact` schema v1
- [x] artifact导出能力
- [x] artifact校验器
- [x] deploy manifest（发布记录）

### 阶段 C: RM Server MVP ✅ 已完成

目标：提供稳定的reward打分服务。

- [x] RM server进程（FastAPI + Uvicorn）
- [x] API: `/healthz`, `/score`, `/batch_score`
- [x] 运行时加载artifact（缺失 `runtime_snapshot` 启动失败）
- [x] 请求日志（stdout + JSONL）
- [x] 闭环LLM评分（server内部按criteria调用LLM，不接受外部传分）
- [x] 评分同构（复用 `RubricEvaluator` 单候选评分内核）

### 阶段 D: RL训练接入与实验编排 ✅ 已完成

目标：让RL训练可直接消费RM server。

- [x] 训练入口支持注入RM endpoint
- [x] 训练实验manifest记录（TrainingManifest / TrainingResultManifest）
- [x] 失败恢复策略（registry + lineage）
- [x] 标准化评测报告（EvalReport）
- [x] 比较视图与回归检测（D3）

### 阶段 E: 基于RL采样自动训练Classifier RM 📋 规划中

目标：基于RL训练采样自动构建去噪打分数据与偏好数据，并交由外部trainer训练classifier RM。

- [ ] RL采样批次导入契约
- [ ] 多次重复打分降噪
- [ ] 原始打分数据集 registry
- [ ] 偏好数据集 registry
- [ ] classifier RM 训练 manifest / result / artifact

### 阶段 F: 训练、Classifier RM与评测监控 📋 规划中

### 阶段 G: 闭环调度 📋 规划中

---

## 优先级矩阵

| 任务 | 用户价值 | 技术难度 | 优先级 | 状态 |
|------|----------|----------|--------|------|
| Harness收尾修缮 | 高 | 中 | P0 | ✅ 已完成 |
| RMArtifact契约 | 高 | 中 | P0 | ✅ 已完成 |
| RM Server MVP | 高 | 中高 | P0 | ✅ 已完成 |
| RL训练接入 | 高 | 中高 | P0 | ✅ 已完成 |
| Classifier RM自动蒸馏 | 高 | 中高 | P0 | 📋 规划中 |
| 监控与告警 | 中高 | 中 | P1 | 📋 规划中 |
| 闭环自动调度 | 中高 | 高 | P2 | 📋 规划中 |

---

## 文档规范

### 创建新计划

1. 在 `exec-plans/active/` 创建新的markdown文件
2. 使用统一模板，填写元数据（状态、优先级、负责人、日期）
3. 关联相关设计文档
4. 完成后移动到 `exec-plans/completed/`

### 更新路线图

修改本文档的"路线图阶段"章节，同步最新状态。

### 记录技术债

在"技术债"表格中新增条目，说明影响与计划解决时间。

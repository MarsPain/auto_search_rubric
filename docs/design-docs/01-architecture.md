# AutoSR 架构演进图（面向 RM Server + RL 训练闭环）

> **版本**: v1.0 | **状态**: 稳定 | **最后更新**: 2026-04-16
> 
> 本文档与 `docs/ROADMAP.md` 保持一致，目标是把当前 Harness 底座延展为：
> `Rubric Search -> RM Artifact -> RM Server -> RL Training -> Eval & Monitoring -> Search Refresh`

---

## 0. 当前落地（截至 2026-04-04）

- ✅ Harness 阶段 A 收尾完成：
  - RNG state 恢复修复
  - `checkpoint_interval_seconds` 真实生效
  - resume 语义契约（`continue_from_checkpoint` / `reseed_from_checkpoint`）
  - scheduler state 可恢复（不再仅 diagnostics）
- ✅ 阶段 B 核心能力已落地：
  - `autosr.rm.data_models.RMArtifact`（schema v1）
  - `autosr.rm.use_cases`（build/export/validate）
  - `autosr.rm.export` CLI 导出入口
  - hash 一致性校验（dataset/config）与 rubric 指纹一致性校验
- ✅ 阶段 B deploy manifest 已完成：
  - `DeployManifest` schema（含发布追溯字段）
  - `record_deploy_manifest` 用例（自动推断 `previous_artifact_id`）
  - `autosr.rm.deploy` CLI（一部署一文件）
- ✅ 阶段 C RM Server MVP 已完成：
  - `autosr.rm.server`（FastAPI + Uvicorn）
  - API：`/healthz`、`/score`、`/batch_score`
  - server 内部 LLM 闭环评分（按 criteria 打分，不接受外部传分）
  - 评分同构（复用 `RubricEvaluator` 单候选评分内核）
  - 请求日志（stdout + JSONL）

---

## 1. 当前架构（保留）

```
┌──────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│                      (autosr/cli.py)                         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     ComponentFactory                          │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ SearchSession + Searcher (iterative/evolutionary)            │
│ + Checkpoint/Resume (SearchCheckpoint, StateManager)         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Search Output JSON + run_records + checkpoints               │
└──────────────────────────────────────────────────────────────┘
```

定位：这一层已经可以作为可靠"搜索执行底座"，不建议回退。

---

## 2. 目标架构（新增主链路）

```
┌──────────────────────────────────────────────────────────────┐
│                    Search Harness Layer                      │
│   (SearchSession / checkpoint / resume / diagnostics)        │
└──────────────────────────────┬───────────────────────────────┘
                               │ best_rubric + provenance
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 RM Artifact Builder & Registry               │
│      (schema validation, versioning, manifest binding)       │
└──────────────────────────────┬───────────────────────────────┘
                               │ artifact_id
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         RM Server                            │
│     /healthz  /score  /batch_score  + request logs          │
└──────────────────────────────┬───────────────────────────────┘
                               │ reward API
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   RL Training Orchestrator                   │
│   (trainer integration, run manifest, retry/recovery)        │
└──────────────────────────────┬───────────────────────────────┘
                               │ train/eval metrics
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Monitoring & Evaluation Plane               │
│ dashboards, alerts, regression checks, experiment compare    │
└──────────────────────────────┬───────────────────────────────┘
                               │ trigger rules
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                 Closed-Loop Search Controller                │
│  (decide refresh, canary rollout, rollback policy)           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 数据与契约分层

### 3.1 Search Checkpoint（已有）

用途：恢复搜索执行状态，不直接作为线上 RM 发布物。

### 3.2 RM Artifact（新增核心契约）

用途：RM server 的唯一加载输入，必须可追溯。

建议最小字段（v1）：
- `artifact_id`
- `created_at_utc`
- `source_session_id`
- `dataset_hash`
- `config_hash`
- `rubric`
- `scoring_policy`
- `normalization`
- `compatibility`

### 3.3 Training Run Manifest（新增）

用途：训练可追溯与可比较。

建议最小字段：
- `training_run_id`
- `rm_artifact_id`
- `search_session_id`
- `dataset_version`
- `trainer_config`
- `code_version`

---

## 4. 阶段化架构落地

### 阶段 A（当前 + 修缮）

目标：稳固 Harness 底座。

当前状态：✅ 已完成（进入维护期）

### 阶段 B（RM Artifact）

在现有搜索输出后新增"构建和校验层"：

```
Search Output -> RM Artifact Builder -> Artifact Store
```

当前状态：✅ 已完成（schema/导出/校验/deploy manifest 已落地）

### 阶段 C（RM Server）

在 artifact 基础上提供统一 reward 服务：

```
RM Artifact -> RM Server -> /score|/batch_score
```

当前状态：✅ 已完成（MVP）

### 阶段 D（RL 训练接入）

训练端通过 endpoint 获取 reward，训练 run 自动绑定 artifact。

### 阶段 E（监控评测）

引入指标看板、告警、回归分析，支持对比不同 artifact 下的训练表现。

### 阶段 F（闭环控制）

根据评测退化自动触发 search refresh，并支持 canary 与 rollback。

---

## 5. API 演进（规划）

### 5.1 当前 API（保留）

```bash
uv run python -m autosr.cli --dataset ... --mode evolutionary --output ...
```

### 5.2 下一步 API（建议）

```bash
# 导出 RM artifact（已实现）
uv run python -m autosr.rm.export --search-output artifacts/best_rubrics.json --out-artifact artifacts/rm_artifacts/rm_v1.json

# 启动 RM server
uv run python -m autosr.rm.server --artifact artifacts/rm_artifacts/rm_v1.json --host 0.0.0.0 --port 8080

# RL 训练接入 RM endpoint
uv run python -m autosr.rl.train --rm-endpoint http://127.0.0.1:8080 --manifest artifacts/training_runs/run_001.json
```

说明：`autosr.rm.export` 与 `autosr.rm.server` 已实现，`autosr.rl.train` 仍为规划项。

---

## 6. 部署架构演进

### 6.1 近期：单机双进程（推荐起步）

```
┌──────────────────────────────────────────────────────────────┐
│                      Single Host                             │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │ RM Server Process   │◀──│ RL Trainer Process           │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
│              ▲                          │                     │
│              │                          ▼                     │
│        RM Artifact Store       Train/Eval Metrics Store      │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 中期：服务拆分 + 监控平面

```
Search Harness -> Artifact Registry -> RM Service Cluster
                                      │
                                      ▼
                               RL Training Jobs
                                      │
                                      ▼
                           Metrics + Alerts + Reports
```

### 6.3 远期：闭环自动化控制

```
Monitoring Regression Trigger -> Search Refresh -> Canary Deploy -> Promote/Rollback
```

---

## 7. 关键质量属性与验收

| 质量属性 | 验收问题 |
|---------|----------|
| 可追溯性 | 能否从任一训练 run 反查 RM artifact 与搜索会话？ |
| 一致性 | RM online 评分与 offline 评分是否一致？ |
| 稳定性 | RM 在训练负载下是否稳定且可恢复？ |
| 可观测性 | 训练、评测、服务指标是否可实时查看并告警？ |
| 可回滚性 | 评测退化时是否能快速回滚到上一个 artifact？ |

---

## 8. 与 ROADMAP 对齐检查

- 保留阶段 A（现有 Harness）作为底座，而非最终目标。
- 后续阶段主线统一为 `Artifact -> RM Server -> RL -> Monitoring -> Closed Loop`。
- 分布式与 benchmark 扩展明确降级为后置项，不抢占主线资源。

---

## 9. 更新记录

### 2026-04-04
- 同步阶段 A 实际完成状态（RNG/interval/resume/scheduler）。
- 同步阶段 B 已交付项（RMArtifact schema、导出、校验）与待办项（deploy manifest）。

### 2026-04-16
- 阶段 B deploy manifest 收尾完成，补齐发布记录契约与 CLI。
- 更新阶段状态：阶段 B 从“进行中”转为“已完成”。

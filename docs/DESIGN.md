# AutoSR 架构设计

> **版本**: v1.0 | **状态**: 稳定 | **最后更新**: 2026-04-04

---

## 设计目标

将 `autosr` 从"单次运行的 rubric 搜索器"演进为"可用于 RL 训练与评测的 Reward Harness"。

```
Rubric Search -> RM Artifact -> RM Server -> RL Training -> Eval & Monitoring -> Search Refresh
```

---

## 架构分层

### 当前架构（Harness底座 - 已完成）

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ComponentFactory                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ SearchSession + Searcher (iterative/evolutionary)           │
│ + Checkpoint/Resume (SearchCheckpoint, StateManager)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│        Search Output JSON + run_records + checkpoints       │
└─────────────────────────────────────────────────────────────┘
```

### 目标架构（RM+RL闭环）

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Harness Layer                     │
│      (SearchSession / checkpoint / resume)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │ best_rubric + provenance
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               RM Artifact Builder & Registry                │
│     (schema validation, versioning, manifest binding)       │
└─────────────────────────────┬───────────────────────────────┘
                              │ artifact_id
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        RM Server                            │
│      /healthz  /score  /batch_score  + request logs         │
└─────────────────────────────┬───────────────────────────────┘
                              │ reward API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  RL Training Orchestrator                   │
└─────────────────────────────┬───────────────────────────────┘
                              │ train/eval metrics
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Monitoring & Evaluation Plane                 │
└─────────────────────────────┬───────────────────────────────┘
                              │ trigger rules
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Closed-Loop Search Controller                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心契约

### Search Checkpoint（恢复用）

用途：恢复搜索执行状态，不直接作为线上RM发布物。

```python
SearchCheckpoint(
    session_id: str
    generation: int
    best_rubrics: dict
    best_scores: dict
    history: dict
    scheduler_state: dict
    rng_state: dict
    config_hash: str
    dataset_hash: str
    schema_version: str = "1.0"
)
```

### RM Artifact（部署用）

用途：RM server的唯一加载输入，必须可追溯。

```python
RMArtifact(
    artifact_id: str
    created_at_utc: str
    source_session_id: str
    dataset_hash: str
    config_hash: str
    rubric: dict
    scoring_policy: dict
    normalization: dict
    compatibility: dict
)
```

### Training Run Manifest（追溯用）

用途：训练可追溯与可比较。

```python
TrainingRunManifest(
    training_run_id: str
    rm_artifact_id: str
    search_session_id: str
    dataset_version: str
    trainer_config: dict
    code_version: str
)
```

---

## 模块职责

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `autosr/harness/session.py` | 搜索会话生命周期 | `SearchSession` |
| `autosr/harness/state.py` | Checkpoint schema | `SearchCheckpoint`, `ResumeValidator` |
| `autosr/harness/storage.py` | 状态持久化 | `StateManager` |
| `autosr/search/` | 搜索算法实现 | `IterativeSearcher`, `EvolutionarySearcher` |
| `autosr/llm_components/` | LLM交互组件 | `LLMInitializer`, `LLMProposer`, `LLMVerifier` |
| `autosr/rm/` | RM Artifact管理 | `RMArtifact`, `ArtifactExporter` |
| `autosr/config.py` | 运行时配置 | `RuntimeConfig` |
| `autosr/data_models.py` | 领域实体 | `Rubric`, `Criterion` |
| `autosr/types.py` | 统一枚举 | `BackendType`, `SearchMode` |

---

## 接口契约

### 当前API（稳定）

```bash
# 搜索
uv run python -m autosr.cli --dataset ... --mode evolutionary --output ...

# 导出RM artifact
uv run python -m autosr.rm.export --search-output ... --out-artifact ...
```

### 规划API

```bash
# 启动RM server
uv run python -m autosr.rm.server --artifact ... --host 0.0.0.0 --port 8080

# RL训练
uv run python -m autosr.rl.train --rm-endpoint http://127.0.0.1:8080 --run-manifest ...
```

---

## 质量属性验收

| 质量属性 | 验收问题 |
|---------|----------|
| 可追溯性 | 能否从任一训练run反查RM artifact与搜索会话？ |
| 一致性 | RM online评分与offline评分是否一致？ |
| 稳定性 | RM在训练负载下是否稳定且可恢复？ |
| 可观测性 | 训练、评测、服务指标是否可实时查看并告警？ |
| 可回滚性 | 评测退化时是否能快速回滚到上一个artifact？ |

---

## 相关文档

- [design-docs/01-architecture.md](design-docs/01-architecture.md) - 详细架构演进
- [PLANS.md](PLANS.md) - 执行计划管理
- [PRODUCT_SENSE.md](PRODUCT_SENSE.md) - 产品方向与需求

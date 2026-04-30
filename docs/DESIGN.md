# Reward Harness 架构设计

> **版本**: v1.3 | **状态**: 稳定 | **最后更新**: 2026-04-26

---

## 设计目标

将项目从"单次运行的 rubric 搜索器"演进为"可用于 RL 训练与评测的 Reward Harness"。新代码优先使用 `reward_harness` 包名；`autosr` 在迁移期内保持兼容。

```
Rubric Search -> RM Artifact -> RM Server -> RL Training -> Classifier RM Distillation -> Eval & Monitoring -> Search Refresh
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
│ SearchSession + Searcher / SteppableSearcher protocols       │
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
                              │ sampled (query, response)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Classifier RM Distillation Plane              │
│   (sample registry, denoising, preference building, train)  │
└─────────────────────────────┬───────────────────────────────┘
                              │ distilled artifact + metrics
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

### 兼容入口策略

- 新代码统一从 `reward_harness.data_models` 导入 `Rubric`、`Criterion`、`PromptExample` 等领域实体。
- `autosr.models` 作为长期兼容 re-export shim 保留，用于历史导入路径和下游扩展；不在当前阶段发出导入级 `DeprecationWarning`，避免破坏既有脚本。
- 兼容 shim 必须保持测试覆盖，确保 `reward_harness.data_models.*` 与 `autosr.data_models.*` 指向同一对象。

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
    runtime_snapshot: dict  # seed/extraction/candidate_extraction/llm(verifier)
)
```

### TrainingManifest（训练前声明）

用途：记录训练启动前绑定的 reward、数据、trainer 代码版本与执行上下文。

```python
TrainingManifest(
    training_run_id: str
    rm_artifact_id: str
    rm_deploy_id: str
    search_session_id: str
    rm_endpoint: str
    dataset: dict
    trainer: dict
    trainer_config: dict
    execution: dict
    tags: list[str]
)
```

### TrainingResultManifest（训练后事实）

用途：记录训练终态、产物路径与失败信息；必须覆盖成功与失败两类 run。

```python
TrainingResultManifest(
    training_run_id: str
    status: str  # succeeded | failed | canceled
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    trainer_code_version: str
    output: dict
    reward_summary: dict
    training_summary: dict
    failure: dict | None
)
```

### RLSampleBatchManifest（RL 采样批次）

用途：登记从 RL 训练侧导入的一批 `(query, response)` 采样，作为 classifier RM 蒸馏的输入源。

```python
RLSampleBatchManifest(
    sample_batch_id: str
    training_run_id: str
    rm_artifact_id: str
    rm_deploy_id: str
    source: dict
    dataset: dict
    payload: dict
)
```

### RepeatedScoreDatasetManifest（重复打分数据集）

用途：记录基于固定 teacher RM 对 RL 采样做重复打分后的聚合结果。

```python
RepeatedScoreDatasetManifest(
    score_dataset_id: str
    sample_batch_id: str
    training_run_id: str
    rm_artifact_id: str
    rm_deploy_id: str
    scoring: dict
    payload: dict
    summary: dict
)
```

### PreferenceDatasetManifest（偏好数据集）

用途：记录基于重复打分构造出的 classifier RM 偏好训练集。

```python
PreferenceDatasetManifest(
    preference_dataset_id: str
    score_dataset_id: str
    training_run_id: str
    pairing_policy: dict
    split: dict
    payload: dict
    summary: dict
)
```

### ClassifierRMTrainingManifest（classifier RM 训练前声明）

用途：记录 classifier RM 训练的输入数据、teacher RM 与 trainer 配置。

```python
ClassifierRMTrainingManifest(
    classifier_training_run_id: str
    preference_dataset_id: str
    score_dataset_id: str
    source_training_run_ids: list[str]
    teacher: dict
    trainer: dict
    model: dict
    objective: dict
    hyperparameters: dict
    output: dict
)
```

### ClassifierRMTrainingResult（classifier RM 训练后事实）

用途：记录 classifier RM 训练终态、指标与输出工件路径。

```python
ClassifierRMTrainingResult(
    classifier_training_run_id: str
    status: str  # succeeded | failed | canceled
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    trainer_code_version: str
    metrics: dict
    output: dict
    failure: dict | None
)
```

### EvalReport（评测报告）

用途：记录 benchmark、指标与 baseline 对比，支持同一 training run 的多次评测。

```python
EvalReport(
    eval_run_id: str
    training_run_id: str
    benchmark: dict
    metrics: dict
    summary: dict
    comparison_baseline: dict | None
)
```

---

## 模块职责

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `reward_harness/harness/session.py` | 搜索会话生命周期 | `SearchSession` |
| `reward_harness/harness/state.py` | Checkpoint schema | `SearchCheckpoint`, `ResumeValidator` |
| `reward_harness/harness/storage.py` | 状态持久化 | `StateManager` |
| `reward_harness/interfaces.py` | 跨域协议边界 | `Searcher`, `SteppableSearcher`, `CheckpointCallback` |
| `reward_harness/search/` | 搜索算法实现 | `IterativeSearcher`, `EvolutionarySearcher` |
| `reward_harness/llm_components/` | LLM交互组件 | `LLMInitializer`, `LLMProposer`, `LLMVerifier` |
| `reward_harness/rm/` | RM Artifact管理 + 服务化评分 | `RMArtifact`, `ArtifactExporter`, `RMScoringService` |
| `reward_harness/rl/` | RL接入契约、实验记录、lineage查询与比较视图 | `TrainingManifest`, `TrainingResultManifest`, `EvalReport` |
| `reward_harness/classifier_rm/` | RL 采样蒸馏与 classifier RM 训练契约（规划） | `RLSampleBatchManifest`, `PreferenceDatasetManifest`, `ClassifierRMTrainingManifest` |
| `reward_harness/config.py` | 运行时配置 | `RuntimeConfig` |
| `reward_harness/data_models.py` | 领域实体 | `Rubric`, `Criterion` |
| `reward_harness/types.py` | 统一枚举 | `BackendType`, `SearchMode` |

---

## 接口契约

### 当前API（稳定）

```bash
# 搜索（推荐入口；兼容入口：autosr.cli）
uv run python -m reward_harness.cli --dataset ... --mode evolutionary --output ...

# 导出RM artifact
uv run python -m reward_harness.rm.export --search-output ... --out-artifact ...

# 启动RM server（闭环LLM评分）
uv run python -m reward_harness.rm.server \
  --artifact artifacts/rm_artifacts/rm_v1.json \
  --host 0.0.0.0 \
  --port 8080 \
  --request-log-path artifacts/rm_server_logs/requests.jsonl

# 记录训练前 manifest
uv run python -m reward_harness.rl.record_manifest --manifest artifacts/training_runs/manifests/run_001.json

# 训练结果回填
uv run python -m reward_harness.rl.record_result --result artifacts/training_runs/results/run_001.json

# 评测结果回填
uv run python -m reward_harness.rl.record_eval --report artifacts/training_runs/evals/eval_001.json

# 查询 lineage
uv run python -m reward_harness.rl.show_lineage --training-run-id run_001
```

### 规划API

```bash
# 登记 RL 采样批次（规划）
uv run python -m reward_harness.classifier_rm.record_sample_batch --manifest artifacts/classifier_rm/sample_batches/manifests/sample_batch_001.json

# 构建重复打分数据（规划）
uv run python -m reward_harness.classifier_rm.build_score_dataset --sample-batch sample_batch_001 --repeat-count 5 --aggregation mean

# 构建偏好数据（规划）
uv run python -m reward_harness.classifier_rm.build_preference_dataset --score-dataset score_dataset_001 --pairing-policy hybrid

# 准备 classifier RM 训练 manifest（规划）
uv run python -m reward_harness.classifier_rm.prepare_training --preference-dataset preference_dataset_001 --trainer-project external-classifier-rm
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

- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构顶层地图（域与分层）
- [design-docs/01-architecture.md](design-docs/01-architecture.md) - 详细架构演进
- [design-docs/02-stage-d-rl-lineage.md](design-docs/02-stage-d-rl-lineage.md) - Stage D 详细设计
- [design-docs/03-stage-e-classifier-rm.md](design-docs/03-stage-e-classifier-rm.md) - Stage E 详细设计
- [PLANS.md](PLANS.md) - 执行计划管理
- [PRODUCT_SENSE.md](PRODUCT_SENSE.md) - 产品方向与需求

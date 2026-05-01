# Reward Harness Stage E 设计：基于 RL 采样的 Classifier RM 自动蒸馏

> **版本**: v1.1 | **状态**: 草案 | **最后更新**: 2026-05-01
>
> 本文档细化 Stage E 的设计边界：基于 Stage D 已建立的 RL handshake 与 lineage 平面，自动收集 RL 训练采样、执行重复打分降噪、构建 classifier RM 训练数据，并通过外部 trainer 完成 classifier RM 训练。

---

## 1. 设计目标

Stage D 解决的是 `RM Server -> RL Training` 的契约与追溯。Stage E 的目标是在这条链路上继续向前，形成一个可重复的“蒸馏副环路”：

```text
Rubric RM Server
  -> RL Sampling
    -> Repeated Scoring / Denoising
      -> Preference Dataset
        -> Classifier RM Training
```

核心目标：

1. 自动采集 RL 训练过程中的 `(query, response)` 采样，沉淀为可追溯的数据源。
2. 对同一 `(query, response)` 执行多次重复打分，降低单次 LLM judge 波动带来的噪声。
3. 同时产出两类训练资产：
   - 原始打分数据：`query`、`response`、`score`
   - 偏好数据：`query`、`chosen_response`、`rejected_response`、`margin`
4. 通过和 Stage D 同风格的文件契约，将 classifier RM 训练交给外部项目执行。
5. 保持完整 lineage：
   `ClassifierRM Artifact -> Training Run -> Preference Dataset -> Score Dataset -> RL Sample Batch -> RL Training Run -> RM Deploy -> RM Artifact -> Search Session`

这一步的价值不是替换 rubric RM，而是让高成本的 rubric RM 在 RL 采样分布上蒸馏出更便宜、更稳定、可持续迭代的 classifier RM。

非目标：

- 不在 `reward_harness` 内实现 classifier trainer。
- 不在本阶段定义 classifier RM 的线上 serving 方案。
- 不把 RL repo 或 classifier RM repo 的内部训练框架并入本仓库。
- 不强制保存所有大体量原始样本到 `reward_harness` 仓库内。

---

## 2. 方案选择

### 2.1 备选方案

#### 方案 A：RL Repo 内部自管数据构造与训练

- RL repo 自己导出采样、自行重打分、自行构造偏好对、自行训练 classifier RM。
- `reward_harness` 只保留最终 artifact 引用。

优点：

- `reward_harness` 改动最少。
- 单个项目可快速试验。

缺点：

- 数据构造口径容易漂移。
- 去噪、pairing、lineage 分散在多个 repo，长期不可维护。
- 无法稳定比较不同 classifier RM 的数据来源。

#### 方案 B：`reward_harness` 负责数据平面，外部项目负责训练

- `reward_harness` 维护采样导入、重复打分、偏好构造、数据 registry 与训练契约。
- 外部 classifier RM repo 只负责读取训练 manifest、执行训练、回填结果。

优点：

- 与 Stage D 的 “Contract + Registry + Reference Flow” 边界一致。
- 数据构造口径集中，可比较、可追溯。
- 后续可以自然接到 monitoring 与 closed-loop。

缺点：

- `reward_harness` 需要新增一层数据蒸馏平面。

#### 方案 C：`reward_harness` 内置端到端 classifier RM 训练

- `reward_harness` 内部实现数据构造、训练循环、模型导出与评测。

优点：

- 体验最完整。

缺点：

- 过早扩大仓库职责边界。
- 会把模型框架、分布式训练、checkpoint 管理等问题带入本仓库。

### 2.2 选定方案

本阶段采用 **方案 B：`reward_harness` 负责数据平面，外部项目负责训练**。

结论：

- `reward_harness` 是 rubric RM 蒸馏数据与 classifier RM lineage 的 system of record。
- 外部 RL repo 负责提供采样。
- 外部 classifier RM repo 负责 trainer 执行。
- 双方通过版本化 manifest/result/artifact 契约握手，而不是通过代码耦合。

---

## 3. 系统边界与职责

### 3.1 `reward_harness` 负责

- 定义 RL 采样批次导入契约。
- 基于固定 `rm_artifact_id` / `rm_deploy_id` 执行重复打分与聚合。
- 构建原始打分数据与偏好数据。
- 维护 append-only dataset registry 与 training registry。
- 生成 classifier RM 训练 manifest。
- 记录 classifier RM 训练结果与 artifact 元数据。
- 提供跨 Stage D / Stage E 的 lineage 查询能力。

### 3.2 外部 RL repo 负责

- 训练过程中的采样导出。
- 样本 payload 的生成与持久化。
- 将采样 batch 通过 manifest 方式登记到 `reward_harness`。

### 3.3 外部 classifier RM repo 负责

- 读取 classifier RM 训练 manifest。
- 执行 tokenizer/model/trainer 相关实现。
- 产出训练结果、指标与模型文件。
- 回填 `ClassifierRMTrainingResult` 与 `ClassifierRMArtifact`。

### 3.4 主数据流

```text
TrainingManifest / TrainingResultManifest
  -> RLSampleBatchManifest
    -> RepeatedScoreDataset
      -> PreferenceDataset
        -> ClassifierRMTrainingManifest
          -> ClassifierRMTrainingResult
            -> ClassifierRMArtifact
```

关键约束：

- 采样、重打分、pairing、训练结果必须拆分成独立对象，不能混在一个大 manifest 里。
- Stage E 消费的是 Stage D 的训练 run lineage，而不是绕开 Stage D 直接接 RL repo 内部日志。
- 任一 classifier RM artifact 都必须能反查到它来自哪些 RL 训练 run 与哪一版 rubric RM。

---

## 4. 核心对象与目录约定

### 4.1 `RLSampleBatchManifest`

用途：登记一次从 RL 训练侧导入的采样批次。

建议字段（v1）：

- `schema_version`
- `sample_batch_id`
- `created_at_utc`
- `training_run_id`
- `rm_artifact_id`
- `rm_deploy_id`
- `source`
  - `project`
  - `code_version`
  - `checkpoint_id` optional
  - `sampling_strategy`
- `dataset`
  - `dataset_id`
  - `dataset_version`
  - `split`
- `payload`
  - `format` (`jsonl`)
  - `uri`
  - `sha256`
  - `row_count`
- `notes` optional

对应的样本记录建议至少包含：

- `sample_id`
- `query_id`
- `query`
- `response`
- `response_hash`
- `policy_step` optional
- `episode_id` optional
- `metadata` optional

设计说明：

- `sample_batch_id` 是 Stage E 数据构造的入口主键。
- `response_hash` 用于去重与重打分幂等。
- payload 可以存放在外部 RL repo、本地路径或对象存储，`reward_harness` 只要求可寻址与可校验。

### 4.2 `RepeatedScoreDatasetManifest`

用途：登记一次对采样数据执行重复打分后的原始打分数据集。

建议字段（v1）：

- `schema_version`
- `score_dataset_id`
- `created_at_utc`
- `sample_batch_id`
- `training_run_id`
- `rm_artifact_id`
- `rm_deploy_id`
- `scoring`
  - `mode` (`offline_artifact` recommended, `server_endpoint` optional)
  - `repeat_count`
  - `aggregation`
  - `scoring_code_version`
- `payload`
  - `raw_trace_uri`
  - `aggregated_uri`
  - `row_count`
  - `sha256`
- `summary`
  - `kept_rows`
  - `deduped_rows`
  - `avg_score_std`

推荐聚合后的原始打分记录格式：

```json
{
  "query": "...",
  "response": "...",
  "score": {
    "total": 7.4,
    "criteria": {
      "correctness": 4.7,
      "helpfulness": 2.7
    },
    "repeats": 5,
    "total_std": 0.38,
    "criteria_std": {
      "correctness": 0.21,
      "helpfulness": 0.29
    },
    "aggregation": "mean"
  }
}
```

设计说明：

- 面向训练的“原始打分数据”保持你要求的核心结构：`query`、`response`、`score`。
- 为了支持降噪，`score` 中额外记录重复次数、聚合方式和方差摘要。
- 逐次单次打分 trace 不直接喂给 trainer，但必须保留引用，便于复盘。

### 4.3 `PreferenceDatasetManifest`

用途：把重复打分后的样本转成 classifier RM 可训练的偏好对数据。

建议字段（v1）：

- `schema_version`
- `preference_dataset_id`
- `created_at_utc`
- `score_dataset_id`
- `training_run_id`
- `pairing_policy`
  - `strategy`
  - `min_margin`
  - `max_pairs_per_query`
  - `uncertainty_gate`
- `split`
  - `train_queries`
  - `val_queries`
  - `test_queries`
- `payload`
  - `pairs_uri`
  - `row_count`
  - `sha256`
- `summary`
  - `num_queries`
  - `num_pairs`
  - `dropped_low_margin_pairs`
  - `dropped_high_variance_pairs`

偏好记录推荐最小格式：

```json
{
  "query": "...",
  "chosen_response": "...",
  "rejected_response": "...",
  "margin": 1.8
}
```

推荐附加字段：

- `query_id`
- `chosen_score`
- `rejected_score`
- `chosen_std`
- `rejected_std`
- `source_sample_ids`

设计说明：

- 面向训练的数据最小面保持你要求的格式：`query`、`chosen_response`、`rejected_response`、`margin`。
- query 级 split 必须在 pair 生成前完成，避免同一 query 泄漏到不同集合。

### 4.4 `ClassifierRMTrainingManifest`

用途：声明一次 classifier RM 训练准备怎么跑。

建议字段（v1）：

- `schema_version`
- `classifier_training_run_id`
- `created_at_utc`
- `preference_dataset_id`
- `score_dataset_id`
- `source_training_run_ids`
- `teacher`
  - `rm_artifact_id`
  - `rm_deploy_id`
- `trainer`
  - `project`
  - `repo_url` optional
  - `code_version`
  - `entrypoint`
- `model`
  - `base_model`
  - `tokenizer`
  - `max_length`
- `objective`
  - `type` (`pairwise_classifier`)
  - `loss`
  - `margin_field`
- `hyperparameters`
- `output`
  - `output_dir`
  - `expected_artifact_dir`
- `notes` optional

### 4.5 `ClassifierRMTrainingResult`

用途：记录 classifier RM 训练终态。

建议字段（v1）：

- `schema_version`
- `classifier_training_run_id`
- `status`
- `started_at_utc`
- `finished_at_utc`
- `duration_seconds`
- `trainer_code_version`
- `metrics`
  - `train_loss`
  - `val_loss`
  - `pair_accuracy`
  - `calibration` optional
- `output`
  - `best_checkpoint_path` optional
  - `model_artifact_path` optional
  - `log_path` optional
- `failure` optional

### 4.6 `ClassifierRMArtifact`

用途：登记训练完成后得到的 classifier RM 工件，供后续 serving / eval / compare 使用。

建议字段（v1）：

- `schema_version`
- `classifier_rm_artifact_id`
- `created_at_utc`
- `classifier_training_run_id`
- `preference_dataset_id`
- `score_dataset_id`
- `teacher_rm_artifact_id`
- `base_model`
- `score_schema`
- `artifact_uri`
- `metadata`

### 4.7 `reward_harness` 侧目录约定

建议在现有 `artifacts/` 下新增：

```text
artifacts/
  classifier_rm/
    sample_batches/
      manifests/
    score_datasets/
      manifests/
    preference_datasets/
      manifests/
    training_runs/
      manifests/
      results/
    artifacts/
```

约定：

- registry JSON 放在 `reward_harness` 管理目录下。
- 大体量 payload 默认通过 manifest 中的 `uri` 引用外部路径或对象存储。
- 如需本地落盘，可按 `<id>/part-*.jsonl` 的方式组织，但这不是硬契约。

---

## 5. 关键流程设计

### 5.1 RL 采样导入

1. 外部 RL repo 在训练中或训练后导出 `(query, response)` 采样批次。
2. 生成 `RLSampleBatchManifest`，声明该批次来自哪个 `training_run_id`、哪个 teacher RM。
3. `reward_harness` 校验：
   - `training_run_id` 已在 Stage D registry 中存在
   - `rm_artifact_id` / `rm_deploy_id` 与来源训练 run 不冲突
   - payload `sha256`、`row_count` 与 manifest 一致

### 5.2 多次重复打分降噪

推荐默认策略：

- 对每个唯一 `(query_id, response_hash)` 执行 `N=5` 次独立打分。
- 聚合使用 `mean`，同时保留 `std`。
- 逐 criterion 和总分都记录均值与方差。

推荐执行模式：

- **首选**：`offline_artifact`
  - 直接加载固定 `RMArtifact` 在本地/批处理环境中重放评分
  - 避免依赖在线 endpoint 的瞬时状态
- **可选**：`server_endpoint`
  - 明确绑定 `rm_deploy_id`
  - 适合验证和小规模回放，不作为默认大量数据通道

设计说明：

- Stage E 的去噪目标不是“找到绝对真值”，而是压低单次 LLM judge 抖动。
- `mean + std` 比“只重复后取一次最终分”更适合后续 pair 过滤。

### 5.3 偏好对构造

对每个 `query_id`：

1. 收集该 query 的所有去噪后候选。
2. 按 `score.total` 从高到低排序。
3. 依据 pairing policy 生成偏好对。

推荐默认策略：`hybrid`

- `best_vs_worst`：每个 query 至少保留强信号 pair
- `adjacent_hard_negative`：补充相邻排名 pair，保留更细粒度边界
- `cap_per_query`：限制每个 query 的 pair 数，避免头部 query 主导数据集

pair 保留条件建议同时满足：

- `margin >= min_margin`
- `margin > uncertainty_gate * (chosen_std + rejected_std)`

默认建议：

- `min_margin = 0.8`
- `uncertainty_gate = 1.0`
- `max_pairs_per_query = 4`

设计说明：

- 只按分数排序然后做全量两两配对会造成 `O(n^2)` 膨胀和大量简单样本，v1 不推荐。
- margin 和 uncertainty 同时作为闸门，可以把“高分差但高波动”的 pair 剔除掉。

### 5.4 数据切分与去偏

硬约束：

- 按 `query_id` 做 train/val/test 切分，不能按 pair 随机切分。
- 单个 `training_run_id` 的样本占比需要可配置上限，防止某次 RL run 垄断数据分布。
- 去重键默认使用 `(query_id, response_hash)`。

建议附加策略：

- 对极短回复、模板化回复、空回复设置过滤器。
- 保留每个 query 的响应多样性统计，便于判断是否采样塌缩。

### 5.5 classifier RM 训练编排

1. `reward_harness` 基于 `PreferenceDatasetManifest` 生成 `ClassifierRMTrainingManifest`。
2. 外部 classifier RM repo 消费该 manifest。
3. 训练完成后回填 `ClassifierRMTrainingResult`。
4. 如训练成功，再登记 `ClassifierRMArtifact`。

这一段和 Stage D 的原则一致：

- `reward_harness` 负责契约和 registry。
- trainer 进程生命周期、分布式策略、checkpoint 管理由外部 repo 自行处理。

---

## 6. 质量门与 lineage 查询

### 6.1 数据质量门

Stage E 至少要能回答：

1. 这份 preference dataset 来自哪些 RL run？
2. 它对应的 teacher RM 是哪一版 artifact / deploy？
3. 某个 pair 的 margin 来自哪组原始打分记录？
4. 哪些样本因低 margin 或高方差被丢弃？

### 6.2 训练质量门

至少记录：

- 数据规模：query 数、pair 数、各 split 行数
- 打分稳定性：平均 `std`、高方差样本占比
- 训练效果：`pair_accuracy`、`val_loss`
- 追溯链完整性：是否能从 artifact 回查到 teacher RM 与 RL run

### 6.3 lineage 查询视图

Stage E 后应支持的典型查询：

```text
classifier_rm_artifact
  -> classifier_training_run
  -> preference_dataset
  -> score_dataset
  -> sample_batch
  -> training_run
  -> rm_deploy
  -> rm_artifact
  -> search_session
```

这条链是后续 Stage F 监控与 Stage G 闭环调度的基础。

---

## 7. 推荐 CLI 与用例边界

规划中的入口建议：

```bash
# 登记 RL 采样批次
uv run python -m reward_harness.classifier_rm.record_sample_batch \
  --manifest artifacts/classifier_rm/sample_batches/manifests/sample_batch_001.json

# 基于 teacher RM 重复打分并构建原始打分数据
uv run python -m reward_harness.classifier_rm.build_score_dataset \
  --sample-batch sample_batch_001 \
  --repeat-count 5 \
  --aggregation mean

# 构建偏好对
uv run python -m reward_harness.classifier_rm.build_preference_dataset \
  --score-dataset score_dataset_001 \
  --pairing-policy hybrid \
  --min-margin 0.8

# 生成 classifier RM 训练 manifest
uv run python -m reward_harness.classifier_rm.prepare_training \
  --preference-dataset preference_dataset_001 \
  --trainer-project external-classifier-rm

# 回填训练结果
uv run python -m reward_harness.classifier_rm.record_training_result \
  --result artifacts/classifier_rm/training_runs/results/classifier_run_001.json
```

模块职责建议：

- `reward_harness/classifier_rm/data_models.py`: schema
- `reward_harness/classifier_rm/use_cases.py`: record/build/prepare
- `reward_harness/classifier_rm/io.py`: manifest/payload IO
- `reward_harness/classifier_rm/lineage.py`: cross-stage lineage query

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| RL 采样分布过窄，训练出的 classifier RM 泛化差 | 高 | 记录 run 分布与 query 覆盖度，强制多 run 汇总 |
| 重复打分成本过高 | 高 | 先做 dedup，再打分；默认离线模式；支持采样上限 |
| LLM judge 方差仍偏大 | 高 | 记录 `std` 并在 pair 构建时做 uncertainty gating |
| pair 爆炸导致数据失衡 | 中高 | 使用 `hybrid + cap_per_query`，禁止全量两两配对 |
| 数据泄漏导致验证指标虚高 | 高 | query 级 split，训练前校验无交叉 |
| classifier trainer 口径漂移 | 中高 | 强制 manifest schema 版本与 artifact metadata 回填 |

---

## 9. Stage E 落地切分

### Stage E1：Sample Registry & Score Dataset

目标：打通 RL 采样导入与重复打分。

最小交付：

- `RLSampleBatchManifest` schema
- `RepeatedScoreDatasetManifest` schema
- 采样导入/校验 CLI
- 重复打分与聚合用例

### Stage E2：Preference Builder & Dataset QA

目标：把去噪打分稳定转成 classifier RM 可训练偏好数据。

最小交付：

- `PreferenceDatasetManifest` schema
- hybrid pairing policy
- query 级 split 与数据质量摘要
- lineage 查询补齐到 `score_dataset -> sample_batch -> training_run`

### Stage E3：External Classifier RM Training Handshake

目标：打通外部 classifier RM trainer 的训练前后契约。

最小交付：

- `ClassifierRMTrainingManifest`
- `ClassifierRMTrainingResult`
- `ClassifierRMArtifact`
- 参考交互时序与回填流程文档

---

## 10. Go/No-Go

Stage E 完成后，系统应能稳定回答：

1. 某个 classifier RM 是用哪版 rubric RM 蒸馏出来的？
2. 它训练所用的偏好数据来自哪些 RL run、哪些 query？
3. 任意一个 pair 的 margin 能否反查到对应原始打分记录？
4. 训练失败时，是否仍然能保留完整 manifest/result 记录？
5. 是否能在不侵入 trainer 代码的前提下，替换不同外部 classifier RM 项目？

如果这些问题仍无法稳定回答，则说明 Stage E 还没有形成可靠的数据与训练闭环。

---

## 11. 相关文档

- [docs/design-docs/02-stage-d-rl-lineage.md](02-stage-d-rl-lineage.md)：提供 Stage D 的训练接入与 lineage 设计。
- [docs/design-docs/01-architecture.md](01-architecture.md)：提供从 Harness 到 RM/RL 闭环的总体演进图。
- [docs/ROADMAP.md](../ROADMAP.md)：记录阶段目标与优先级。

本文件补足 Stage E 的详细设计，是后续执行计划与实现工作的直接输入。

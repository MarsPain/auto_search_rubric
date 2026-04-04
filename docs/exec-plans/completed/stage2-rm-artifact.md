# Stage 2: RM Artifact 契约与部署

> **状态**: 部分完成 🚧 | **优先级**: P0 | **创建日期**: 2026-04-04 | **完成日期**: - (deploy manifest 待补)

---

## 目标

把"best rubric JSON"升级为可部署、可追溯的 RM artifact。

---

## 已完成任务 ✅

### 1. 定义 RMArtifact schema v1 ✓

新增 `autosr/rm/data_models.py`:

```python
@dataclass
class RMArtifact:
    artifact_id: str                    # 唯一标识
    created_at_utc: str                 # ISO格式时间戳
    source_session_id: str              # 来源搜索会话
    dataset_hash: str                   # 数据集指纹
    config_hash: str                    # 配置指纹
    rubric: dict                        # rubric内容
    scoring_policy: dict                # 评分策略
    normalization: dict                 # 归一化配置
    compatibility: dict                 # 兼容性信息
```

### 2. artifact 导出能力 ✓

新增 `autosr/rm/use_cases.py`:
- `build_artifact()`: 从搜索结果构建artifact
- `export_artifact_to_json()`: 导出为JSON

新增 CLI 入口 `autosr.rm.export`:

```bash
uv run python -m autosr.rm.export \
  --search-output artifacts/best_rubrics.json \
  --out-artifact artifacts/rm_artifacts/rm_v1.json
```

### 3. artifact 校验器 ✓

新增 `autosr/rm/validators.py`:
- `ArtifactValidator`: schema + 必填字段校验
- `HashConsistencyChecker`: dataset/config hash 一致性校验
- `RubricFingerprintChecker`: rubric 内容指纹校验

---

## 待完成任务 📋

### Deploy Manifest（发布记录）

目标：谁在何时发布了哪个 artifact。

建议字段：
```python
@dataclass
class DeployManifest:
    deploy_id: str
    artifact_id: str
    deployed_at_utc: str
    deployed_by: str              # 操作人/服务账号
    deployment_target: str        # 目标环境
    previous_artifact_id: str     # 回滚用
    rollback_policy: dict         # 回滚策略
```

---

## 交付物

- `autosr/rm/` 子包：
  - `data_models.py` - RMArtifact schema
  - `use_cases.py` - build/export
  - `validators.py` - 校验逻辑
  - `export.py` - CLI入口
- `artifacts/rm_artifacts/*.json`
- `run_records/*` 与 RM artifact 的关联字段（`source_session_id`）

---

## Go/No-Go 检查

- [x] 定义 RMArtifact schema v1
- [x] 新增 artifact 导出能力
- [x] 增加 artifact 校验器
- [ ] 定义 RM 发布记录（deploy manifest）
- [ ] 给定同一 artifact，RM 打分结果可重复
- [ ] 每次 RM 部署都可追溯到搜索会话和数据集版本

---

## 关联文档

- [设计文档](../../DESIGN.md)
- [产品感](../../PRODUCT_SENSE.md)
- [架构演进](../../design-docs/01-architecture.md)

# Stage 0 Implementation Summary

## 目标
在不改变搜索算法行为的前提下，补齐会话化和可恢复改造的地基。

## 已完成任务

### 1. 新增 `autosr/harness/session.py` ✓
- `SearchSession` 类：Session 生命周期包装器，不侵入算法核心
- 支持 `create()` 工厂方法创建新会话
- 支持 `run_to_completion()` 委托给底层 searcher
- API 设计兼容 Stage 1 的 resume/step 执行模式（当前抛出 NotImplementedError）

### 2. 明确 `SearchCheckpoint` v1 schema ✓
- 新增 `autosr/harness/state.py`
- `SearchCheckpoint` dataclass 定义完整的 checkpoint schema:
  - `session_id`: 会话标识
  - `generation`: 当前代数
  - `best_rubrics`/`best_scores`/`history`: 搜索状态
  - `scheduler_state`/`rng_state`: 可恢复状态
  - `config_hash`/`dataset_hash`: 兼容性校验
  - `schema_version`: 前向兼容
- 完整的序列化/反序列化支持 (to_dict/from_dict/to_json/from_json)
- `CheckpointValidationError` 用于数据验证

### 3. 增加 resume 兼容检查 ✓
- `ResumeValidator` 类：配置/数据集哈希校验
- `ResumeValidationResult`：验证结果封装
- `ResumeCompatibilityError`：不兼容时抛出
- `compute_config_hash()`: 配置哈希计算（确定性，与 key 顺序无关）
- `compute_dataset_hash()`: 数据集文件哈希计算

## 测试覆盖

新增 `tests/test_harness_stage0.py`，包含 29 个测试：

### SearchCheckpoint Schema 测试
- checkpoint 创建和验证
- 序列化/反序列化 roundtrip
- JSON 序列化 roundtrip
- schema version 兼容性检查

### ResumeValidator 测试
- 兼容 checkpoint 验证通过
- 不兼容 config hash 检测
- 不兼容 dataset hash 检测
- validate_or_raise 行为

### Hash 函数测试
- config hash 确定性
- config hash 与 key 顺序无关
- config hash 对值敏感
- dataset hash 计算

### SearchSession 测试
- session 创建（带/不带 session_id）
- run_to_completion 执行
- 重复执行保护
- get_result 行为
- resume/step API placeholder（Stage 1 实现）
- checkpoint API placeholder（Stage 1 实现）
- session metadata 获取

### 向后兼容测试
- CLI 不使用 harness 时正常工作
- harness wrapper 与直接使用产生相同结构的结果

## Go/No-Go 检查

- [x] 不启用 checkpoint 时，CLI 输出与现状一致（结构和关键字段一致）
- [x] 架构回归测试通过（含 `tests/test_architecture_refactor.py`）

## 文件结构

```
autosr/harness/
├── __init__.py          # 公开 API 导出
├── session.py           # SearchSession 生命周期管理
└── state.py             # SearchCheckpoint schema + ResumeValidator

tests/
└── test_harness_stage0.py  # Stage 0 完整测试套件
```

## API 示例

### 基本用法（Stage 0）
```python
from autosr.harness import SearchSession

session = SearchSession.create(
    prompts=prompts,
    config=config,
    factory=factory,
)
result = session.run_to_completion()
```

### Checkpoint Schema
```python
from autosr.harness import SearchCheckpoint, ResumeValidator

checkpoint = SearchCheckpoint(
    session_id="session_001",
    generation=5,
    best_rubrics={...},
    best_scores={...},
    history={...},
    scheduler_state={...},
    rng_state={...},
    config_hash="sha256...",
    dataset_hash="sha256...",
)

# 验证兼容性
validator = ResumeValidator(current_config_hash, current_dataset_hash)
validator.validate_or_raise(checkpoint)
```

## 进入 Stage 1 的准备

Stage 0 已建立以下基础，Stage 1 将实现：
1. `StateManager` 类：原子文件写入持久化
2. `SearchSession.resume()`：从 checkpoint 恢复
3. `SearchSession.run_step()`：单步执行 + 每代 checkpoint
4. CLI 新增 `--resume-from`, `--checkpoint-every-generation` 等参数

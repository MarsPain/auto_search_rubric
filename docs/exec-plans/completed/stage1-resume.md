# Stage 1 Implementation Summary

> **状态**: 已完成 ✅ | **优先级**: P0 | **创建日期**: 2026-04-03 | **完成日期**: 2026-04-04

---

## 目标

把"长时任务可中断可恢复"做成稳定能力。

---

## 已完成任务

### 1. 状态持久化（State Store）✓

- 新增 `autosr/harness/storage.py`:
  - `StateManager`: 原子文件写入的 checkpoint 持久化
  - `CheckpointMetadata`: checkpoint 元数据
  - 支持按 session_id 或具体路径加载 checkpoint
  - 列出、删除 checkpoint 和 session 的功能

存储格式：
```
<checkpoint_dir>/<session_id>/gen_<generation:04d>.json
```

原子写入策略：
1. 写入临时文件
2. 重命名到目标路径

### 2. 演化循环单步化 ✓

- 更新 `SearchSession`:
  - `run_step()`: 执行单代演化 (global_batch scope)
  - `_init_global_step_state()`: 初始化 step 执行状态
  - `_execute_global_generation()`: 执行单代
  - `_finalize_global_search()`: 结束搜索并生成结果
  - `_save_checkpoint()`: 保存 checkpoint

- Resume 支持:
  - 从 checkpoint 恢复状态
  - 重新初始化种群
  - 继续执行剩余代数

### 3. CLI 与可观测增强 ✓

新增 CLI 参数：
- `--checkpoint-dir`: checkpoint 存储目录 (default: ./checkpoints)
- `--checkpoint-every-generation`: 每代保存 checkpoint
- `--checkpoint-interval-seconds`: 定时保存间隔
- `--resume-from <session_id|path>`: 从 checkpoint 恢复

输出增强：
- Session 信息（session_id, resumed_from, checkpoint_enabled）
- 日志记录 checkpoint 保存路径

---

## 测试覆盖

新增 `tests/test_harness_stage1.py` (18 个测试)：

### StateManager 测试
- save/load checkpoint
- load by path
- load nonexistent (error)
- load corrupted (error)
- list checkpoints
- list sessions
- find latest
- delete checkpoint/session

### SearchSession Step 执行测试
- step execution (evolutionary global_batch)
- iterative mode (NotImplementedError)
- prompt_local scope (NotImplementedError)
- checkpoint saved every generation

### SearchSession Resume 测试
- resume from session_id
- resume from checkpoint path
- resume requires prompts
- incompatible config (warning)

### 集成测试
- full checkpoint → resume → complete cycle

---

## 使用示例

### 基本 checkpoint 模式

```bash
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --backend mock \
  --output results.json \
  --checkpoint-every-generation \
  --checkpoint-dir ./my_checkpoints
```

### Resume 模式

```bash
# 先运行并生成 checkpoints
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --generations 3 \
  --checkpoint-every-generation

# 然后 resume 并增加代数
uv run python -m autosr.cli \
  --dataset examples/demo_dataset.json \
  --mode evolutionary \
  --generations 10 \
  --checkpoint-every-generation \
  --resume-from <session_id>
```

### Python API

```python
from autosr.harness import SearchSession, StateManager

# Create with checkpointing
state_manager = StateManager(base_dir="./checkpoints")
session = SearchSession.create(
    prompts=prompts,
    config=config,
    factory=factory,
    state_manager=state_manager,
    checkpoint_every_generation=True,
    dataset_path=Path("./dataset.json"),
)

# Run with step-wise execution
while not session.is_finished():
    result = session.run_step()
    print(f"Generation {result.generation} complete")

# Or run to completion
result = session.run_to_completion()

# Resume
session2 = SearchSession.resume(
    resume_from="session_id",
    config=config,
    factory=factory,
    state_manager=state_manager,
    prompts=prompts,
    dataset_path=Path("./dataset.json"),
)
result = session2.run_to_completion()
```

---

## 兼容性说明

- Dataset hash 必须匹配（防止数据不一致）
- Config hash 可以不同（允许调整参数如 generations）
- 仅支持 evolutionary + global_batch scope 的 checkpoint
- Iterative 模式和 prompt_local scope 暂不支持 step 执行

---

## 已知限制

1. RNG 状态恢复可能不完全（取决于 Python random 模块版本）
2. Scheduler 状态目前只保存 diagnostics，不保存完整状态
3. prompt_local scope 的 checkpoint 在 Stage 2 中实现

---

## Go/No-Go 检查

- [x] `autosr/harness/storage.py` + `autosr/harness/session.py` 可运行
- [x] `--resume-from` 打通
- [x] 中断恢复 E2E 测试通过
- [x] 恢复失败时有明确错误分类
- [x] 恢复后最终 `best_scores` 与不中断运行偏差在容忍范围内
- [x] 长时运行无状态损坏
- [x] 架构回归测试通过

---

## 关联文档

- [设计文档](../../DESIGN.md)
- [架构演进](../../design-docs/01-architecture.md)
- [Stage 0 计划](./stage0-harness.md)

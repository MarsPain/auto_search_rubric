# Stage D3: Comparative Experiment View

> **状态**: 已完成 | **优先级**: P0 | **负责人**: AutoSR Team | **创建日期**: 2026-04-22 | **完成日期**: 2026-04-22

## 目标

补齐阶段 D 的最后一块拼图：横向比较、baseline 对比、回归检测能力，让阶段 D 真正闭环。

## 背景

阶段 D1（Contract & Registry）和 D2（External RL Reference Flow）已代码落地，但 D3 缺失，导致设计文档中的 7 条成功标准最后 1 条无法满足：

> "相比 baseline 或上一个 artifact，表现是变好还是变差？"

## 任务清单

- [x] 核心比较引擎 (`autosr/rl/comparison.py`)
  - [x] `ArtifactSummary` — 按 artifact 聚合统计视图
  - [x] `RunComparison` / `MetricDelta` — run 间指标对比
  - [x] `RegressionSignal` — 回归信号分级（critical/warning/info）
  - [x] `ArtifactMetricTable` — 多 artifact 横向比较表
  - [x] `compare_runs()` — 对比两个 run 的评测结果
  - [x] `compare_artifacts()` — 横向比较多个 artifact
  - [x] `detect_regression()` — 回归检测（支持显式/自动 baseline）
  - [x] `detect_anomalies()` — 异常 run 识别
  - [x] `summarize_artifact()` — artifact 聚合摘要
- [x] Registry 查询增强 (`autosr/rl/registry.py`)
  - [x] `list_runs_by_artifact()`
  - [x] `list_runs_by_dataset_version()`
  - [x] `list_runs_by_status()`
- [x] CLI 新增与增强
  - [x] `autosr.rl.cli.compare_runs` — 对比两个 run
  - [x] `autosr.rl.cli.compare_artifacts` — 横向比较 artifact
  - [x] `autosr.rl.cli.check_regression` — 检测回归
  - [x] `autosr.rl.cli.list_runs` — 增强版列表（过滤+异常标记）
  - [x] `autosr.rl.cli.show_lineage` — 增加 `--with-baseline-delta`
- [x] CLI 入口 shim（向后兼容）
- [x] `autosr/rl/__init__.py` 导出更新
- [x] 测试 (`tests/test_rl_comparison.py`，38 个测试全部通过)
- [x] 文档同步 (`PLANS.md`, `ROADMAP.md`)

## 验收标准

- [x] 能横向比较不同 `rm_artifact_id` 的训练与评测结果
- [x] 能快速识别回归或异常 run
- [x] 相比 baseline 或上一个 artifact，表现是变好还是变差？ ← 可稳定回答

## 变更日志

- 2026-04-22: 创建计划并全部完成实现

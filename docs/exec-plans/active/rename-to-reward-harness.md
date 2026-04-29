# Rename to Reward Harness

> **状态**: 活跃 | **优先级**: P1 | **负责人**: AutoSR Team | **创建日期**: 2026-04-29

## 目标

将项目主命名从 `auto_search_rubric` / `autosr` 平滑迁移到 `Reward Harness` / `reward_harness`，让项目名称匹配当前 RM+RL 自动迭代优化 harness 的架构边界，同时保留旧入口兼容性。

## 背景

阶段 A-D 已完成后，项目已经覆盖搜索会话、RM artifact、RM server、RL 训练接入、实验 registry、lineage 查询与比较视图。后续阶段 E-G 还会继续扩展 classifier RM 蒸馏、监控与闭环调度。

`auto_search_rubric` 只描述早期 rubric 搜索能力，已经无法准确表达当前系统。命名设计决策见 [Reward Harness Renaming 设计决策](../../design-docs/05-reward-harness-renaming.md)。

## 范围

### 包含

- 新增 `reward_harness` 推荐包入口。
- 保留 `autosr` 兼容入口。
- 更新 README、AGENTS、ARCHITECTURE、DESIGN、ROADMAP、PRODUCT_SENSE 中的主名称与命令示例。
- 更新可复现脚本生成逻辑中的推荐模块路径。
- 增加兼容性测试，验证新旧入口行为一致。
- 更新文档校验与单元测试。

### 不包含

- 不删除 `autosr` 包。
- 不修改已发布 artifact / registry 的历史主键。
- 不重写已完成执行计划中的历史事实。
- 不实现新的 RM/RL 功能。

## 任务清单

- [ ] Task 1: 建立 `reward_harness` 包级兼容入口
  - 新增 `reward_harness/__init__.py`，复用现有 `autosr` 顶层公开对象。
  - 新增 `reward_harness/cli.py`，转发到 `autosr.cli.main`。
  - 新增 `reward_harness` 下必要子包 shim：`rm`、`rl`、`search`、`harness`、`content_extraction`、`llm_components`、`prompts`、`run_records`。
  - 修改 `pyproject.toml` 的 package discovery，使 `reward_harness*` 被包含。

- [ ] Task 2: 补新旧入口兼容测试
  - 新增 `tests/test_reward_harness_compat.py`。
  - 验证 `reward_harness.data_models.Rubric is autosr.data_models.Rubric`。
  - 验证 `python -m reward_harness.cli --help` 成功退出。
  - 验证至少一个 RM CLI shim 与一个 RL CLI shim 可显示 help。

- [ ] Task 3: 更新运行时生成的推荐命令
  - 修改 `autosr/run_records/use_cases.py`，让新生成的 replay script 优先使用 `reward_harness.cli`。
  - 保留旧脚本和旧测试兼容，必要时在测试中同时接受 `autosr.cli` 与 `reward_harness.cli`。
  - 更新 `tests/test_cli_reproducibility.py` 中的期望。

- [ ] Task 4: 更新文档主叙事
  - `README.md` / `README.zh.md`: 标题改为 Reward Harness，首屏说明历史名称和兼容入口。
  - `AGENTS.md`: 标题和核心约束改为 Reward Harness，但保留 `autosr` 兼容命令。
  - `docs/ARCHITECTURE.md`: 顶层地图使用 Reward Harness 作为主名，包名说明采用 `reward_harness` 推荐、`autosr` 兼容。
  - `docs/DESIGN.md`: 更新目标架构中的推荐导入路径与 CLI 示例。
  - `docs/PRODUCT_SENSE.md`: 将产品愿景主体从 AutoSR 改为 Reward Harness。
  - `docs/ROADMAP.md`: 保持阶段内容不变，更新路线图主名与 API 演进策略。

- [ ] Task 5: 更新示例脚本与开发命令
  - 更新 `examples/*.py` 与 `examples/*.sh` 中面向用户的推荐导入/命令。
  - 更新 `scripts/run_formal_search.sh` 的推荐模块路径；若兼容风险高，则保留旧入口并在注释中解释。
  - 确认 `scripts/run_tests_unit.sh` 和 `scripts/run_quality_checks.sh` 不因包名变化失效。

- [ ] Task 6: 质量门禁
  - 运行 `uv run python -m unittest tests.test_reward_harness_compat tests.test_cli_reproducibility`。
  - 运行 `./scripts/run_tests_unit.sh`。
  - 运行 `uv run python scripts/validate_docs.py`。
  - 运行 `./scripts/run_quality_checks.sh`。

- [ ] Task 7: 收尾
  - 确认 active plan 全部勾选。
  - 将本计划移动到 `docs/exec-plans/completed/`。
  - 更新 `docs/PLANS.md` 的 Active / Completed 表。
  - 在变更日志中记录迁移完成日期。

## 验收标准

- [ ] `uv run python -m reward_harness.cli --help` 成功运行。
- [ ] `uv run python -m autosr.cli --help` 继续成功运行。
- [ ] 核心领域对象的新旧导入路径指向同一对象。
- [ ] 新文档把 Reward Harness 作为主名，`autosr` 作为兼容入口。
- [ ] 旧 artifact / registry 文件无需迁移即可继续被读取。
- [ ] 单元测试通过。
- [ ] 文档校验通过。
- [ ] 质量检查通过。

## 风险与处理

| 风险 | 影响 | 处理 |
|------|------|------|
| Python 模块 shim 造成对象身份不一致 | 兼容测试失败，下游 isinstance 判断异常 | shim 必须 re-export 原对象，并用测试锁定 |
| CLI help 或 subprocess 入口遗漏 | 用户新命令不可用 | 每个对外 CLI 至少覆盖 help 测试 |
| 大量文档历史记录被误改 | 时间线混乱 | 已完成计划只保留历史事实，不做全量替换 |
| replay script 改名破坏旧测试 | 可复现实验脚本不稳定 | 测试同时验证新推荐入口和旧兼容入口 |
| 过早删除 `autosr` | 外部脚本断裂 | 本计划明确不删除旧包 |

## 关联文档

- [Reward Harness Renaming 设计决策](../../design-docs/05-reward-harness-renaming.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [DESIGN.md](../../DESIGN.md)
- [ROADMAP.md](../../ROADMAP.md)

## 变更日志

- 2026-04-29: 创建命名迁移执行计划。


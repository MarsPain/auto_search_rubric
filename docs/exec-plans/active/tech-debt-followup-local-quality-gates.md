# Tech Debt Follow-up: Local Quality Gates

> **状态**: 活跃 | **优先级**: P1 | **负责人**: AutoSR Team | **创建日期**: 2026-04-26

## 目标

在 Stage E Readiness 已清偿阻塞债务的基础上，继续处理非阻塞技术债，并优先把质量门禁收敛到本地可重复执行的工作流。

## 范围

本计划不引入 GitHub Actions 或其他远端 CI/CD。中短期质量保证以本地命令为准，确保开发者在提交前可以稳定运行：

- `./scripts/run_tests_unit.sh`
- `uv run python scripts/validate_docs.py`
- `./scripts/run_quality_checks.sh`
- mypy 静态类型检查入口

## 任务清单

- [ ] 明确本地质量门禁的唯一推荐命令组合，并同步到相关文档
- [ ] 将 mypy 纳入本地质量检查计划，先建立可运行入口，再分阶段收紧规则
- [ ] 决定 ruff format 的推进策略：继续作为显式报告，或在完成批量格式化后纳入硬门禁
- [ ] 为 `selection_strategies.py` 增加直接单元测试
- [ ] 为 `mix_reward.py` 增加直接单元测试
- [ ] 为 `adaptive_mutation.py` 增加直接单元测试
- [ ] 修复 `rm/use_cases.py` 中 LLM 默认值重复定义问题
- [ ] 将 tournament selection 的 `id()` 去重替换为语义稳定的指纹去重
- [ ] 明确 `autosr.models` 兼容 shim 的长期保留或弃用策略

## 验收标准

- [ ] `./scripts/run_tests_unit.sh` 通过
- [ ] `uv run python scripts/validate_docs.py` 通过
- [ ] `./scripts/run_quality_checks.sh` 通过或明确报告非硬门禁项
- [ ] mypy 本地入口可执行，并在文档中说明当前检查范围
- [ ] `docs/PLANS.md` 与技术债审计报告的状态一致

## 关联文档

- [技术债审计](../tech-debt/tech-debt-audit-2026-04.md)
- [Stage E Readiness - Tech Debt Sprint](../completed/stage-e-readiness-tech-debt-sprint.md)
- [计划管理](../../PLANS.md)

## 变更日志

- 2026-04-26: 创建后续清债计划；明确中短期采用本地质量门禁，不引入 GitHub CI/CD。

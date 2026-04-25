# Stage E Readiness: Tech Debt Sprint

> **状态**: 活跃 | **优先级**: P0 | **负责人**: AutoSR Team | **创建日期**: 2026-04-25

## 目标

在进入 Stage E（基于 RL 采样的 Classifier RM 自动蒸馏）之前，清偿会影响 lineage、checkpoint、配置可复现性和开发质量门禁的阻塞债务。

## 范围

本计划只处理 Stage E 的进入门槛，不实现 `autosr/classifier_rm/` 的新业务能力。

## 任务清单

- [x] 同步 Stage D 完成状态到 `PLANS.md`、`ROADMAP.md`、`ARCHITECTURE.md` 与 `DESIGN.md`
- [x] 修复 `TrainingManifest.from_json` 重复声明
- [x] 修复 `LineageView` 可变字段默认值写法
- [x] 修复 registry fallback 扫描中的重复 eval 读取
- [x] 明确 `create_verifier_with_extraction(prompts)` 的兼容参数策略
- [x] 统一原子写 JSON 原语并替换 RM/RL 局部重复实现
- [x] 修复 `CheckpointCallback` 类型别名重复
- [x] 修复 `_config_to_dict` 手工白名单导致的 hash 覆盖风险
- [x] 细化 checkpoint 保存异常策略，避免静默失败
- [ ] 定义 `Searcher` / `SteppableSearcher` 协议，解耦 `SearchSession` 与具体搜索实现
- [ ] 配置基础静态工具链（优先 ruff，mypy 分阶段推进）

## 验收标准

- [ ] `./scripts/run_tests_unit.sh` 通过
- [ ] `uv run python scripts/validate_docs.py` 通过
- [ ] Stage E 阻塞项在技术债审计报告中标注清偿状态
- [ ] `docs/PLANS.md` 中本计划状态与实际文件位置一致

## 关联文档

- [技术债审计](../tech-debt/tech-debt-audit-2026-04.md)
- [Stage E 设计](../../design-docs/03-stage-e-classifier-rm.md)
- [架构设计](../../DESIGN.md)
- [路线图](../../ROADMAP.md)

## 变更日志

- 2026-04-25: 创建计划，并完成第一批低风险代码债与文档状态同步。
- 2026-04-25: 完成共享原子写原语收敛，并统一 `CheckpointCallback` 类型契约。
- 2026-04-25: 修复 config hash 全量字段覆盖，并将 checkpoint 保存失败改为显式异常。

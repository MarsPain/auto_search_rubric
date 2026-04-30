# Reward Harness Renaming 设计决策

> **版本**: v1.0 | **状态**: 草案 | **最后更新**: 2026-04-29

本文档记录项目从 `auto_search_rubric` / `autosr` 迁移到 `Reward Harness` / `reward_harness` 的命名决策。执行计划见 [rename-to-reward-harness.md](../exec-plans/completed/rename-to-reward-harness.md)。

---

## 1. 背景

项目最初聚焦于自动搜索 rubric，因此仓库名 `auto_search_rubric` 与包名 `autosr` 能准确描述早期能力。

随着阶段 A-D 完成，系统边界已经扩展为：

```text
Rubric Search
  -> RM Artifact
  -> RM Server
  -> RL Training
  -> Classifier RM Distillation
  -> Eval & Monitoring
  -> Search Refresh
```

当前主线已经不是单次 rubric 搜索，而是 reward model 与 RL 训练之间的自动迭代优化底座。继续使用 `auto_search_rubric` 会把项目锚定在早期子能力上，削弱 RM server、RL registry、classifier RM 数据平面、监控与闭环调度的产品表达。

---

## 2. 命名目标

新名称需要满足：

1. 覆盖 rubric RM、classifier RM、RM server、RL 训练接入、评测与闭环调度。
2. 不承诺仓库内置完整 RL trainer 或 classifier RM trainer。
3. 能与现有文档中的 “Reward Harness” 语言自然对齐。
4. 允许旧入口平滑兼容，避免一次性破坏现有脚本、测试与外部引用。
5. 与 Python 包名、CLI 示例、文档标题保持一致。

---

## 3. 决策

采用以下新命名：

| 层级 | 新名称 | 说明 |
|------|--------|------|
| 产品/项目名 | Reward Harness | 对齐系统职责：reward 工程化、版本化、部署、训练接入与闭环追溯 |
| 仓库名 | `reward-harness` | 面向 Git/release/外部文档的推荐名称 |
| Python 包名 | `reward_harness` | 新代码与新文档优先使用 |
| 旧包名 | `autosr` | 作为兼容入口保留一段迁移期 |
| 旧仓库名 | `auto_search_rubric` | 在文档中标记为历史名称 |

`AutoSR` 不再作为主品牌推进。它可以在迁移期内作为历史名称或兼容入口出现，但不再承担新能力的产品表达。

---

## 4. 兼容策略

迁移采用 “先新增新名，再保留旧名，再逐步收敛引用” 的方式。

### 4.1 包入口

- 新增 `reward_harness` 包作为推荐导入路径。
- `autosr` 初期保留为兼容 shim。
- 新代码优先从 `reward_harness.*` 导入。
- 旧代码中的 `autosr.*` 不在第一阶段强制删除。

### 4.2 CLI 入口

新入口：

```bash
uv run python -m reward_harness.cli
uv run python -m reward_harness.rm.export
uv run python -m reward_harness.rm.server
uv run python -m reward_harness.rl.record_manifest
```

兼容入口：

```bash
uv run python -m autosr.cli
uv run python -m autosr.rm.export
uv run python -m autosr.rm.server
uv run python -m autosr.rl.record_manifest
```

第一阶段必须保证两组入口都能运行。是否给旧入口增加 deprecation warning，需要在执行时单独评估；默认不在导入阶段发 warning，以免破坏测试与脚本输出。

### 4.3 Artifact 与 registry

已有 artifact、deploy manifest、training manifest、eval report 中的历史字段不做破坏性迁移。新生成记录可以逐步使用 `reward_harness` 作为工具名或 runtime package 标识，但 lineage 主键、artifact id、training run id 不应因重命名而变化。

---

## 5. 非目标

本次重命名不做以下事情：

- 不改变 RMArtifact、TrainingManifest、EvalReport 等领域 schema 的业务含义。
- 不把外部 RL trainer 或 classifier RM trainer 并入本仓库。
- 不删除 `autosr` 兼容入口。
- 不把所有历史文档一次性改写为只保留新名称。
- 不修改已经完成计划中的历史事实描述。

---

## 6. 迁移阶段

### 阶段 1：文档与计划

- 记录命名设计决策。
- 创建 active 执行计划。
- 更新 `docs/PLANS.md`。

### 阶段 2：代码兼容入口

- 新增 `reward_harness` 包入口。
- 让新入口复用现有实现。
- 补测试验证 `reward_harness.*` 与 `autosr.*` 的关键对象一致。

### 阶段 3：文档与示例迁移

- README、AGENTS、ARCHITECTURE、DESIGN、ROADMAP 中优先展示新入口。
- 保留 “Legacy `autosr` compatibility” 小节。
- 更新示例脚本与可复现脚本生成逻辑。

### 阶段 4：质量门禁与收尾

- 运行单元测试与文档校验。
- 确认旧入口仍兼容。
- 将执行计划移动到 `docs/exec-plans/completed/`。

---

## 7. 成功标准

- 新用户能从文档中理解项目主名是 Reward Harness。
- 新代码可以通过 `reward_harness` 导入核心模块。
- 旧命令 `uv run python -m autosr.cli` 继续可用。
- 新命令 `uv run python -m reward_harness.cli` 可用。
- docs validation 与单元测试通过。
- 已完成的历史计划仍能保留当时的 AutoSR 语境，不制造时间线混乱。


# Complete Migration from autosr to reward_harness

> **状态**: 活跃 | **优先级**: P1 | **负责人**: AutoSR Team | **创建日期**: 2026-04-30

---

## 目标

将项目代码主体从 `autosr/` 物理迁移到 `reward_harness/`，使 `reward_harness` 成为代码 canonical 位置，`autosr` 降级为纯兼容 shim。完成 `pyproject.toml`、测试、脚本、文档中的剩余命名收敛。

第一阶段（已完成）通过新增 `reward_harness/` shim 层实现了**入口兼容**；本阶段的目标是完成**代码归属反转**与**引用彻底收敛**。

---

## 背景

当前状态：

- `autosr/` 包含 78 个 Python 文件，是全部实际业务代码的物理位置。
- `reward_harness/` 包含 37 个 Python 文件，均为从 `autosr` re-export 的 shim。
- 当前 `reward_harness` shim 只覆盖顶层模块与部分 CLI 入口；`reward_harness.search.config`、`reward_harness.harness.session`、`reward_harness.rm.data_models`、`reward_harness.rl.registry` 等深层模块路径尚不可导入，但 README / ARCHITECTURE 已经把这些路径描述为推荐实现位置。
- `tests/` 中约 194 处引用仍从 `autosr.*` 导入。
- `pyproject.toml` 的 `name = "autosr"`，pip 安装名仍是旧名称。
- `scripts/run_quality_checks.sh` 与 `tests/test_static_tooling.py` 仍只覆盖 `autosr`，尚未把新增的 `reward_harness` shim 包纳入质量门禁。

目标状态：

- `reward_harness/` 是业务代码的 canonical 物理位置。
- `autosr/` 仅包含兼容 shim（import + re-export），无业务逻辑。
- 所有文档中宣称的 `reward_harness.*` 深层模块路径均可导入。
- `tests/` 优先从 `reward_harness.*` 导入。
- `pyproject.toml` 的 `name = "reward_harness"`。
- Ruff / format / mypy 质量门禁同时覆盖 `reward_harness` 和仍受支持的 `autosr` 兼容 shim。
- 旧入口 `python -m autosr.cli` 继续可用。

---

## 范围

### 包含

- `autosr/` → `reward_harness/` 代码物理迁移。
- `autosr/` 重建为**反向兼容 shim**（从 `reward_harness` re-export）。
- `reward_harness/` 内部所有 import、logger、docstring、CLI help 中的 `autosr` → `reward_harness`。
- `tests/` 中所有 import 路径迁移到 `reward_harness.*`。
- `pyproject.toml` 中的 `name`、`description`、`src` 更新。
- `scripts/run_quality_checks.sh` 与 `tests/test_static_tooling.py` 中的硬编码路径更新，确保两个包都进入质量门禁。
- `README.md`、`AGENTS.md` 中剩余实现路径引用的更新。

### 不包含

- **不删除** `autosr/` 兼容包（继续支持旧入口）。
- **不修改** `artifacts/`、`run_records/` 中的历史产物。
- **不修改** 已完成执行计划中的历史事实描述。
- **不迁移** `docs/design-docs/` 中的历史设计上下文（仅更新实现路径引用）。

---

## 阶段与任务

### Phase 1: 自动化脚本准备

- [ ] **Task 1.1**: 编写 `scripts/migrate_internal_refs.py`
  - 支持 `--dry-run` 模式，输出将要替换的引用清单。
  - 支持 `--path` 指定目标目录（`reward_harness/`、`tests/`、`scripts/`）。
  - 替换规则（按优先级，避免误伤）：
    1. `from autosr.` → `from reward_harness.`
    2. `import autosr.` → `import reward_harness.`
    3. `logging.getLogger("autosr` → `logging.getLogger("reward_harness`
    4. `python -m autosr.` → `python -m reward_harness.`（docstring / 注释 / 字符串中）
    5. `"autosr.` → `"reward_harness.`（字符串字面量中的模块路径，如 subprocess 调用）
  - 排除：`__pycache__`、`.git/`、`artifacts/`、`run_records/`、`.venv/`。
  - 对 `tests/` 单独支持 `--tests-only` 模式，同时处理 `assertLogs("autosr` → `assertLogs("reward_harness`。

- [ ] **Task 1.1a**: 编写/生成模块清单
  - 基于 `find autosr -name '*.py'` 生成迁移清单，明确每个旧模块的目标状态：
    - `move`: 业务实现移动到 `reward_harness`。
    - `shim`: 旧路径保留为兼容 re-export。
    - `skip`: 历史产物或非包代码不迁移。
  - 清单必须覆盖深层模块，例如：
    - `autosr.search.config` → `reward_harness.search.config`
    - `autosr.harness.session` → `reward_harness.harness.session`
    - `autosr.rm.data_models` → `reward_harness.rm.data_models`
    - `autosr.rl.registry` → `reward_harness.rl.registry`
    - `autosr.llm_components.base` → `reward_harness.llm_components.base`
    - `autosr.content_extraction.strategies` → `reward_harness.content_extraction.strategies`
  - 清单必须单独标注 `autosr/rl/*.py` 与 `autosr/rl/cli/*.py` 的目标，避免扁平化 CLI 入口互相覆盖。

- [ ] **Task 1.2**: 运行 dry-run，人工 review 替换清单
  - 确认没有误替换（如 `"autosr"` 出现在非代码上下文中）。
  - 确认 `autosr/models.py` 这种特殊兼容层的行为：迁移后 `reward_harness/models.py` 应保持兼容 re-export 角色，`autosr/models.py` 则从 `reward_harness.models` re-export。
  - 确认所有文档已公开的 `reward_harness.*` 深层模块路径都在迁移清单中。

- [ ] **Task 1.3**: 补齐迁移前兼容缺口（若 Phase 2 不能同一变更立即落地）
  - 如果本计划分多次提交执行，先为当前已公开的深层路径新增临时 shim，避免 `reward_harness.search.config` 等路径在迁移窗口内不可导入。
  - 如果 Phase 2 在同一变更中立即完成，可跳过临时 shim，但必须在 commit 说明中写明该缺口由物理迁移直接关闭。

---

### Phase 2: 核心代码迁移与反向 shim（最关键，建议单独 commit）

- [ ] **Task 2.1**: 物理移动代码
  - 严格按 Task 1.1a 的模块清单执行，避免“一次性移动所有文件”覆盖目标文件。
  - 对普通业务模块使用 `git mv` 将 `autosr/` 下实现移动到 `reward_harness/` 下对应位置。
  - 不直接移动旧的 flat compatibility wrapper（例如 `autosr/rl/record_manifest.py`）到 `reward_harness/rl/record_manifest.py`；这些旧 wrapper 的目标是后续重建为 `autosr` shim。
  - **特殊处理 `autosr/rl/cli/`**：`reward_harness` 已采用扁平化 CLI 结构（`reward_harness/rl/record_manifest.py` 等）。因此：
    - 将 `autosr/rl/cli/*.py` 中的**真正实现**移动到 `reward_harness/rl/*.py`，覆盖 `reward_harness` 中现有的第一阶段 shim。
    - `autosr/rl/*.py` 中原有的 backward-compatible flat wrapper 不作为实现来源；迁移后在 `autosr/rl/*.py` 重建为指向 `reward_harness.rl.*` 的 shim。
    - 迁移后 `reward_harness/rl/` 下无 `cli/` 子目录（保持与第一阶段一致的扁平化公开接口）。
  - 保留其余子目录结构：`content_extraction/`、`harness/`、`llm_components/`、`prompts/`、`rl/verl/`、`rm/`、`run_records/`、`search/`。

- [ ] **Task 2.2**: 在 `reward_harness/` 上运行替换脚本
  - 运行 `scripts/migrate_internal_refs.py --path reward_harness/`。
  - 更新 logger 名称（如 `"autosr.search"` → `"reward_harness.search"`）。
  - 更新 docstring / help text 中的 CLI 示例。
  - **核心约束**：`reward_harness/` 内部必须**零 `autosr` 引用**（运行 `grep -r "autosr" reward_harness/ --include="*.py"` 应无结果）。

- [ ] **Task 2.3**: 重建 `autosr/` 兼容 shim
  - `autosr/__init__.py`：从 `reward_harness` re-export 公开对象。
  - `autosr/cli.py`：从 `reward_harness.cli` 导入 `main`。
  - 各子包 `__init__.py`：从 `reward_harness.xxx` re-export。
  - 所有旧深层公共模块路径都必须保留对应 shim，例如 `autosr.search.config`、`autosr.harness.session`、`autosr.rm.data_models`、`autosr.rl.registry`。
  - `autosr/rl/*.py`（backward-compatible CLI 入口点）：从 `reward_harness.rl.*` 导入 `main`。
  - `autosr/rl/cli/*.py`（因为真正实现已扁平化到 `reward_harness/rl/`）：从 `reward_harness.rl.*` 导入 `main`。
  - 其他顶层模块（`config.py`、`data_models.py`、`evaluator.py` 等）：简单 re-export。
  - `autosr/models.py`：从 `reward_harness.models` re-export（保持兼容路径）。
  - `autosr/mix_reward.py`、`autosr/mock_components.py`：从 `reward_harness` 对应模块 re-export（若 `reward_harness` 已新增这些 shim）。

- [ ] **Task 2.4**: 验证无循环依赖
  - `python -c "import autosr; import reward_harness"` 成功。
  - `python -c "import reward_harness.search.config; import reward_harness.harness.session; import reward_harness.rm.data_models; import reward_harness.rl.registry"` 成功。
  - `python -c "import autosr.search.config; import autosr.harness.session; import autosr.rm.data_models; import autosr.rl.registry"` 成功。
  - `python -m autosr.cli --help` 成功。
  - `python -m reward_harness.cli --help` 成功。
  - `python -m autosr.rl.record_manifest --help` 成功。
  - `python -m autosr.rl.cli.record_manifest --help` 成功。
  - `python -m reward_harness.rl.record_manifest --help` 成功。

---

### Phase 3: 测试与外围迁移

- [ ] **Task 3.1**: 迁移测试代码
  - 在 `tests/` 上运行替换脚本：`scripts/migrate_internal_refs.py --path tests/ --tests-only`。
  - 将 `from autosr.` 改为 `from reward_harness.`。
  - 将 `python -m autosr.rl.xxx` 等 subprocess 调用改为 `python -m reward_harness.rl.xxx`。
  - 将 `self.assertLogs("autosr.` 改为 `self.assertLogs("reward_harness.`。
  - **保留并扩展** `tests/test_reward_harness_compat.py` 中的兼容测试：
    - 验证 `autosr.*` 可用，且对象身份与 `reward_harness.*` 一致。
    - 验证文档公开的深层 `reward_harness.*` 模块路径可导入。
    - 验证 legacy flat CLI 路径（如 `autosr.rl.record_manifest`）与 legacy nested CLI 路径（如 `autosr.rl.cli.record_manifest`）继续可用。

- [ ] **Task 3.2**: 更新 `pyproject.toml`
  - `name = "reward_harness"`
  - `description = "Reward Harness — automated rubric search and reward model engineering"`
  - `src = ["autosr", "reward_harness", "tests", "scripts"]`（影响 ruff 与 mypy 的扫描范围；`autosr` 仍是受支持兼容包，不能从质量门禁中移除）

- [ ] **Task 3.3**: 更新脚本
  - `scripts/run_quality_checks.sh`：ruff check / format 目标路径同时包含 `autosr reward_harness tests scripts`。
  - `scripts/run_quality_checks.sh`：mypy 目标从 `autosr/mix_reward.py` 更新为迁移后的 canonical 文件，同时保留必要的 shim 覆盖。
  - `tests/test_static_tooling.py`：同步断言 `autosr` 与 `reward_harness` 都在 ruff / format / mypy 相关目标中。
  - 检查 `scripts/validate_docs.py` 是否有硬编码包名或路径引用。

- [ ] **Task 3.4**: 更新文档中的剩余引用
  - `README.md` / `README.zh.md`：
    - 更新 pyproject.toml 名称说明。
    - 明确 `autosr` 为 legacy 兼容入口（而非“迁移期内同时支持”）。
  - `AGENTS.md`：更新常用命令中的路径引用；更新“项目结构”中的包名说明。
  - `docs/ARCHITECTURE.md`、`DESIGN.md`：更新实现路径引用（非历史事实部分）。

---

### Phase 4: 质量门禁

- [ ] **Task 4.1**: 运行单元测试
  - `./scripts/run_tests_unit.sh`
  - 所有测试必须通过（包括 `test_reward_harness_compat`）。

- [ ] **Task 4.2**: 运行质量检查
  - `./scripts/run_quality_checks.sh`
  - ruff lint、format、mypy 通过。

- [ ] **Task 4.3**: 运行兼容性测试
  - `uv run python -m unittest tests.test_reward_harness_compat`
  - 验证 `autosr.data_models.Rubric is reward_harness.data_models.Rubric` 仍然成立。
  - 验证深层模块导入与 CLI 兼容路径均可用。

- [ ] **Task 4.4**: 运行文档校验
  - `uv run python scripts/validate_docs.py`

- [ ] **Task 4.5**: 运行集成测试（如有）
  - `./scripts/run_tests_integration.sh`

---

### Phase 5: 收尾

- [ ] **Task 5.1**: 确认 `reward_harness/` 内部零 `autosr` 残留
  - `grep -r "autosr" reward_harness/ --include="*.py"` 应无任何结果。

- [ ] **Task 5.2**: 确认 `autosr/` 为纯 shim
  - `autosr/` 下所有 `.py` 文件应只包含 `import` / `from ... import` / `__all__`，无业务逻辑、无类定义、无函数实现（`main()` 的调用除外）。

- [ ] **Task 5.3**: 更新 `docs/PLANS.md`
  - 在 Active 表中添加本计划，完成后移动到 `completed/`。

- [ ] **Task 5.4**: 编写变更日志
  - 记录迁移日期、影响范围、向后兼容保证。

---

## 验收标准

- [ ] `python -m reward_harness.cli --help` 成功。
- [ ] `python -m autosr.cli --help` 成功（兼容）。
- [ ] `python -m reward_harness.rl.record_manifest --help` 成功。
- [ ] `python -m autosr.rl.record_manifest --help` 成功（兼容）。
- [ ] `python -m autosr.rl.cli.record_manifest --help` 成功（兼容）。
- [ ] `reward_harness.search.config`、`reward_harness.harness.session`、`reward_harness.rm.data_models`、`reward_harness.rl.registry` 均可导入。
- [ ] `autosr.search.config`、`autosr.harness.session`、`autosr.rm.data_models`、`autosr.rl.registry` 均可导入（兼容）。
- [ ] 单元测试全部通过。
- [ ] ruff / mypy 通过。
- [ ] `reward_harness/` 内部无 `autosr` 引用（`grep` 验证）。
- [ ] `autosr/` 下无业务逻辑代码（纯 re-export）。
- [ ] `pyproject.toml` 的 `name` 为 `reward_harness`。
- [ ] `tests/test_reward_harness_compat.py` 通过（对象身份一致）。

---

## 风险与处理

| 风险 | 影响 | 处理 |
|------|------|------|
| `git mv` + 内容修改导致 git 无法追踪重命名 | 代码审查困难，`git blame` 断裂 | 分 commit：先纯 `git mv`，再内容替换； reviewer 使用 `--find-renames` |
| 遗漏的内部 import 导致 `ImportError` | 测试失败，运行时崩溃 | 替换脚本带 dry-run + 人工 review；Phase 2.4 用 `grep` 强制清零；运行全量测试 |
| 循环导入（`autosr` shim → `reward_harness` → `autosr`） | 启动失败 | **硬性约束**：`reward_harness/` 内部零 `autosr` 引用；用 `python -c "import autosr"` 验证 |
| `autosr/rl/cli/` vs `reward_harness/rl/` 结构差异 | CLI 入口路径混乱 | 保持 `reward_harness` 的扁平化结构；`autosr/rl/cli/` 作为 shim 指向扁平化位置 |
| 已公开的 `reward_harness.*` 深层路径不可导入 | 新用户按 README / ARCHITECTURE 使用时失败 | 迁移前补临时 shim，或在同一变更中完成物理迁移；兼容测试必须覆盖深层导入 |
| 质量门禁只检查旧包或只检查新包 | shim 或 canonical 实现出现 lint / format 漏检 | `autosr` 与 `reward_harness` 在兼容期都必须进入 ruff / format / mypy 目标 |
| 大 diff 导致 review 困难 | 遗漏错误 | 按 Phase 分 commit；每阶段 review 通过后再进行下一阶段 |
| `pyproject.toml` 改名导致已安装环境失效 | 本地开发环境需重新安装 | 在 AGENTS/README 中注明；使用 `uv` 时通常重新解析依赖 |

---

## Commit 策略

建议按以下顺序提交，每步均可独立 review 与回滚：

1. **`chore: add migration inventory and optional deep shims`** — 新增模块清单、迁移脚本；若 Phase 2 不同 commit 立即落地，则先补齐已公开的 `reward_harness.*` 深层 shim（Phase 1）。
2. **`feat: migrate core code from autosr to reward_harness`** — `git mv` + 替换 + 反向 shim（Phase 2）。
   - 这是最大的 commit，建议仅包含 `autosr/` → `reward_harness/` 的移动与 `autosr/` shim 重建。
   - 移动顺序以模块清单为准，尤其避免 `autosr/rl/*.py` flat wrapper 覆盖 `autosr/rl/cli/*.py` 的真实实现。
3. **`test: migrate test imports to reward_harness`** — `tests/` 替换（Phase 3.1）。
4. **`chore: update config and docs for reward_harness`** — `pyproject.toml`、`scripts/`、`docs/` 更新（Phase 3.2~3.4）。
5. **`docs: complete reward_harness migration plan`** — `PLANS.md` 更新（Phase 5）。

---

## 关联文档

- [Reward Harness Renaming 设计决策](../../design-docs/05-reward-harness-renaming.md)
- [第一阶段执行计划（已完成）](../completed/rename-to-reward-harness.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [DESIGN.md](../../DESIGN.md)

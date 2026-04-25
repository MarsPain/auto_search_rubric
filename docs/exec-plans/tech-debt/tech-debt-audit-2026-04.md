# AutoSR 技术债务全面审计报告

> **审计日期**: 2026-04-23  
> **审计范围**: `autosr/` 核心包全量代码 + `tests/` 测试套件 + `docs/` 文档体系  
> **审计方法**: 静态代码分析、架构契约审查、测试覆盖缺口扫描、文档一致性校验  
> **版本基准**: v1.1 (Stage D 已完成，Stage E–G 规划中)  

---

## 执行摘要

本次审计对 `autosr/` 下 78 个 Python 源文件、28 个测试文件、约 13,000 行生产代码进行了全面扫描。结论：**代码结构良好、分层清晰，但存在 5 项架构级风险、8 项代码级缺陷、7 项质量级缺口**。建议在进入 Stage E（Classifier RM 蒸馏）之前，集中一个 Sprint（约 2 周）完成"技术债务清偿"，否则上层模块将继承底层的不稳定性。

**关键数字：**

| 维度 | 数量 | 说明 |
|------|------|------|
| 🔴 架构级债务 | 5 | 可能在未来导致大规模重构 |
| 🟡 代码级债务 | 8 | 已知 bug、重复逻辑、隐蔽缺陷 |
| 🟢 质量级债务 | 7 | 测试缺口、工具缺失、文档不一致 |
| 合计 | 20 | 详见下文分级清单 |

### 清偿进度

截至 2026-04-25，已完成第一批低风险清偿：

- 2.1 `TrainingManifest.from_json` 重复声明
- 2.4 `registry.py` fallback 双重 I/O
- 2.5 `LineageView` 可变默认写法
- 2.8 `create_verifier_with_extraction` 兼容参数策略说明
- 3.6 文档状态同步

---

## 一、🔴 架构级债务（Architectural Debt）

### 1.1 Searcher step/checkpoint 协议缺失，导致 SearchSession 与 EvolutionaryRTDSearcher 紧耦合

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/harness/session.py` 第 335、508、554、557、562、570、579、593、600、607、623、630 行 |
| **描述** | `SearchSession` 直接调用 `EvolutionaryRTDSearcher` 的 10+ 个私有方法（`_init_global_state`, `_score_population`, `_log_generation_progress`, `_update_generation_bests`, `_handle_stagnation`, `_select_hard_prompts`, `_evolve_selected_prompts`, `_finalize_best_from_population`, `_collect_margin_improvement` 等）。同时，`autosr/interfaces.py` 当前只有 `Verifier`、`PreferenceJudge`、`RubricProposer`、`RubricInitializer` 四个协议，搜索算法本身缺少可 step/checkpoint 的正式协议约束。 |
| **影响** | 高。搜索算法的任何内部重构都会破坏 harness 的 checkpoint/resume 能力。当前 harness 无法透明支持新的搜索算法，违反了"底座不可回退"原则。 |
| **根因** | 缺少正式的 "SteppableSearcher" 协议。Harness 需要窥探 searcher 的内部状态来做 checkpoint，但没有定义公共接口来获取/恢复这些状态。 |
| **建议修复** | 1. 在 `autosr/interfaces.py` 中定义 `Searcher` + `SteppableSearcher` 两级 Protocol，暴露 `search()` / `get_algorithm_state()` / `restore_algorithm_state()` / `step()` / `is_finished()` 等公共契约；<br>2. 将 `EvolutionaryRTDSearcher` 的私有方法中需要被 harness 调用的部分提升为协议实现；<br>3. `SearchSession` 仅通过协议与 searcher 交互，并用类型测试保障新搜索算法满足 harness 需求。 |
| **估算** | 3–4 天（含测试改造） |
| **阻塞** | 是。不解决此问题，Stage E 中引入新的搜索变体（如 classifier-guided search）将极其困难。 |

---

### 1.2 Harness 的 step-wise 执行范围残缺

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/harness/session.py` 第 444–457 行 |
| **描述** | `run_step()` 仅在 `evolutionary + global_batch` 组合下有效；`prompt_local` 和 `iterative` 模式显式抛出 `NotImplementedError`。 |
| **影响** | 中。对于需要细粒度 prompt-level checkpoint 的长尾场景（大 prompt 集合、逐 prompt 恢复），当前底座无法支持。 |
| **根因** | 实现优先级排序：先保证 `global_batch` 闭环，再扩展其他模式。 |
| **建议修复** | 1. 为 `IterativeRTDSearcher` 和 `prompt_local` 搜索实现 step-wise 状态暴露；<br>2. 在 `SearchSession` 中统一通过协议调用，消除模式判断分支。 |
| **估算** | 2–3 天 |

---

### 1.3 原子写 JSON 逻辑三处重复

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/io_utils.py` (`save_rubrics`)、`autosr/rl/io.py` (`_save_json_payload`)、`autosr/rm/io.py` (`_save_json_payload`) |
| **描述** | 三个模块各自实现了"写入临时文件 → fsync → 原子重命名"的相同模式。 |
| **影响** | 中。重复代码导致维护成本上升；如果未来需要统一增加文件权限控制、备份策略或写入校验，需要改三处。 |
| **根因** | IO 工具类未在项目早期统一抽象；`rl/` 和 `rm/` 作为后续阶段独立开发，各自引入了局部工具。 |
| **建议修复** | 在 `autosr/io_utils.py` 中引入 `atomic_write_json(path: Path, data: Any)` 和 `atomic_write_text(path: Path, content: str)` 两个原子写原语；替换 `rl/io.py` 和 `rm/io.py` 中的重复实现。 |
| **估算** | 0.5 天 |

---

### 1.4 配置验证逻辑双重定义

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/config.py`（`SearchAlgorithmConfig`）、`autosr/search/config.py`（`EvolutionaryConfig`） |
| **描述** | `population_size`、`generations`、`tournament_size`、`diversity_weight` 等进化参数在两个层级的配置类中都有定义和验证逻辑，存在漂移风险。 |
| **影响** | 中。如果两处默认值或约束条件不同，会导致 CLI 层和算法层对同一配置的理解不一致。 |
| **根因** | `SearchAlgorithmConfig` 作为 RuntimeConfig 的子结构面向 CLI/序列化；`EvolutionaryConfig` 作为算法内部结构面向搜索逻辑。两者没有明确的单向转换契约。 |
| **建议修复** | 1. 明确 "CLI 配置 → 算法配置" 的单向转换契约（现有 `SearchAlgorithmConfig.to_evolutionary_kwargs()` 可作为入口，必要时补充 `EvolutionaryConfig.from_runtime(config.search)`）；<br>2. 保留 `EvolutionaryConfig` 作为算法层可独立使用的配置对象，但通过测试锁定它与 `SearchAlgorithmConfig` 的默认值、枚举转换和约束一致性；<br>3. 将重复校验规则抽成共享 helper 或集中测试，避免两层配置静默漂移。 |
| **估算** | 1 天 |

---

### 1.5 `CheckpointCallback` 类型别名重复

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/search/evolutionary.py` 第 32 行、`autosr/factory.py` 第 43 行 |
| **描述** | 完全相同的 `Callable[[dict[str, Rubric], dict[str, float], dict[str, list[float]]], None]` 类型别名在两处独立定义。 |
| **影响** | 低。但它是架构层"类型契约分散"的症状——如果回调签名需要扩展（例如增加 generation 序号），需要改两处。 |
| **根因** | 缺少统一的回调/事件协议文件。 |
| **建议修复** | 移动到 `autosr/interfaces.py` 或 `autosr/types.py`，两处统一引用。 |
| **估算** | 0.25 天 |

---

## 二、🟡 代码级债务（Code-Level Debt）

### 2.1 `TrainingManifest.from_json` 被声明两次（真实 Bug）

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/rl/data_models.py` 第 117–131 行 |
| **描述** | 第 117 行的 `from_json` 返回类型标注为 `"LineageIndex"`，且 schema 参数名也不一致；第 125 行才是正确的 `TrainingManifest` 版本。后者在运行时完全 shadow 前者。 |
| **影响** | 中。如果外部代码通过 `TrainingManifest.from_json.__annotations__` 做反射，或在静态类型检查时使用第一个签名，会产生误导。虽然运行时行为正确，但这是 schema 契约层面的噪音。 |
| **根因** | 复制粘贴 LineageIndex 的 from_json 方法后未完全修改。 |
| **建议修复** | 删除第 117–123 行的错误声明。 |
| **估算** | 5 分钟 |

---

### 2.2 `_config_to_dict` 手动选取字段导致哈希碰撞风险

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/harness/session.py` 第 765–799 行 |
| **描述** | `_config_to_dict` 手工枚举了 `RuntimeConfig` 的子结构用于计算 `config_hash`。虽然 `candidate_extraction` 已经被加入，但如果未来 `RuntimeConfig` 新增字段（如 `metrics`、`callbacks`），这些字段不会被纳入哈希，导致"配置不同但哈希相同"的碰撞。 |
| **影响** | 中。config hash 是 resume 安全性的关键契约。碰撞会导致"恢复时配置已变但系统未察觉"，进而产生不可复现的搜索结果。 |
| **根因** | 手工白名单模式在配置演进时天然脆弱。 |
| **建议修复** | 1. 使用 `dataclasses.asdict(config)` 做全量递归转换（配合 Enum/Path 的序列化处理）；<br>2. 对敏感字段（如 `api_key`）显式排除，而非对普通字段显式包含；<br>3. 增加单元测试：每次 `RuntimeConfig` 结构变更时自动验证哈希覆盖度。 |
| **估算** | 0.5 天 |

---

### 2.3 `rm/use_cases.py` 内联硬编码 LLM 默认值

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/rm/use_cases.py` 第 77–106 行 |
| **描述** | `_build_runtime_snapshot` 中内联写死了 `base_url`、`timeout`、`max_retries`、`default_model` 等默认值，与 `autosr/config.py` 中的 `LLMBackendConfig` 默认值重复。如果 `config.py` 的默认值调整，artifact 中的 `runtime_snapshot` 可能 silently diverge。 |
| **影响** | 中。RM Artifact 的核心价值是"可追溯、可复现"，但硬编码默认值破坏了"运行时真实配置 == 快照配置"的一致性。 |
| **根因** | `runtime_snapshot` 构建逻辑与配置类之间缺少单向引用。 |
| **建议修复** | 从 `LLMBackendConfig` 的类属性或 dataclass 默认值中读取，禁止在 `use_cases.py` 中内联重复定义。 |
| **估算** | 0.5 天 |

---

### 2.4 `registry.py` 双重 I/O（性能与一致性隐患）

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/rl/registry.py` 第 219–223 行 |
| **描述** | `list_evals_for_training_run` 的 fallback scan 循环中，对同一文件路径两次调用 `self.get_eval(p.stem)`。 |
| **影响** | 低–中。在高 eval 量场景下浪费 I/O；更关键的是两次读取之间文件可能被外部修改，导致判断不一致（TOCTOU）。 |
| **根因** | 代码简洁性不足，未将第一次读取结果缓存到局部变量。 |
| **建议修复** | 改为单次读取并缓存：`eval_data = self.get_eval(p.stem); if eval_data is not None and eval_data.training_run_id == training_run_id: ...` |
| **估算** | 10 分钟 |

---

### 2.5 `LineageView` 使用 `None` 可变默认并依赖 `__post_init__`

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/rl/lineage.py` 第 23–24 行 |
| **描述** | `eval_benchmarks: list[str] = None` 和 `upstream_chain: dict[str, str] = None` 标注了 `# type: ignore[assignment]`，依靠 `__post_init__` 在实例化后修正。这不是 dataclass 的惯用写法，且对静态类型检查不友好。 |
| **影响** | 低。当前功能正常，但属于 Python dataclass 反模式；如果未来移除 `slots=True` 或改用 `pydantic`，可能暴露隐患。 |
| **根因** | 对 dataclass 可变默认值的处理不熟悉。 |
| **建议修复** | 使用 `field(default_factory=list)` 和 `field(default_factory=dict)`，移除 `# type: ignore`。 |
| **估算** | 10 分钟 |

---

### 2.6 广泛的 `except Exception` 捕获可能掩盖真实 Bug

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/harness/session.py` 第 713 行（`_save_checkpoint`）、`autosr/harness/storage.py` 第 126、186、251 行 |
| **描述** | `_save_checkpoint` 捕获所有异常后只记录 error 并返回 `None`，会让调用方继续执行，形成 checkpoint 静默失败风险。`storage.py` 中的 broad catch 语义不同：`save_checkpoint` 用于清理临时文件后重新抛出，`load_checkpoint` 会 rewrap 为 `CheckpointCorruptedError`，`list_checkpoints` 会跳过坏 checkpoint 元数据；这些需要明确哪些是预期容错，哪些应 loud fail。 |
| **影响** | 中。在长线搜索任务中，checkpoint 保存失败如果被吞掉，用户会以为状态已保存，实际恢复时可能丢失大量进度。读取/list 阶段的容错则需要文档化，避免被误改成过度严格。 |
| **根因** | 保存失败、读取损坏、列表元数据容错三类异常策略没有清晰分层。 |
| **建议修复** | 1. 定义 `CheckpointIOError` 等细分异常；<br>2. `_save_checkpoint` 对 `OSError`（磁盘、权限）和序列化错误（`TypeError`、`ValueError`）loud fail（记录 ERROR 并抛出）；<br>3. 保留 `load_checkpoint` 的腐坏 checkpoint rewrap 行为，但缩窄捕获范围；<br>4. 明确 `list_checkpoints` 跳过坏元数据是有意容错，并用测试覆盖。 |
| **估算** | 0.5 天 |

---

### 2.7 选择策略使用 `id()` 进行去重

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/search/selection_strategies.py` 第 64、73、81–82 行 |
| **描述** | `select_parents_tournament` 使用 `id(rubric)` 作为去重键。虽然当前 deepcopy 后的对象 `id()` 仍然唯一，但如果未来引入 rubric 缓存池或 flyweight 模式，语义上的去重应基于内容指纹。 |
| **影响** | 低。当前无实际 bug，但属于语义不严谨的实现。 |
| **根因** | 便捷性优先于语义正确性。 |
| **建议修复** | 将 `id(rubric)` 替换为 `rubric.fingerprint()` 或 `_fingerprint(rubric)`（`search/strategies.py` 中已有 `_fingerprint` 函数可复用）。 |
| **估算** | 0.25 天 |

---

### 2.8 `create_verifier_with_extraction` 丢弃 `prompts` 参数

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/factory.py` 第 206–232 行 |
| **描述** | `ComponentFactory.create_verifier_with_extraction(self, prompts)` 接收 `prompts` 参数但立即用 `_ = prompts` 丢弃，注释说明是"API compatibility"。 |
| **影响** | 低。属于接口污染——调用方传递了数据集，但工厂不使用，增加了认知负担。不过该参数是兼容层，直接删除会影响 `cli.py`、`rm/server.py`、`create_components_for_dataset`、测试 spy 以及潜在外部扩展。 |
| **根因** | 早期某版本曾根据 prompts 内容动态决定提取策略，后来移除但保留了参数。 |
| **建议修复** | 不要立即硬删除。短期保留可选参数并在 docstring 中明确"兼容保留、当前未使用"；若决定收敛 API，则先发出 `DeprecationWarning`，同步修改 `cli.py`、`rm/server.py`、`factory.py` helper 与测试，下一阶段再删除。 |
| **估算** | 0.25–0.5 天 |

---

## 三、🟢 质量级债务（Quality Debt）

### 3.1 `selection_strategies.py` 零测试覆盖

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/search/selection_strategies.py`（240 行） |
| **描述** | 实现了 rank、tournament、top-k-diverse 三种父代选择策略，但没有任何直接单元测试。当前仅通过 `test_search.py` 中的端到端搜索间接覆盖。 |
| **影响** | 高。选择策略是进化搜索的核心决策逻辑；其正确性直接影响收敛速度和多样性保持。零覆盖意味着边界条件（如种群大小 < tournament_size、全同分数、极端 diversity_weight）无保护。 |
| **建议修复** | 新增 `tests/test_selection_strategies.py`，覆盖：三种策略的基础选择、tournament 概率边界、diversity 筛选逻辑、空种群/单个体退化场景。 |
| **估算** | 1 天 |

---

### 3.2 `mix_reward.py` 零测试覆盖

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/mix_reward.py`（26 行） |
| **描述** | `blended_reward()` 和 `EtaScheduler.value()` 是纯函数/数值逻辑，无任何测试。 |
| **影响** | 中。该模块是 Stage D 后引入的 reward blending 基础，未来 Classifier RM 蒸馏和 RM 混合打分都会依赖它。无回归保护。 |
| **建议修复** | 新增 `tests/test_mix_reward.py`，覆盖 eta clamp、warmup/ramp 边界、progress 计算精度。 |
| **估算** | 0.5 天 |

---

### 3.3 `adaptive_mutation.py` 缺少直接单元测试

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/search/adaptive_mutation.py`（457 行） |
| **描述** | 四种 mutation schedule（fixed、success_feedback、exploration_decay、diversity_driven）和 diversity metric 仅通过 `test_evolutionary_decoupling.py` 间接引用，缺少对 scheduler 状态转换、success feedback 权重更新、diversity 计算公式的直接测试。 |
| **影响** | 中。自适应变异是进化搜索区别于简单搜索的关键，其调度逻辑的正确性决定搜索效率。 |
| **建议修复** | 新增 `tests/test_adaptive_mutation.py`，覆盖各 scheduler 的 `get_state` / `from_state` 一致性、success feedback 权重变化方向、exploration decay 的单调性。 |
| **估算** | 1 天 |

---

### 3.4 缺少静态代码质量工具链

| 属性 | 内容 |
|------|------|
| **位置** | `pyproject.toml`、项目根目录 |
| **描述** | 项目未配置 ruff、black、mypy、pre-commit 等任何自动质量工具。`pyproject.toml` 中没有任何 `[tool.ruff]`、`[tool.mypy]` 配置节。 |
| **影响** | 中高。随着代码量增长（当前 `autosr/` 下 78 个 Python 源文件，全仓库 109 个 Python 文件），人工 code review 无法覆盖所有类型不一致、未使用变量、导入循环、格式问题。Stage E 计划引入更多模块，工具链缺失将显著降低开发效率。 |
| **建议修复** | 1. 引入 `ruff` 作为 linter + formatter（替代 flake8 + black，配置简单）；<br>2. 引入 `mypy` 做静态类型检查（代码已有 extensive type hints，收益高）；<br>3. 配置 `pre-commit` hook 在提交前自动运行；<br>4. 首次启用时可能会暴露大量既有 warning，建议分阶段修复（先配置，再分期清偿既有 debt）。 |
| **估算** | 1 天（配置）+ 2 天（修复既有 warning，可分期） |

---

### 3.5 缺少 CI/CD 配置

| 属性 | 内容 |
|------|------|
| **位置** | 项目根目录（无 `.github/workflows/`） |
| **描述** | 没有 GitHub Actions 或其他 CI 配置。测试和文档校验依赖开发者本地手动执行。 |
| **影响** | 中。在多人协作或阶段性合并时，无法保证主干代码始终通过测试；也无法在 PR 阶段拦截类型错误或测试失败。 |
| **建议修复** | 新增 `.github/workflows/ci.yml`，包含：单元测试（`run_tests_unit.sh`）、文档校验（`validate_docs.py`）、ruff 检查、mypy 检查。 |
| **估算** | 0.5 天 |

---

### 3.6 文档状态部分不一致

| 属性 | 内容 |
|------|------|
| **位置** | `docs/ROADMAP.md` 第 213 行、`docs/ARCHITECTURE.md` 第 125 行 |
| **描述** | `PLANS.md` 已将 Stage D 标注为完成，`stage-d3-comparative-view.md` 也已经位于 `exec-plans/completed/`；但 `ROADMAP.md` 的优先级矩阵仍把"RL 训练接入与实验编排"标为"🚧 设计完成，待实现"，`ARCHITECTURE.md` 仍将 "RL / Classifier RM / Monitoring / Closed Loop" 整体描述为 Stage D+ 路线图内容。 |
| **影响** | 低。对开发者造成认知混乱，新人可能误以为 Stage D 尚未实现，或误判当前 domain map 中 `autosr/rl/` 的架构地位。 |
| **建议修复** | 1. 更新 `ROADMAP.md` 优先级矩阵中 RL 训练接入状态为 ✅ 已完成；<br>2. 更新 `ARCHITECTURE.md` 当前阶段基线，将 Stage D / `autosr/rl/` 明确纳入已完成域，并仅保留 Stage E–G 为规划中；<br>3. 如修改架构域定义，同步补充 `autosr/rl/` 的核心包列表。 |
| **估算** | 15 分钟 |

---

### 3.7 `models.py` 遗留兼容层弃用策略不完整

| 属性 | 内容 |
|------|------|
| **位置** | `autosr/models.py` |
| **描述** | `llm_components/factory.py` 的 `create_llm_components` 函数已经在运行时发出 `DeprecationWarning`，但 `models.py` 只是 `data_models.py` 的 re-export shim，尚未有明确弃用策略或保留期限。 |
| **影响** | 低。新开发者可能误用 legacy 入口，导致代码分散在两条实例化路径上。 |
| **建议修复** | 1. 决定 `autosr.models` 是长期兼容入口还是限期弃用 shim；<br>2. 若限期弃用，在模块导入或相关对象 re-export 处添加清晰 warning / deprecated 注解，并补测试避免 warnings 破坏现有兼容测试；<br>3. 在 `DESIGN.md` 或兼容性说明中记录弃用时间表。 |
| **估算** | 0.25–0.5 天 |

---

## 四、汇总与执行建议

### 4.1 按 Sprint 分组的清偿计划

建议将技术债务清偿作为一个**独立 Sprint（Tech Debt Sprint）**插入 Stage D 与 Stage E 之间，预计 **2 周**。

#### Week 1：架构与代码修复（高风险优先）

| 天 | 任务 | 对应债务项 |
|----|------|-----------|
| 1 | 修复 `TrainingManifest.from_json` 重复声明 + `LineageView` 可变默认 + `registry.py` 双重 I/O + `create_verifier_with_extraction` 兼容策略 | 2.1, 2.4, 2.5, 2.8 |
| 2 | 统一原子写 JSON 原语 + 修复 `CheckpointCallback` 重复 + 选择策略 `id()` → fingerprint | 1.3, 1.5, 2.7 |
| 3 | 消除 `rm/use_cases.py` 硬编码默认值 + 修复 `_config_to_dict` 哈希碰撞风险 + 细化 `except Exception` | 2.2, 2.3, 2.6 |
| 4–5 | 设计并实现 `SteppableSearcher` 协议，解耦 `SearchSession` 与 `EvolutionaryRTDSearcher` | 1.1 |

#### Week 2：测试补齐与工具链建设

| 天 | 任务 | 对应债务项 |
|----|------|-----------|
| 6 | 补齐 `selection_strategies.py` 测试 + `mix_reward.py` 测试 | 3.1, 3.2 |
| 7 | 补齐 `adaptive_mutation.py` 测试 | 3.3 |
| 8 | 配置 ruff + mypy，修复首批 critical warning | 3.4 |
| 9 | 配置 GitHub Actions CI + 修复剩余 type/lint issues | 3.5, 3.4（续） |
| 10 | 视 Stage E 依赖决定是否补齐 `prompt_local` / `iterative` 的 step-wise 执行；同步文档状态与兼容层弃用策略 | 1.2, 3.6, 3.7 |

### 4.2 不可跳过的阻塞项

以下债务**必须在进入 Stage E 之前解决**，否则上层建筑将继承不稳定性：

- **1.1 Searcher step/checkpoint 协议缺失** — Stage E 可能引入新的搜索策略（如 classifier-guided mutation），没有协议层无法集成。
- **2.2 `_config_to_dict` 哈希碰撞** — 任何新增配置字段都会破坏 resume 安全性。
- **2.6 广泛 `except Exception`** — 长线搜索任务（Stage E 的反复蒸馏实验）对 checkpoint 可靠性要求更高。
- **3.4 静态工具链** — 在代码规模突破 100 个文件之前建立工具链，成本远低于后期补课。

### 4.3 可延后的非阻塞项

以下债务可以延后到 Stage E 开发过程中顺手修复：

- 1.2 `prompt_local` / `iterative` step-wise（除非 Stage E 明确需要这些搜索作用域）
- 2.7 `id()` 去重（当前无实际 bug）
- 3.6 文档状态同步（纯文档维护）

---

## 五、审计方法说明

本次审计综合使用了以下手段，确保结论可复现：

1. **代码扫描**：`grep -r` 搜索 TODO/FIXME/HACK/XXX/deprecated/bare-except/hardcoded-secrets 等模式，并对结果做人工复核。
2. **测试覆盖映射**：将 28 个测试文件与 `autosr/` 下 78 个源文件逐一对照，识别零覆盖模块。
3. **架构契约审查**：对照 `docs/DESIGN.md` 中的分层依赖规则，检查 `cli.py` → `factory.py` → `use_cases.py` → domain 的调用链路是否合规。
4. **数据流审查**：追踪 `SearchCheckpoint` 和 `RMArtifact` 两个核心契约的生成、序列化、校验全链路。
5. **静态配置审计**：检查 `pyproject.toml`、脚本、CI 配置。

---

*本报告为活文档。每完成一项债务清偿，应在此文件中标注完成状态，并更新 `docs/PLANS.md` 的技术债表格。*

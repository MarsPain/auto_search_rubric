# Reward Harness 架构总览地图

> **定位**: 顶层架构入口（稳定）  
> **版本**: v1.1 | **状态**: 稳定 | **最后更新**: 2026-04-17

---

## 1. 文档职责边界

- 本文档回答：系统由哪些层组成、核心域如何划分、包依赖方向是什么。
- [DESIGN.md](DESIGN.md) 回答：为什么这样设计、阶段目标与契约演进。
- [design-docs/01-architecture.md](design-docs/01-architecture.md) 回答：详细演进图与阶段落地细节。

原则：`ARCHITECTURE.md` 保持稳定、短小、可快速扫读；演进细节下沉到 design-docs。

---

## 2. 顶层架构地图

```text
CLI / Integration Boundary
  -> Component Assembly (factory + runtime config)
    -> Search Harness (session / checkpoint / resume via searcher protocols)
      -> Search Domain (iterative / evolutionary strategies)
        -> LLM & Content Extraction Adapters
          -> Evaluation & Run Records
            -> RM Artifact Domain (build / validate / export)
              -> Artifacts & Checkpoints Storage
```

---

## 3. 核心域定义（Domain Map）

### 3.1 Harness 域（执行底座）

- 目标：让搜索过程可长时运行、可恢复、可追溯。
- 核心包：
  - `reward_harness/harness/session.py`（legacy 兼容：`autosr/harness/session.py`）
  - `reward_harness/harness/state.py`（legacy 兼容：`autosr/harness/state.py`）
  - `reward_harness/harness/storage.py`（legacy 兼容：`autosr/harness/storage.py`）
- 关键边界：Harness 只通过 `Searcher` / `SteppableSearcher` 公共协议驱动算法，不调用具体搜索器私有方法。

### 3.2 Search 域（算法层）

- 目标：提供 iterative / evolutionary 等搜索策略与调度能力。
- 核心包：
  - `reward_harness/search/use_cases.py`
  - `reward_harness/search/iterative.py`
  - `reward_harness/search/evolutionary.py`
  - `reward_harness/search/strategies.py`

### 3.3 LLM 与内容抽取适配域

- 目标：封装模型调用、提示词加载、文本抽取策略，隔离外部模型差异。
- 核心包：
  - `reward_harness/llm_components/`
  - `reward_harness/llm_client.py`
  - `reward_harness/prompts/`
  - `reward_harness/content_extraction/`

### 3.4 评估与记录域

- 目标：评估 rubric 质量并持久化 run records。
- 核心包：
  - `reward_harness/evaluator.py`
  - `reward_harness/run_records/use_cases.py`

### 3.5 RM Artifact 域（阶段 B）

- 目标：将搜索结果升级为可部署、可追溯的 RM 工件。
- 核心包：
  - `reward_harness/rm/data_models.py`
  - `reward_harness/rm/use_cases.py`
  - `reward_harness/rm/export.py`
  - `reward_harness/rm/io.py`

### 3.6 RL 接入与实验记录域（阶段 D）

- 目标：维护 RL 训练接入契约、append-only registry、lineage 查询与比较视图。
- 核心包：
  - `reward_harness/rl/data_models.py`
  - `reward_harness/rl/registry.py`
  - `reward_harness/rl/lineage.py`
  - `reward_harness/rl/comparison.py`
  - `reward_harness/rl/verl/`

### 3.7 共享模型与类型域

- 目标：统一领域实体、枚举和跨模块契约，降低耦合。
- 核心包：
  - `reward_harness/data_models.py`
  - `reward_harness/types.py`
  - `reward_harness/config.py`
  - `reward_harness/interfaces.py`

---

## 4. 包分层与依赖方向

```text
Entry Layer:
  reward_harness/cli.py (legacy compatible: autosr/cli.py)
    -> Composition Layer:
         reward_harness/factory.py
         reward_harness/config.py
    -> Use-Case Layer:
         reward_harness/*/use_cases.py
    -> Domain Layer:
         reward_harness/search/*
         reward_harness/harness/*
         reward_harness/rm/*
         reward_harness/data_models.py
         reward_harness/types.py
    -> Infra / Adapter Layer:
         reward_harness/llm_client.py
         reward_harness/io_utils.py
         reward_harness/content_extraction/*
         reward_harness/run_records/*
```

依赖规则：

1. `cli.py` 只编排，不承载业务规则。
2. `factory.py` 负责组装依赖，不写领域决策。
3. `use_cases.py` 可依赖 domain 与 adapter，但 domain 不反向依赖 use case。
4. `data_models.py` 与 `types.py` 为跨域共享契约，避免循环依赖。
5. `rm/` 依赖 search 输出契约，但不反向侵入 search 实现。

---

## 5. 当前阶段架构基线

- 阶段 A（Harness 底座）已完成并作为不可回退基线。
- 阶段 B（RM Artifact）已完成：schema、导出、校验、deploy manifest。
- 阶段 C（RM Server MVP）已完成：服务进程、评分API、artifact运行时加载、请求日志。
- 阶段 D（RL 训练接入与实验编排）已完成：训练 manifest/result/eval registry、lineage 查询、比较视图与回归检测。
- 阶段 E–G（Classifier RM / Monitoring / Closed Loop）仍在路线图中。

参考：
- [ROADMAP.md](ROADMAP.md)
- [PLANS.md](PLANS.md)
- [design-docs/01-architecture.md](design-docs/01-architecture.md)

---

## 6. 变更准则

发生以下变更时，必须同步更新本文档与相关设计文档：

- 新增或重构核心域边界（例如新增 `reward_harness/rl/`）。
- 调整包层级或依赖方向规则。
- 引入新的顶层运行链路（例如 server/runtime plane）。

最小同步清单：

1. 更新 `ARCHITECTURE.md` 顶层地图与域定义。
2. 更新 `DESIGN.md` 的目标架构或契约章节。
3. 必要时新增/更新 `docs/design-docs/` 详细设计。

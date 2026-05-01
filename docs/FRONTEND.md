# Reward Harness Frontend Scope

> **版本**: v1.1 | **状态**: 受限范围 | **最后更新**: 2026-05-01

---

## 当前范围

Reward Harness 当前以后端 CLI / 服务化能力为主，**没有独立前端应用**。

前端相关内容仅包括：
- 文档与可视化输出规范（如运行结果、监控报表）
- 未来 RM/RL 监控看板的信息架构约束

---

## 设计约束

1. 任何前端页面必须消费已版本化的后端契约（artifact / manifest / metrics）。
2. 不允许前端绕过后端契约读取内部中间状态文件。
3. 新增前端功能时，先在 `docs/design-docs/` 增加设计文档，再落代码。
4. 监控看板的指标定义必须与 `docs/PRODUCT_SENSE.md` 的北极星指标保持一致。

---

## 近期计划

- 阶段 C-D 期间补充 `docs/product-specs/` 下的 RM 监控与训练可视化规格。
- 阶段 E 启动时，新增专门的前端设计文档（如 `design-docs/02-monitoring-ui.md`）。

---

## 关联文档

- [DESIGN.md](DESIGN.md)
- [ROADMAP.md](ROADMAP.md)
- [PRODUCT_SENSE.md](PRODUCT_SENSE.md)

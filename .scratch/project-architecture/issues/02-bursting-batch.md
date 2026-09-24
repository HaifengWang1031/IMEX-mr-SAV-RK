# 02: Bursting：固定／自适应对照批次

**What to build:** 用一份批次配置执行多个独立 bursting 算例，记录成员运行编号、兼容结果复用以及单项失败和重试历史。

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] 配置集中说明共同参数与各算例差异，展开后保存完整生效配置及稳定 case-id。
- [x] 批次记录各成员 pending/running/completed/failed 状态、run-id、复用情况和日志；单项失败不抹去其他成员结果。
- [x] 已有兼容完成结果可复用；显式重试指定失败项创建新运行尝试，保留此前编号，不宣称断点续算。
- [x] 小规模固定／自适应对照可完成、重载和复用；注入单项失败验证隔离与定向重试，提供使用说明。

## Comments

2026-09-24：按用户“全仓库快速迁移、smoke 即可”指令统一推进。七主题新入口与通用分析已可运行，验收见 docs/migration-20260924.md。已用短程数据检查对应运行、记录或分析路径。

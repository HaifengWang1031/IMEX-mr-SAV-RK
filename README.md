# IMEX-mr-SAV-RK

周期 Navier–Stokes 涡量–流函数 Fourier 拟谱实验。正式计算统一使用新求解器，参数集中设置；计算、批次和分析分别记录。

- [实验使用说明](experiments/README.md)：七类实验、配置、运行、批次、重试和图表。
- [新求解器](solver/README.md)：每个具体格式独立文件，统一推进及可扩展自适应策略。
- [项目架构](docs/project-architecture.md)与[迁移记录](docs/migration-20260924.md)。
- [旧文件归档](archive/legacy/migration.json)：原路径、归档路径和 SHA256；历史大数据原位保留。

```bash
PY=/Users/wanghaifeng/miniconda3/envs/pde/bin/python
$PY -m experiments.bursting.run
$PY -m experiments.bursting.analyze --runs /这里填写上一步打印的运行目录
```

默认配置是短程 smoke。正式研究先复制并修改 JSON；分析明确选择已完成运行，不触发计算。

| 目录 | 内容 |
|---|---|
| solver | 正式求解器 |
| experiments | 实验代码、配置、交互 notebook |
| runs | 每次计算的有效配置、结果、日志；批次成员清单 |
| reports | 每次分析的输入、配置、图表、日志 |
| tests、tools | 检查与仓库维护 |
| docs、CONTEXT.md | 架构与领域约定 |
| archive/legacy | 原始脚本、notebook 与旧局部工作流 |
| .scratch | 本地迁移任务票 |

旧求解器已移除，对照测试使用冻结的数值基准。根目录旧入口已归档，旧命令请改用上述模块入口。

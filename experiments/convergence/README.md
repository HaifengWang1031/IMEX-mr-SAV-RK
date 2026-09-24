# 时间收敛与给定变步长：显式 ETDRK4 参考运行

本入口使用独立新求解器。默认配置是短程 smoke，不代表论文参数或收敛结论。

```bash
python -m experiments.convergence.run --show-config
python -m experiments.convergence.run
python -m experiments.convergence.run --batch experiments/convergence/configs/batch.json
python -m experiments.convergence.analyze --runs /absolute/path/to/run
```

每次运行打印结果目录；分析显式选取该目录，批次分析使用 `--batch /absolute/path/to/batch`。
详细参数、参考解与统计配方说明见 [工作流说明](../README.md)。
旧入口及原始参数原样保存在 archive/legacy，原数据仍在原位置。

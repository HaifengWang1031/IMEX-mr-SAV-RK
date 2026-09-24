# 受迫 SDIRK2 普通／SAV 误差与稳定性对照

本入口使用独立新求解器。默认配置是短程 smoke，不代表论文参数或收敛结论。

```bash
python -m experiments.sdirk2_transition.run --show-config
python -m experiments.sdirk2_transition.run
python -m experiments.sdirk2_transition.run --batch experiments/sdirk2_transition/configs/batch.json
python -m experiments.sdirk2_transition.analyze --runs /absolute/path/to/run
```

每次运行打印结果目录；分析显式选取该目录，批次分析使用 `--batch /absolute/path/to/batch`。
详细参数、参考解与统计配方说明见 [工作流说明](../README.md)。
旧入口及原始参数原样保存在 archive/legacy，原数据仍在原位置。

# 双剪切层演化

本入口使用独立新求解器。默认配置是短程 smoke，不代表论文参数或收敛结论。

```bash
python -m experiments.shearflow.run --show-config
python -m experiments.shearflow.run
python -m experiments.shearflow.run --batch experiments/shearflow/configs/batch.json
python -m experiments.shearflow.analyze --runs /absolute/path/to/run
```

每次运行打印结果目录；分析显式选取该目录，批次分析使用 `--batch /absolute/path/to/batch`。
详细参数、参考解与统计配方说明见 [工作流说明](../README.md)。
旧入口及原始参数原样保存在 archive/legacy，原数据仍在原位置。

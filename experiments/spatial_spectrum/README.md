# 同一连续 Fourier 初值的空间分辨率对照

本入口使用独立新求解器。默认配置是短程 smoke，不代表论文参数或收敛结论。

```bash
python -m experiments.spatial_spectrum.run --show-config
python -m experiments.spatial_spectrum.run
python -m experiments.spatial_spectrum.run --batch experiments/spatial_spectrum/configs/batch.json
python -m experiments.spatial_spectrum.analyze --runs /absolute/path/to/run
```

每次运行打印结果目录；分析显式选取该目录，批次分析使用 `--batch /absolute/path/to/batch`。
详细参数、参考解与统计配方说明见 [工作流说明](../README.md)。
旧入口及原始参数原样保存在 archive/legacy，原数据仍在原位置。

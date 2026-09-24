# 数值实验工作流

所有正式入口统一调用 `solver`。每个实验有集中 JSON 配置、运行与分析命令及只读交互 notebook。配置默认是 smoke 规模；论文参数保存在原文件的归档副本中，正式计算前显式调整配置。此次迁移验收不包括长时间统计或完整精度研究。

## 执行

从仓库根目录使用 PDE 环境：

```bash
PY=/Users/wanghaifeng/miniconda3/envs/pde/bin/python
$PY -m experiments.bursting.run --show-config
$PY -m experiments.bursting.run
$PY -m experiments.bursting.run --config my-config.json
$PY -m experiments.bursting.run --batch experiments/bursting/configs/batch.json
```

将 `bursting` 换成 `convergence`、`mean_reverting`、`adaptive_tolerance`、`shearflow`、`spatial_spectrum` 或 `sdirk2_transition` 即使用对应实验。覆写配置递归合并已知键，未知参数报错。`--output-root` 指定 runs 的父根，每次打印具体运行或批次目录。相同配置、源码与环境的完成结果默认复用；`--rerun` 保留旧结果并创建新尝试。

失败项重试：

```bash
$PY -m experiments.bursting.run --retry-batch /absolute/batch/directory --cases failed-case-id
```

批次保留成员的所有尝试路径；只重试指定失败或未完成成员。失败记录保留错误、日志及可获得的积分前缀。重试从初值开始，不是断点续算。

## 参数

| 参数 | 含义 |
|---|---|
| N、threads、nu、domain | 方形空间网格、FFTW 线程、黏性、矩形区域；domain 顺序 xa,ya,xb,yb |
| forcing | 涡量外力 none/cos_x/cos_y，及幅值、波数 |
| initial | trig/random/isotropic 的模态、RMS、随机种子；bursting 的 eps；shear 的 rho、delta |
| scheme、gamma | 具体格式名及 SAV 回归参数 |
| root_selection | 当前只支持 legacy；其他值报错，不默换选根规则 |
| talbot_nodes | ETD-mrSAV-MS2-L 的 Talbot 节点数，旧实现默认 10 |
| mode、T、dt、steps | fixed/prescribed/adaptive，终止时间、固定步长、完整给定序列 |
| snapshots | 请求时刻；固定/给定序列严格落在网格，自适应选择最近接受节点 |
| warmup_time、warmup_dt | ETDRK4 预热长度及步长，实际长度向上取整；主积分重新从零计时 |
| adaptive | algorithm、atol/rtol、q_tolerance、exponent、safety、max_growth、initial_dt、min_dt/max_dt |
| max_steps、log_interval | 步数保护上限（null 可取消）、程序日志间隔秒数 |

普通算法不创建虚设 q。支持 `sdirk2`、`sdirk2_mrsav`、`etdrk4`、`etdms2`、`etd_mrsav_ms2_b`、`etd_mrsav_ms2_l`、`mrsav_bdf2`、`imex_euler`、`legacy_linear_etd`。

`legacy_linear_etd` 忠实保留旧 ETD 的线性扩散加外力公式，**没有 NS 对流项**，因此明确使用这个名字。缺失源码的 `ETD_mrGSAV_MS2_b` 不作为现有 mrSAV 的别名；剪切层的新配置明确比较 SDIRK2 与 SDIRK2-mr-SAV，不能视为旧 mrGSAV 实验复现。

自适应 `embedded`/`sav` 需要格式提供嵌入估计；`doubling` 只用于单步格式。多步启动沿用各格式的默认方案。上下限可为 null，只有达到有效下限且指标不满足才强制接受并计数。

## 独立分析

```bash
$PY -m experiments.bursting.analyze --runs /absolute/run-A /absolute/run-B
$PY -m experiments.bursting.analyze --batch /absolute/batch
$PY -m experiments.convergence.analyze --runs /absolute/run-A --reference /absolute/reference-run --recipe errors
$PY -m experiments.spatial_spectrum.analyze --runs /absolute/grid32 /absolute/grid48 --recipe spectrum
$PY -m experiments.bursting.analyze --runs /absolute/run-A --recipe pchip_kde --sample-dt 0.001
$PY -m experiments.bursting.analyze --runs /absolute/run-A --recipe events --sample-dt 0.001 --threshold 10
```

数据必须已存在；分析不会调用计算或选择最新运行。结果进入 reports/<实验>/<分析编号>，含 analysis.json、analysis.log、图表与 CSV。新分析不覆盖旧报告，批次成员会固定解析成具体运行编号。

基础曲线使用实际积分节点；能量为 0.5∫|u|²，enstrophy 为 0.5∫ω²。谱是 Fourier 壳中的空间平均 enstrophy；误差分析对匹配物理设置的末状态计算绝对 L² 与相对 L² 误差。参考解必须通过单独的显式计算获得；给定变步长使用最大实际步长作横坐标。

PCHIP/KDE 与事件分析要求显式均匀采样间隔，可用 `--window start end` 限定窗口。事件数是采样序列上从 ≤threshold 到 >threshold 的上穿次数；KDE 使用 SciPy 默认带宽。它们是明确的新公共配方，旧 notebook 的定制带宽、分组和论文图排版仍完整归档，未宣称逐图复现。

分析支持新 NPZ 运行和已有带身份信息的 bursting HDF5 运行。历史无 manifest 的散装文件暂保留原位，需先核实来源，不能按文件名伪造完成身份。当前报告默认拒绝失败/未完成记录；显式 `--allow-failed` 可展示有可读有限前缀的失败运行，标明实际终止时间，末时刻误差留空。

## 短程验收

```bash
$PY tools/smoke_experiments.py --output /tmp/imex-project-smoke
$PY -m unittest discover -s tests -q
```

smoke 覆盖七类实验的批次、保存、加载、复用、图表，附带参考误差、给定序列、cos(x) 变体、谱和均匀时间统计。通过只表示工作流可用，不证明论文数值结论。

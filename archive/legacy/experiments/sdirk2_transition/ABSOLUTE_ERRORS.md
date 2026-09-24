# 绝对误差：步长与时间两个方向的比较

参照项目根目录 `sdirk2_fixed_convergence_table.tex` 的“每个时间点下列出两种方法的 Error/Rate”结构，另存当前新算例的表格。当前参数仍为 `nu=0.2, N=256, T=6`，原多模态初值、非零涡量外力 `cos(x)`，比较 SDIRK2 与 mr-SAV 的 `gamma=5,1000`。原表对应原算例，本次没有覆盖它。

所有新表与新图的误差均为

\[
E_\omega(\tau,t)=\left(h_xh_y\sum_{i,j}|\omega_{ij}(\tau,t)-\omega_{{\rm ref},ij}(t)|^2\right)^{1/2}.
\]

不除以参考解范数，也不转换为百分数。误差在独立周期网格点上求和，重复端点不计入。

## 查看入口

已补充 gamma=100、500，使用相同算例、13 个步长及四个物理时间。所有四组的并列表见 [absolute_errors_gamma_comparison.md](absolute_errors_gamma_comparison.md)。新增两组仅补算这些表格所需的固定步长结果，下面原有的密集时间曲线仍对应 gamma=5、1000。

- [独立绝对误差 notebook](absolute_errors.ipynb)：默认读取已保存数据，可直接查看两类图和各张表。
- [误差随步长变化](../../fig/sdirk2_transition/absolute_errors/error_vs_tau.png)：分别固定 `t=1.5,3,4.5,6`。
- [误差随时间变化](../../fig/sdirk2_transition/absolute_errors/error_vs_time.png)：分别固定 `tau=0.0625,1/12,0.09375,0.15`。
- [gamma=5 的 LaTeX Error/Rate 表](absolute_errors_gamma5.tex)；[直接查看数据](absolute_errors_gamma5.md)。
- [gamma=1000 的 LaTeX Error/Rate 表](absolute_errors_gamma1000.tex)；[直接查看数据](absolute_errors_gamma1000.md)。
- [gamma=100 的 LaTeX Error/Rate 表](absolute_errors_gamma100.tex)；[直接查看数据](absolute_errors_gamma100.md)。
- [gamma=500 的 LaTeX Error/Rate 表](absolute_errors_gamma500.tex)；[直接查看数据](absolute_errors_gamma500.md)。
- [密集过渡步长的终点误差](absolute_errors_transition.md)。
- [tau=1/12 的时间误差数据](absolute_errors_time.md)。

两张 LaTeX 表采用实际 `tau` 作第一列，不再用需要换算的 `k`。主表包含 13 个步长，在同样四个物理时刻列出 Error/Rate。相邻步长的斜率按

\[
p_i(t)=\frac{\log(E_\omega(\tau_{i-1},t)/E_\omega(\tau_i,t))}{\log(\tau_{i-1}/\tau_i)}
\]

计算。表中的 Rate 是原始观测斜率，失稳/非渐近区的很大数值不表示获得高阶收敛；缺失或非正误差的 Rate 标为 `--`。`stopped` 表示方法在观测时间之前已经因预定阈值停止，不能与实际算出的 NaN 混同。

## 两个方向的现象

固定时间看步长：在 `t=1.5,3`，大部分步长的三组误差几乎重合；到 `t=4.5`，最大的步长已出现分离；到 `t=6`，较大步长的原 SDIRK2 误差快速放大，mr-SAV 则抑制该增长，gamma=5 的误差明显低于 gamma=1000。不能只根据 t=6 的表推断早期也有同等程度的优势。

固定步长看时间：`tau=0.0625` 的三组误差始终接近，并随本例的演化总体减小；`tau=1/12` 的三组误差到 `t=5.5` 仍然接近，此后才显著分离。具体地，`t=5.5` 时误差依次为 `2.121532e-2, 2.118613e-2, 2.121532e-2`；`t=6` 时变为 `2.717305, 0.1395836, 1.580903`。gamma=5 的后期误差也并非单调增长，它在 t=5.75 后下降。

`tau=0.09375` 的快速增长出现得更早；`tau=0.15` 的原 SDIRK2 在 `t=4.95` 触发数值爆破停止条件。时间图保留停止前的实际样本并标记停止时间；停止后的曲线不延长。

## 时间节点、数据和验证

主表使用所有指定观测时刻都落在固定时间网格上的步长。因此密集过渡区的奇数步数（例如 75、73）不塞进主表的中间时间列，而在独立终点表中比较。所有主表值直接读取原有 `omega_absolute` 数据。

时间曲线补算时，使用方法网格与 1536 步 ETDRK4 网格的公共节点：若方法共有 n 步，公共节点数由 `gcd(n,1536)` 决定。四条固定步长的采样间隔分别为 `0.0625,0.25,0.09375,0.75`。没有插值、没有为输出截短时间步，也没有用不同实际时刻冒充同一时间。图中连线仅用于连接离散样本，不意味着新增采样。

补算的 ETDRK4 末状态与原参考解逐点一致；12 个方法运行的末状态也都与此前持久化数据逐点一致，停止状态一致。参考解的时间/空间精度验证沿用同算例的 `validation.json`。本次没有改变求解器。

时间序列和元数据保存在 `data/sdirk2_transition/absolute_errors/`，索引为 `index.json`，纯误差数据为 `absolute_time_errors.json`。原始 NPZ 保留实际采样、状态及异常位置；JSON 分析文件将缺失误差写为 null，并配合状态解释。参考解保存 129 个公共节点流场（约 65 MiB 未压缩），方法只保存误差、q 与末状态，不保留每个方法的全场历史。

验证：两张 LaTeX 表在临时文档中用 pdflatex 编译通过；图像已目视检查；notebook 使用 Python 单元顺序执行与结构审计核对，不将其称为 Jupyter 内核端到端验证。

## 重现

```bash
PY=/Users/wanghaifeng/miniconda3/envs/pde/bin/python

# 显式补算时间序列；配置相容时复用已保存结果
$PY experiments/sdirk2_transition/absolute_errors.py compute

# 只重建绝对误差表格与图，不运行求解器
$PY experiments/sdirk2_transition/absolute_errors.py analyze

# 重建四个 gamma 的绝对误差表及并列表，不运行求解器
$PY experiments/sdirk2_transition/gamma_tables.py
```

新增脚本将补算与分析分开。改变图例、颜色或文字不触发积分。补算记录不包含继续积分所需的 checkpoint；中断后从初值重跑，完整保存的相容结果复用。

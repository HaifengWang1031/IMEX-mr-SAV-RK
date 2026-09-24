# 固定步长 SDIRK2 / SDIRK2-mr-SAV 独立比较

绝对误差的步长/时间对照入口：[absolute_errors.ipynb](absolute_errors.ipynb)，说明与 LaTeX 表格见 [ABSOLUTE_ERRORS.md](ABSOLUTE_ERRORS.md)。

从 `01_Convergence_Analysis.ipynb` 的 “Fix Stepsize (Comparison for original SDIRK2 and SDIRK2-mr-SAV scheme)” 独立提取。原 notebook 和求解器不作修改。

实验目的：寻找同一有外力算例的三个步长区间：小步长二阶精度接近、过渡区 mr-SAV 的物理解误差更小、大步长原方法数值爆破而 mr-SAV 保持有界。主指标为**相同步长、相同物理时间的涡量相对 L² 误差**；速度相对误差为辅助指标。保持有界不能替代精度验证。

## 模型、配置和公平比较

计算周期区域 `[0,2π]²` 上的二维不可压缩 NS 涡量方程：

\[
\omega_t+u\cdot\nabla\omega=\nu\Delta\omega+F\cos(k_f x),
\qquad -\Delta\psi=\omega,\quad u=(\psi_y,-\psi_x).
\]

这里的外力是**涡量方程右端**的外力。所有算例均要求 `F != 0`。使用现有 Fourier 伪谱空间离散及其 2/3 去混叠约定；原 SDIRK2 的代码名为 `IMEX_RK2`。不修改其 Butcher 系数、非线性项或 mr-SAV 标量方程。

`run.py` 的 `Case` 是唯一模型配置入口；CLI 覆盖后把实际配置随每次计算保存。`grid` 是每方向独立周期点数；构造初值时才补周期端点。`amplitude` 是初始涡量 RMS；`modes` 是固定物理频谱范围；`gamma` 为 mr-SAV 回归参数。`threads=1` 固定 FFTW 线程数，计时是本机测量但不是主要比较指标。

默认 `trig` 初值为

\[
\omega_0=C\sum_{k,m=1}^{10}(k^2+m^2)^{-3/2}\cos(kx)\cos(my),
\qquad \operatorname{RMS}(\omega_0)=A.
\]

`random` 在上述模态中加入固定种子的随机相位；`shear` 使用光滑周期剪切及非平行扰动。每对比较固定同一初值、外力、黏性、空间网格、终止时间和步长；选根规则也预先固定。默认 `legacy` 与原 notebook 相同；`nearest` / `farthest` 是独立敏感性检查，不在搜索中按结果切换。

## 计算与分析

从项目根目录执行，解释器建议：

```bash
PY=/Users/wanghaifeng/miniconda3/envs/pde/bin/python

# 当前 notebook 参数的基线（CLI 默认 N=64，正式复现显式用 256）
$PY experiments/sdirk2_transition/run.py compare --grid 256

# 第一轮 21 组有外力参数，低分辨率筛选
$PY experiments/sdirk2_transition/run.py screen

# 排序已保存的扫描；不执行求解器
$PY experiments/sdirk2_transition/run.py report

# 小规模驱动、保存、重载和失败路径验证
$PY experiments/sdirk2_transition/check_workflow.py
```

正式候选与精确重现命令见 `RESULTS.md`。独立 notebook `comparison.ipynb` 默认只读取推荐结果；没有数据时明确报错。运行计算必须显式打开计算开关或使用 CLI。

`--counts` 给出整数步数 `n`，实际固定步长为 `T/n`，整个积分不截短步长。终点总是保存；`T/4,T/2,3T/4` 只在它们恰好落到该时间网格上时保存。因此奇数步数只有初末快照；多时刻比较只使用双方实际拥有的相同时刻，绘图缺点保留为空，不插值填充。

每次只保存至多五个状态、每步标量诊断、末状态及配置，内存不随步数保留全部流场。单个 256² 算例的六个 float64 状态约 3 MiB；另有 FFT 临时数组和诊断。无继续积分式 checkpoint：被中断的算例从初值重跑，已经完整保存的算例可复用。

## 证据和状态

`data/sdirk2_transition/` 保存第一轮与扩展探索，`validated/` 保存支持任意整数步数的最终驱动产生的验证数据。每次运行一个 NPZ 和一个 JSON。运行身份包含实际配置、算法名、步数、停止阈值及驱动/求解器 SHA256。源文件改动导致新身份，旧证据不覆盖。环境版本和 Git HEAD 同时保存。不同参数列表可以增量增加计算；比较 JSON 记录各自成员。

写入前 JSON 标记 `incomplete`，NPZ 原子替换成功后写终态 JSON。NPZ 内也包含终态元数据；如果中断恰好发生在两次原子替换之间，以完整 NPZ 为完成依据。没有完整 NPZ 的项重跑。已保存的失败项默认复用，不隐式重试；重新评估可用新的输出目录。勿启动多个进程写同一个运行身份。

每步诊断列依次是 `time,q,omega_rms,omega_max_abs,omega_mean,forced_rms_bound`。由周期问题的涡量能量估计，连续解满足

\[
\operatorname{RMS}(\omega(t))\le A+|F|t/\sqrt2.
\]

这是用于识别极端异常增长的物理上界，并非精度指标。`solution_blowup` 的预定数值停止条件是 RMS 超过该上界的 **100 倍**；`nonfinite`、`floating_point_error`、`solver_failure` 分开记录。达到停止条件意味着强烈数值失稳，**不声称连续 PDE 有限时爆破，也不声称该步已产生 NaN**。尚未触发停止也可能有很大误差。

失败时保留有效前缀、已得到的异常末状态、实际物理时间和异常消息。绘图不把失败项补成终点误差；状态表展示失败。原始异常值不改成有限值。受迫问题中能量可以上升；保存瞬时外力做功用于辅助解释，不将能量上升当作失败。

候选筛选用相邻已扫描步长的终点误差比 `R <= 0.8`，并检查小步长误差接近；这只是经验筛选条件。最终仍需参考步长减半、空间加密和多时刻检查。整个扫描（包括 mr-SAV 更差的算例）保留，结果不代表该方法对所有初值或参数均更准确。

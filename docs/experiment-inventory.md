# 实验归属与迁移清单

> 2026-09-24 已执行快速迁移：下表记录的是迁移前归属证据，旧入口现位于 archive/legacy 下的对应原路径。当前入口和边界见 [迁移记录](migration-20260924.md) 与 [实验说明](../experiments/README.md)。

盘点日期：2026-09-13。基于 notebook 单元内容、脚本和已有 README 的只读检查，未执行 notebook 或生产计算。主题归属是源码证据；活跃程度、历史文件对应的完整参数和科学有效性尚未据此确认。

| 建议实验名 | 现有入口与内容证据 | 历史产物线索 | 迁移状态 |
|---|---|---|---|
| `convergence` | 根目录 `01_Convergence_Analysis.ipynb` 含 ETD-mrSAV 定步、普通/SAV SDIRK2 对照、给定变步长三部分；`compare_etdrk4_reference.py` 说明采用第一部分的初值和外力 | 根目录三份 `*convergence_table.tex` / `variable_step_etd_mrsav_ms2_l_table.tex`；代码引用 `data/etdrk4_reference_tau_comparison_T*.npz`；`fig/` | 待提取正式配置、计算和分析；参考解验证作为相关独立运行 |
| `mean_reverting` | `02_run_test_diagnostics.py` 计算 gamma 对照，`02_Mean_Reverting_Test.ipynb` 加载并分析 | `data/test_bursting_diagnostics.h5`；`fig/mrSAV_gamma_comparison.*`；`gamma_effect_table.tex` | 已有计算/分析分工，待运行记录迁移 |
| `adaptive_tolerance` | `03_run_mrsav_varstep_exp.py` 与 `03_mrSAV-VarStep-Exp.ipynb` 比较容差、网格及扰动 | `data/03_mrSAV_varstep*.npz`；`logs/03_mrSAV_varstep_exp.log`；`fig/adaptive_step_tol_*.pdf` | 待核对输入对应关系、集中参数及独立运行记录 |
| `bursting` | `04_run_bursting.py/.sh`、`04_plan.sh`、两份 `04_long_time_stability*.ipynb`；`06_long_time_stability_analysis.ipynb` 是 ETD-mrGSAV/gamma/ETDRK4 比较分支 | `data/ns_*_bursting_*.h5`；`fig/vs_diagnostics*.pdf` 等；新记录在 `runs/bursting/` 和 `runs/bursting_nextgen/` | 新计算记录已实现；独立分析记录待迁移；各分析分支保留 |
| `shearflow` | `05_shearflow.ipynb` 含双剪切层计算、图表和文字 | `fig/shear_layer_comparison.pdf`、`fig/shear_layer_error_enstrophy.pdf` | 先核实缺失求解器来源，再安排迁移 |
| `spatial_spectrum` | 两份 `07_isotropic_spatial_spectral_test*.ipynb`，无外力与 cos(x) 外力变体；跨网格 Fourier 截断和谱比较 | 分别使用 `data/isotropic_spatial_sdirk2_mrsav[_cosx]/` 与对应 `figures/` 子目录 | 两变体独立配置；计算与分析触发待分离 |
| `sdirk2_transition` | 已有目录 README 说明从 01 的 SDIRK2 对照提取；包含 run/analyze/validate 及 gamma、绝对误差、nu=.01 扩展 | `data/sdirk2_transition/`；`fig/sdirk2_transition/`；源码目录内部分 md/tex/json | 已有局部记录协议，待接口核验与输出归属迁移 |

表中路径模式是读写源码提供的线索，不表示每个模式对应文件均存在。当前存在的候选文件逐项列于 [历史索引](../archive/legacy-index.json)，共有数量与大小由工具重新生成。

## 不能直接合并或搬动的内容

- 两份 04 notebook 有不同分析内容。带 `2` 的版本包含 ETD 定步输入、PCHIP 重采样和 KDE；两者却可能写同名图。应转为独立分析配方，保留各自采样与统计语义。正文参数与文件名不完全一致时，后续读取 HDF5 元数据核验，不猜测。
- 06 的 ETD/gamma 比较不能仅因主题相似就并入同一张 bursting 图；需保留其输入选择和诊断定义。
- 03 notebook 读取带 `_128.npz` 的文件，而文档示例/脚本默认命名不同，不能直接替换为某个“最新”结果。
- 05 导入 `vs_ns_periodic_mrSAV_solver`，此次项目文件盘点未找到对应模块。尚未确认该入口可运行。
- 07 notebook 存在“优先生产、否则 smoke”的自动数据选择；cosx 变体当前计算配置含 `RUN_PRODUCTION=True`，与开头说明不一致。迁移分析入口时须改为显式输入选择，并分离生产计算。此次未执行或修改这些 notebook。
- transition 的 `absolute_errors.py`、`nu001.py` 混有计算与输出；`recommended.json` 是输入选择，不能与一般报告一起随意搬走。`run.py` 向旧 Solver 传入 `root_selection`，而此次搜索未找到旧 Solver 对应参数；属于待单独核验的接口疑点。
- `data/gamma_effect/`、`data/sdirk2_cubic_root_selection/`、`data/tolerance_search/` 等目录及根目录 `tabel.png` 暂仅登记路径。未逐一读元数据建立来源，不根据名称补造参数或实验归属。

## 后续迁移顺序与核验

建议先迁移 bursting 的分析入口，建立“一组固定/自适应运行 → 一份明确输入的分析记录 → 图表”的完整实例；随后整理 transition，复用确实相同的记录/加载逻辑；再提取收敛性与变步长实验。其他主题根据当前研究需求安排。

本轮完成了源码内容盘点和历史文件元数据索引，没有移动、删除或执行旧实验，也没有声称所有历史产物已确认归属。每迁移一个实验，在此补充新入口、原入口去向、输入来源核验和小规模验证结果。

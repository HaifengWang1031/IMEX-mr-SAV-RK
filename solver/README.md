# 独立的数值格式研究框架

本目录是项目唯一正式求解器，原 nextgen 内容已提升至此。旧单体求解器及其兼容导出已移除。

## 文件职责

```text
solver/
├── core.py                 # 状态、已接受历史、试算/决策、结果与保存
├── fourier_ns.py           # Fourier 空间运算、NS 外力与诊断
├── integrate.py            # 固定、给定序列、自适应的共享推进
├── schemes/
│   ├── sdirk2.py
│   ├── sdirk2_mrsav.py
│   ├── etdms2.py
│   ├── etd_mrsav_ms2_b.py
│   └── etdrk4.py            # 默认多步启动与参考用途
└── adaptivity/
    ├── error_based.py       # 嵌入误差；可替换归一化和控制器
    ├── sav_based.py         # 原有物理误差 + 标量偏离联合策略
    └── step_doubling.py     # 显式选择的单步倍步长估计
```

一个具体数值格式一个文件；一个完整自适应算法一个文件。共同的小约定集中在 `core.py`，不为每个约定再增加文件。推进流程不按格式或算法名称分派。

当前迁移：普通 SDIRK2（原名 `IMEX_RK2`）、SDIRK2-mr-SAV、ETDMS2、ETD-mr-SAV-MS2-b 和 ETDRK4。多步组的普通/SAV 版本使用相同的指数传播、非线性中点外推与外力时刻；这是代码公式层面的配对，不据此宣称两者有相同的稳定性定理。原有空间算子、选根和 ETD 系数计算细节均沿用；本次迁移不改这些数学实现。

## 最小调用

从仓库根目录、在 `pde` 环境中执行：

```python
import numpy as np
from solver import FourierNS, integrate
from solver.schemes import SDIRK2, SDIRK2MrSAV

model = FourierNS(
    nu=0.025, shape=(32, 32), threads=1,
    forcing=lambda x, y, t: np.cos(y),
)
x, y = model.X[:-1, :-1], model.Y[:-1, :-1]
initial = model.initial_state(0.1*np.cos(x)*np.cos(2*y))

# 普通格式不产生虚设的 q；换成 SDIRK2MrSAV(gamma=1000) 即可比较。
result = integrate(
    model, SDIRK2(), initial, (0.0, 0.02),
    dt=0.001, snapshots=[0, 0.01, 0.02],
)
print(result.times, result.auxiliary, result.stats)
result.save("/tmp/sdirk2-example.npz")  # 已存在时明确报错，不覆盖
```

`model.initial_state` 接收不含重复周期边界的内部涡量数组，并按旧约定去除均值。`State` 本身是通用状态容器，支持多个不同形状的具名物理场和多个具名实标量辅助变量；当前实际物理模型只实现了 NS。流函数、速度通过模型运算按需取得，未作为额外独立演化场。

给定步长改用 `steps=[...]`，其和必须为 `T-T0`；不能同时传入 `dt`。固定步长也要求区间可整除，避免旧求解器的隐式 `ceil` 越过终点。格式必须声明支持变步长，才允许给定变步长或自适应运行；声明表示实现有相应公式，不代表任意步长比都已通过稳定性验证。

## 自适应策略与控制器

```python
from solver.adaptivity import SAVControl

algorithm = SAVControl(
    tolerances={"omega": (1e-12, 5e-5)},  # atol、rtol
    scalar_limits={"q": (1.0, 1e-2)},    # 目标值、偏离容差
)
result = integrate(
    model, SDIRK2MrSAV(gamma=1000), initial, (0, 0.02),
    adaptive=algorithm, initial_dt=5e-4,
    min_dt=None, max_dt=None,
    snapshots=[0, 0.01, 0.02],
)
```

- `EmbeddedErrorControl` 使用格式配套的嵌入结果，对实验选择的场分别归一化并取最大值。可以替换 `indicator(model, history, trial, dt)`，返回至少包含无量纲 `error` 的指标字典；自定义 indicator 路径只请求主解。
- `SAVControl` 复现现有“涡量嵌入误差 + 辅助变量偏离”联合控制，可选择多个标量。`|q-1|` 是偏离指标，不是辅助变量局部误差估计。这里保留原 `1e-16` 偏移和步长更新公式。
- `StepDoubling` 显式执行一整步、两个半步，使用所声明阶数缩放物理场差，接受两个半步的结果，不做 Richardson 外推。它只允许单步格式；不是多步法的自动兜底。它不默认估计辅助变量误差。
- 普通 SDIRK2 当前没有迁入配套嵌入对，不能直接配 `EmbeddedErrorControl`；可显式研究 `StepDoubling`。例如针对二阶格式，可传入 `ProportionalController(exponent=1/3)`。这一路径已做工作流验证，尚未进行全面自适应精度研究。

更换控制器时，可向 `EmbeddedErrorControl` 或 `StepDoubling` 传入可调用对象 `controller(error, dt)`，返回建议步长。控制器可实现 `reset()`、`on_accept(metrics, forced)`、`on_reject(metrics)`，管理自己的历史。格式数值历史仍只包含已接受状态。

上下限均可为 `None`。上限只限制步长；达到下限仍未通过指标时强制接受并计数。终点剩余区间若短于下限，允许最后一步缩短到终点，并将这个剩余区间视为最后一步的有效下限。非有限解、非有限指标或时间无法前进，不会被强制接受。无下限时没有隐藏的正步长下限；用户可以额外提供 `max_steps` 作为显式计算预算。

## 新增一个格式

在 `schemes/` 新建文件，继承轻量 `Scheme` 约定，定义：

- `name`、`history_size`、`variable_step`；需要相关估计时声明 `order`、`embedded`。
- `auxiliary_defaults`：普通格式为空字典，SAV 格式按自身定义给出标量初值；可覆盖 `initialize` 实现更复杂的初始化。
- `step(model, history, dt, *, estimate=False)`：手写公式，返回 `Trial(State(...))`；配套嵌入结果放在 `embedded_fields`。定步不请求嵌入结果，避免额外 FFT。
- 多步格式提供 `startup(model, history, dt)`，或由实验通过 `integrate(..., startup=...)` 替换。

`history.states[-1]` 是当前状态，`history.time` 是当前时刻，`history.steps` 是状态间原始已接受步长。需要当前试探步长时使用参数 `dt`。这些状态及数组只读；试算不得原地修改它们。模型的有界系数缓存与试算历史无关，可以复用；不能把“上一试算的非线性项”作为已接受历史缓存。

当前两个多步格式默认使用 ETDRK4 生成一个启动步，并保持辅助标量初值为 1。其他辅助初值需要显式替换启动方案。启动步不经过主格式的嵌入误差接受判据，单独记录 `startup_steps`；这是迁移时保留的原行为，启动误差研究需要配套启动方案。

新格式可以在实验中直接导入并实例化，无需修改 `integrate.py`。`schemes/__init__.py` 仅提供常用格式的导入便利，不是运行时注册表。

## 新增一个完整自适应算法

在 `adaptivity/` 新建文件，继承 `AdaptiveAlgorithm`，实现：

```python
def attempt(self, model, scheme, history, dt):
    # 本算法可调用格式一次或多次；决定指标、接受条件和建议步长。
    # 不得向 history 提交候选状态。
    return Assessment(trial, accept, next_dt, metrics)
```

`validate` 检查与格式/状态的兼容性，`reset` 开始新运行；`on_accept` 和 `on_reject` 更新算法自己的历史。推进流程执行它的决定，并统一处理用户设置的上下限、历史提交和快照。只调节下一步而不拒绝的研究算法也可表达为每次返回 `accept=True`，但当前未内置具体的此类科学算法。

## 快照与结果

- 固定/给定步长：请求必须在真实网格上，仅允许浮点舍入误差；不插值。
- 自适应：跨过请求时刻后比较前后两个已接受节点，以中点判定最近者；等距取较早节点，不改变步长以命中请求。
- `requested_times` 保留原请求顺序；`snapshot_times` 是实际存储节点；`snapshot_indices` 将每个请求映射到存储位置。多个请求可共享同一状态。
- `actual_snapshot_times` 给出逐请求的实际时刻。失败结果中未完成请求的映射为 `-1`，实际时刻为 `NaN`。
- `times`、`steps`、`auxiliary`、`diagnostics` 保存标量轨迹；物理场仅保留短历史、请求快照和最终状态，不存每一步全场。
- 控制指标使用 `control/` 前缀；初值或启动阶段未计算的指标以 `NaN` 明示，不伪装成零误差。

```python
from solver import Result, IntegrationError
saved = Result.load("/tmp/sdirk2-example.npz")  # 只读，不计算
# saved.snapshot_fields["omega"][saved.snapshot_indices[i]] 对应第 i 个请求
# 失败记录需要先检查 snapshot_indices[i] >= 0
```

积分失败抛出 `IntegrationError`，其中 `.result` 包含已接受前缀；非有限候选单独放在 `failed_state`，可保存原始 NaN/inf。结果保存不使用 pickle，不覆盖已有文件。它是结果记录，不是续算检查点。

`progress` 接收限频进度字典，包含物理时间、接受/拒绝/强制接受次数和 wall time。新统计中的 wall time 与旧 `CPU_time` 口径不同，不直接对比性能数值。

## 试用与验证范围

实验使用统一入口：

```bash
python -m experiments.bursting.run --show-config
python -m experiments.bursting.run
```

参数位于 `experiments/bursting/configs/default.json`；使用 `--config` 传入覆写配置。
完整用法见 [实验说明](../experiments/README.md)。

## 2026-09-24 项目迁移

新增独立格式 ETDMrSAVMS2L、MrSAVBDF2、IMEXEuler、LegacyLinearETD；回归对照使用 tests/fixtures 中冻结的数值基准。
LegacyLinearETD 保留旧 ETD 的线性扩散加外力公式，不含对流项。
正式实验统一入口及配置见 [实验说明](../experiments/README.md)。

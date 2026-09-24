# Bursting：配置、运行与产物

目的：对原有周期受迫流算例比较固定步长与自适应积分。保留原初始流函数、外力 `m*cos(m*y)`、ETDRK4 预热、主时间从零开始的约定与求解公式。保存涡量快照、辅助变量、能量/拟涡能等诊断，以及物理时间和计算耗时。

## 修改配置与启动

主配置是 [config.json](config.json)，其中 `_help` 解释各参数。原脚本的默认值未改变，默认 `T=10000` 是长时间生产计算。

从仓库根目录运行。以下先查看实际配置，不进行计算：

```bash
conda activate pde
python 04_run_bursting.py --show-config
```

可编辑主配置，也可用 `--config /path/to/my_bursting.json` 指定另一份 JSON。自定义文件允许只包含要覆盖的字段；未知字段、非有限数或无效组合会被拒绝。优先级：主配置 → 自定义配置 → 命令行。相对 `output_root` 一律以仓库根目录为基准，不随启动目录变化。

短程检查示例（单个算例，非生产结论）：

```bash
python -u 04_run_bursting.py --mode fix --N 32 --T 0.004 --tau 0.001 \
  --warmup-time 0.002 --warmup-tau 0.001 --snapshot-dt 0.002 \
  --output-root /tmp/bursting-smoke
```

自适应示例：

```bash
python -u 04_run_bursting.py --mode adaptive --N 32 --T 0.004 \
  --warmup-time 0 --snapshot-dt 0.002 --output-root /tmp/bursting-smoke
```

原批量入口仍可用：

```bash
./04_run_bursting.sh --m 4 --Re 50 --T 5000 --N 256
```

该 shell 明确运行三个比较算例：自适应、定步 `tau=5e-4`、定步 `tau=1e-3`；这三者分别有运行记录。shell 固定选择这些模式、格式和步长，其余配置可通过配置文件或命令行修改。单独选择格式或步长请调用 Python 入口。

## 结果在哪里

每次真正计算创建一个独立目录并在终端打印路径：

```text
runs/bursting/<身份哈希前缀>-<UTC时间>-<随机标识>/
├── config.json        # 参数展开后的实际配置，包括派生量和执行设置
├── manifest.json      # 完整身份、源码哈希、数值库版本、状态及产物清单
├── run.log            # 配置、分阶段进度、警告、异常、最终结果位置
├── results.h5         # 求解成功返回后保存；保存期间为 results.h5.tmp
├── figures/           # 本次运行的后续分析图片位置
└── tables/            # 本次运行的后续分析表格位置
```

`figures/` 和 `tables/` 是早期预留位置，本入口只计算和保存数据，不自动生成图表。按现已确认的 [项目约定](../../docs/project-architecture.md)，新分析输出进入 `reports/<分析名>/<分析编号>/` 并记录所有输入 run-id；已有预留目录保持兼容。旧 notebook 继续读取旧 `data/` 文件；使用新结果时需明确选择上面打印的 `results.h5`，不会自动跳到“最新一次”。本轮未迁移旧数据或修改 notebook。

## 复用和重新运行

同一组实际生效参数、相关源码和环境默认复用已完成结果，终端显示 `REUSED` 与原记录路径，不修改原 log。固定模式不把自适应参数算入身份，反之亦然；日志间隔和输出目录也不属于数值身份。

需要重新测量耗时或重新计算时，在原命令后添加 `--rerun`，创建新的尝试，保留原记录。仅有配置哈希相同还不足以复用：程序会核对 config、manifest、HDF5 身份、完成标志和关键数组形状/终点。检测到已完成记录损坏时明确报错，检查该记录或显式 `--rerun`。复用检查不是对整个 HDF5 数据做逐字节完整性验证。

捕获异常记 `failed`，保留配置、log 与已有文件；直接杀死进程可能留下 `running`，也视为未完成。再次执行原命令会跳过这些记录；如果另有兼容的已完成记录仍会复用，强制重新计算使用 `--rerun`。重新计算包含初值和预热，不是断点续算。

当前求解器若在积分中途抛出异常，并不提供已裁剪的可靠状态前缀；此时只保证运行记录和日志，不保证有可用 HDF5 或检查点。若积分返回但含 `NaN/inf`，保存原始输出并标记 `nonfinite`，manifest 为 `failed`，不会作为成功结果复用。

## 日志和结束判断

`--log-interval 30` 控制现有逐步进度输出进入 log 的最小间隔。日志记录阶段（warmup/main）、物理时间、诊断和该阶段 wall time；阶段结束立即刷新。没有进度回调的耗时步骤只能等求解器返回进度后记录，日志间隔不是心跳保证。

```bash
tail -f /absolute/path/to/run/run.log
```

以 `manifest.json` 的 `status=completed`、可读的 `results.h5` 及 log 中的 `COMPLETED` 判断结束。成功标记在 HDF5 关闭并原子改名之后才写入。仅有 `running` 或日志仍在增长不表示成功。长计算在自己的终端启动并查看 log；本次整理不自动启动生产计算。

## 读取与解释

```python
from pathlib import Path
import h5py

run_dir = Path("/absolute/path/to/selected/run")
with h5py.File(run_dir / "results.h5", "r") as f:
    assert f.attrs["status"] == "completed"
    tn = f["tn"][:]          # 积分节点
    q = f["q"][:]           # 与 tn 对齐
    snapshot_times = f["tn_s"][:]
    last_omega = f["Omega"][-1]  # 只读需要的场，避免一次读入全部快照

# 新分析图片保存到 reports/<分析名>/<分析编号>/figures/，并记录输入 run-id。
# 读取或重画不会触发任何求解。
```

原 HDF5 字段与属性保留，新增加运行身份、实际配置、采样策略及固定步长 `tau` 数组。自适应结果额外保存 `rel_err`、`controller_err` 等已有诊断和接受/拒绝/强制接受计数；`compute_ref_err=false` 时不能把零 `ref_err` 解读为零误差。

固定快照沿用最近整数步间隔，自适应快照沿用线性插值；用保存的 `tn_s` 判断实际输出时刻。预热仍采用原求解器的 `ceil` 步数，`warmup_actual_end` 记录实际预热终点。`CPU_time` 保留原求解器累计步计时，manifest 中的 `elapsed_wall_seconds` 包含配置记录后的初始化、预热、主积分、检查和保存，二者不是同一口径。

工作流完成只表明数据可追溯且通过基本输出检查，不证明 PDE 精度、长期统计收敛或格式的理论性质。

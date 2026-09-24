# 同一算例：gamma=5 与 gamma=1000

结论：本算例中，gamma=1000 的过渡区与大步长精度明显差于 gamma=5；小步长差异约 0.7%。gamma=1000 仍保留大步长稳定性优势，未在本次 22 个步长中触发失败。不能据此断言 gamma=5 是最优值或较小 gamma 总是更好。

控制：N=256，nu=0.2，T=6，原 K=10、RMS=1 多模态初值，涡量外力 cos(x)，legacy 选根。只改变 gamma，其他配置和源文件哈希均相同。

参考解重新计算后与 gamma=5 的参考快照逐点完全相同；所有原 SDIRK2 对照的保存快照也逐点完全相同，失败状态与物理时间一致。

## T=6 涡量相对 L² 误差（百分数）

| 步长 | 原 SDIRK2 | mr-SAV gamma=5 | mr-SAV gamma=1000 |
|---:|---:|---:|---:|
| 0.011718750 | 4.179574e-05% | 4.148772e-05% | 4.179573e-05% |
| 0.023437500 | 0.0001675638% | 0.0001663195% | 0.0001675638% |
| 0.046875000 | 0.0006722264% | 0.000667131% | 0.0006722262% |
| 0.080000000 | 3.112274% | 1.808792% | 3.110569% |
| 0.081081081 | 5.552791% | 1.250803% | 5.51959% |
| 0.082191781 | 9.861061% | 0.9724821% | 9.153288% |
| 0.083333333 | 17.42815% | 0.8952558% | 10.13953% |
| 0.107142857 | solution_blowup | 1.238212% | 9.887474% |
| 0.150000000 | solution_blowup | 1.443628% | 10.3722% |

小步长 n=512 时，gamma=5 与 gamma=1000 的原始相对误差分别为 4.1487722e-7 与 4.1795735e-7，后者约大 0.74%。

在 tau=6/74 时，误差从约 1.25% 增到 5.52%；在 tau=1/12 时，从约 0.90% 增到 10.14%。tau=0.15 时两个 mr-SAV 计算都完成 T=6，但误差分别约 1.44% 和 10.37%；原 SDIRK2 在 t=4.95 触发预定的数值爆破停止条件。

代码中的非线性修正系数是 1-(1-q)^2。较大的 gamma 加强 q 向 1 的回归。在 tau=6/74 的本次计算中，gamma=1000 的 q 偏离更小，修正在增长后期较弱，轨迹更接近原 SDIRK2。这个解释与保存的 q 轨迹一致，不是关于 gamma 与误差单调关系的定理。

gamma=1000 本次计算限定在 256² 网格，没有另做其自身的 128² 空间加密。参考解与之前已验证的物理参考解完全相同；gamma=5 的空间验证仍保留在 validation.json。

## 文件与重现

- [gamma 对比图](../../fig/sdirk2_transition/gamma1000/gamma_comparison.png)
- [对照审计数据](../../fig/sdirk2_transition/gamma1000/gamma_comparison.json)
- [gamma=1000 原始比较索引](../../data/sdirk2_transition/validated/comparison_cdbdffc79e6ea6fc3699.json)

```bash
PY=/Users/wanghaifeng/miniconda3/envs/pde/bin/python
$PY experiments/sdirk2_transition/run.py compare \
  --grid 256 --nu .2 --gamma 1000 --final-time 6 \
  --counts 40 56 64 68 70 71 72 73 74 75 76 77 78 79 80 82 84 88 96 128 256 512 \
  --reference-steps 1536 --output data/sdirk2_transition/validated

# 仅从已保存数据重画图并核对控制变量
$PY experiments/sdirk2_transition/compare_gamma.py \
  data/sdirk2_transition/validated/comparison_47ec8f825c785777da8f.json \
  data/sdirk2_transition/validated/comparison_cdbdffc79e6ea6fc3699.json \
  --destination fig/sdirk2_transition/gamma1000
```

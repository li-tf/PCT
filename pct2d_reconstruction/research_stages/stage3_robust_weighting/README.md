# 阶段3：稳健过滤、WEPL噪声和质子权重

本目录是独立研究工作区。它只读复用成熟的pair格式、Schulte MLP、RSP评价和
Huber-TV实现，不修改：

- `preprocessing/`
- `analytic_reconstruction/`
- `iterative_reconstruction/`

大型掩码、权重、DDB和重建结果写入各实验在`data/`中的目录；代码侧`qc/`
只保存清单、CSV、日志和总结。

## 最终状态

阶段3已于2026-07-25完成。完整比较和锁定测试流程为**PASS**，但没有新方法
达到预设晋升门槛，最终保留：

- 过滤：`baseline_3sigma`；
- 权重：`equal`。

median/MAD和稳健马氏距离未达到残差p99至少改善5%的门槛；按出射能量拟合的
噪声模型在S2、S3均只有7/10个十分位通过校准，逆方差权重反而恶化验证RMSE、
材料MAPE并引入WEPL负偏差。成熟的预处理和重建代码未被替换。

详细结果见[stage3_summary.md](qc/stage3_summary.md)，机器可读决定见
`qc/stage3_summary.json`。

## 科学设计

划分发生在任何过滤之前。质子身份固定为
`(RunID, paired_row_index)`，使用种子`20260713`的`splitmix64-v1`产生
80%训练、10%验证和10%测试。过滤器、噪声模型及权重归一化只从训练集拟合；
测试集在过滤器和权重冻结后才打开。

过滤候选包括训练集局部3σ、median/MAD和联合稳健马氏距离。两成分GMM仅在
每10个角度抽一个角度进行pilot，不会因一次小规模结果自动进入正式重建。
S2均匀水用于按出射能量拟合WEPL随机标准差，S3检查该模型从Vacuum向Air场景
迁移时的校准。逐质子权重限制在`[0.25,4]`，并检查每角度有效样本量。

阶段3的CUDA反投影实现：

\[
\Delta x=\lambda\,
\frac{A^\mathrm{T}WD_r^{-1}(b-Ax)}
     {A^\mathrm{T}W\mathbf 1}.
\]

当`W=1`时，单元测试检查其分子和分母与成熟等权算子一致。

## 分段运行

完整复现顺序：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action prepare --datasets s2,s3,s4,s5 --jobs 4

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action filter-screen --datasets s2,s4 --jobs 4

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action weight-screen --datasets s2,s4 --jobs 4 --device 0

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action prepare --datasets s1 --jobs 4

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action confirm --datasets s1,s3,s5 --jobs 4 --device 0
```

正式执行严格按上述顺序完成：先冻结过滤器，再运行`weight-screen`，最后处理
高通量S1并打开S1/S3/S5测试集。所有正式动作默认拒绝以`--force`覆盖自己的
完整产物。

轻量检查：

```bash
.venv-gate/bin/python -m unittest \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/test_stage3.py

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action smoke --datasets s2 --device 0
```

`--runs`仅用于开发检查。DDB-FDK和正式迭代会拒绝少于720角度的输入，避免把
不完整角度数据误当作研究结果。

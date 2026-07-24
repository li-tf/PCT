# 阶段3：稳健过滤、WEPL噪声和质子权重

本目录是独立研究工作区。它只读复用成熟的pair格式、Schulte MLP、RSP评价和
Huber-TV实现，不修改：

- `preprocessing/`
- `analytic_reconstruction/`
- `iterative_reconstruction/`

大型掩码、权重、DDB和重建结果写入各实验在`data/`中的目录；代码侧`qc/`
只保存清单、CSV、日志和总结。

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

第一批长计算：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action prepare --datasets s2,s3,s4,s5 --jobs 4

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action filter-screen --datasets s2,s4 --jobs 4
```

等待检查并冻结过滤器后才运行`weight-screen`。最终才处理高通量S1并打开
S1/S3/S5测试集。所有正式动作默认拒绝以`--force`覆盖自己的完整产物。

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

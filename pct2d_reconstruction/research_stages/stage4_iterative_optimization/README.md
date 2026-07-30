# 阶段4：固定MLP下的迭代重建优化

本目录是独立研究工作区。它固定使用阶段3保留的局部3σ过滤、等权数据和
Schulte水MLP，只优化OS-SART松弛调度、数据损失、Huber-TV、停止epoch及子集
数量。成熟的`preprocessing/`、`analytic_reconstruction/`和
`iterative_reconstruction/`不会被修改。

大型图像和检查点写入各数据集的`data/reconstruction_data/.../stage4/`；
代码侧`qc/`只保存配置、选择结果、指标和总结。

## 最终状态（2026-07-27）

- GPU两角度smoke test通过，quadratic更新与阶段3等权算子的分子、分母差异均为0；
- 松弛调度冻结为初值`0.25`、衰减`0.2`；
- Huber 3/5 mm均未改善验证WEPL RMSE或p99，保留quadratic数据项；
- Huber-TV冻结为固定权重`0.0125`，停止在第5 epoch；
- 相对同轮无正则化结果，S2有效RSP RMSE、水区标准差和S5标称RSP RMSE分别
  改善约41.5%、81.3%和9.6%；S4材料MAPE仅轻微恶化且仍通过约束；
- 18/36子集比较已经完成：36子集的平均验证WEPL RMSE仅改善`0.0537%`，
  未达到预设`0.2%`门槛，因此冻结18子集；
- S1--S5锁定测试已经完成，所有WEPL、材料和MTF安全检查通过；
- S2/S3水区标准差平均降低`42.58%`，达到实质改善门槛；
- 最终决定为`PROMOTE_STAGE4`，详细结论见`qc/stage4_summary.md`。

该冻结配置已通过Stage 6B校准三场景和Stage 7探测器效应实验继续验证。S1、
S4、S5的正式Stage 6B检查点仍在本地；S2、S3和早期阶段4大型检查点已进入
第一批冷归档。若要完整复跑以下筛选流程，先按
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)
恢复相应数据。

## 数据纪律

- 训练使用阶段3过滤前80/10/10划分中的train质子及`baseline_3sigma`掩码；
- 每个epoch只用validation质子选择参数；
- `frozen_final.json`生成前，程序拒绝读取test质子；
- 阶段3否决的逆方差权重不在本阶段重新搜索。

Huber数据损失通过IRLS因子

\[
q_i=\min(1,\delta/|b_i-A_ix|)
\]

修改当前批次的OS-SART分子和分母。它与图像域Huber-TV是两个独立模块。将
`delta`设为`None`时，GPU smoke test要求结果逐元素复现阶段3等权算子。

## 分批命令

先运行轻量检查：

```bash
.venv-gate/bin/python -m unittest \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/test_stage4.py

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action smoke --datasets s2 --device 0
```

耗时任务必须按以下顺序执行，并在每一步完成后检查代码侧选择JSON：

```bash
# 1. 松弛因子与衰减
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action relaxation-screen --datasets s2,s4 --device 0

# 2. quadratic与Huber数据损失
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action loss-screen --datasets s2,s4 --device 0

# 3. Huber-TV权重、衰减和停止epoch
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action regularization-screen --datasets s2,s4,s5 --device 0

# 4. 18/36子集
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action subset-screen --datasets s2,s4,s5 --device 0

# 5. 冻结后测试
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action confirm --datasets s1,s2,s3,s4,s5 --device 0
```

重复相同命令会读取已有epoch并继续缺失部分。只有明确需要重算当前候选时才使用
`--force`；配置哈希不一致时程序拒绝混写。

另开终端可用只读状态入口查看整个Stage4动作的汇总进度，而不只是当前单个epoch：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action status --datasets s2,s4,s5
```

持续刷新：

```bash
watch -n 30 '.venv-gate/bin/python pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py --action status --datasets s2,s4,s5'
```

该动作只读取`epoch_metrics.csv`和选择JSON，不加载GPU、不创建结果，也不会干扰
重建。查看历史最终确认时可把数据集改为`s1,s2,s3,s4,s5`；若数据已归档，
状态页只反映仍在本地的产物。

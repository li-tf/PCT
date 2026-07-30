# Stage 6B：独立WEPL标定

本阶段使用独立OpenGATE水板数据拟合单调水射程函数，不使用S1--S5重建指标
调参。成熟的`bb78`模型和历史结果保持不变。阶段已于2026-07-29完成，最终
决定为`PASS（PROMOTE_G4_WATER_CALIBRATED）`。

工作站仿真完成并把数据复制到
`data/simulation_data/results0728_stage6b_wepl_calibration/`后执行：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6b_wepl_calibration/run_stage6b.py \
  --action all --datasets s2,s3,s1,s4,s5
```

`all`依次完成ROOT配对和稳健抽样、训练/验证拟合、一次性打开锁定测试集，并在
测试通过后完成S2、S3、S1、S4、S5的直接WEPL pairs、DDB、no-Hann初值和
Stage 4五轮GPU重建。S2/S3总是最先重建；S2水区未进入`1.000±0.003`时自动
停止，不会继续消耗S1/S4/S5的GPU时间。重复执行会复用已通过的产物。

只查看状态：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6b_wepl_calibration/run_stage6b.py \
  --action status
```

冻结模型位于`qc/g4_water_calibrated.json`。正式迭代入口通过：

```text
--wepl-model g4_water_calibrated
--wepl-calibration <上述JSON>
```

加载该模型。`bb78`仍是默认值，因此旧命令和旧结果不会被隐式改变。

核心结果：

- 锁定测试平均/最大绝对相对偏差为`0.0461%/0.1657%`；
- S2/S3水区均值为`0.999641/0.999635`，门控通过；
- S4大材料柱MLIC-MAPE由`1.1987%`降至`0.2551%`；
- S5平均fMTF50/fMTF10相对提高`1.69%/5.07%`；
- S1小铝柱相对MLIC仍低`1.307%`，作为材料和小目标剩余误差保留。

完整结论见[`qc/stage6b_summary.md`](qc/stage6b_summary.md)。

当前冻结模型、标定QC以及S1/S4/S5的`stage6b_calibrated`正式重建均保留在
本地，是后续复用的主输入。S2/S3门控数据已经迁入第一批冷归档；只有重新执行
五数据集完整`all`流程时才需要恢复它们。恢复清单见
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)。

若不希望一次运行全部耗时任务，可按顺序分别使用：

```text
--action ingest
--action fit
--action verify
--action prepare-pairs
--action project
--action analytic
--action reconstruct
```

`prepare-pairs`不会修改原pairs，只在各数据集的`stage6b_calibrated/`下写直接
WEPL副本。DDB和重建同样进入该隔离目录。锁定测试失败时程序返回非零状态并
阻止后续阶段。

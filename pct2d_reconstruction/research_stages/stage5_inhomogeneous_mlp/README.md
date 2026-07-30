# 阶段5：非均匀MLP与迭代更新MLP

本目录是独立研究实现，不修改成熟的`preprocessing/`、
`analytic_reconstruction/`或`iterative_reconstruction/`。阶段5先使用
`results0724_mlp_truth_pilot`的Geant4逐step轨迹验证非均匀MLP的路径上限，
只有通过预注册门槛后才进行S1--S5全通量重建。

## 最终状态

阶段5已于2026-07-27闭环。72个角度中得到222,901条可用真实轨迹；真值材料图
驱动的非均匀MLP对全部/强异质验证路径的平均改善仅为`0.006%/0.074%`，
bootstrap 95%下限为`-0.194%`，未达到Level 1门槛。程序按预注册规则未启动
Level 2固定图像非均匀MLP和Level 3交替更新，最终决定为
`RETAIN_STAGE4_LEVEL1_FAIL`。这是一项有效的负结果，不是运行失败。

MLP truth pilot的ROOT、预处理缓存和大型轨迹数据已于2026-07-30迁入第一批
冷归档；代码、QC和[`stage5_summary.md`](qc/stage5_summary.md)仍在本地。
完整复跑前需按
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)
恢复数据。

## 方法

非均匀MLP使用

\[
T(u)=RScP(u)T_w(E(u))
\]

构造局部散射能力。内部能量由入口和出口测量分别进行正向、反向Euler传播后按
深度融合，再数值积分三个Fermi--Eyges散射矩。Level 2从固定no-Hann初值生成
RScP图；Level 3在epoch 1、3和5之前阻尼更新RScP图。图像更新参数完全继承
阶段4冻结配置。

## 单命令运行

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage5_inhomogeneous_mlp/run_stage5.py \
  --action all --datasets s1,s2,s3,s4,s5 --jobs 4 --device 0
```

命令支持断点续跑。相同配置下重复运行会跳过已经完成且通过QC的ROOT分片、
候选和epoch；不要在正常续跑时添加`--force`。

另开终端查看聚合进度：

```bash
watch -n 30 '.venv-gate/bin/python pct2d_reconstruction/research_stages/stage5_inhomogeneous_mlp/run_stage5.py --action status --datasets s1,s2,s3,s4,s5'
```

Level 1或Level 2未通过时，`--action all`会正常生成负结果报告并结束，不会把
科学门槛失败误报为程序错误。

## 输出

- 代码侧：`qc/`保存manifest、进度、选择结果、CSV、图和
  `stage5_summary.md`；
- 轨迹缓存：`data/preprocessing_data/results0724_mlp_truth_pilot/stage5/`；
- 重建检查点：各数据集`data/reconstruction_data/.../stage5/`。

快速检查：

```bash
.venv-gate/bin/python -m unittest \
  pct2d_reconstruction/research_stages/stage5_inhomogeneous_mlp/test_stage5.py

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage5_inhomogeneous_mlp/run_stage5.py \
  --action smoke --device 0
```

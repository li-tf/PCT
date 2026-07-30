# 研究阶段工作区

本目录集中保存阶段性研究代码、只读分析、QC、图表和技术总结，避免在
`pct2d_reconstruction/`顶层为每个实验单独新增目录。大型ROOT、pairs、DDB和
重建图像仍只写入仓库根目录的`data/`。

```text
research_stages/
├── stage1_material_calibration/    S6材料能量、有效RSP和Air WEPL标定（PASS）
├── stage2_diagnostic_phantoms/     S2--S5边界、材料与分辨率诊断（PASS）
├── stage3_robust_weighting/        稳健过滤、WEPL噪声与权重（PASS，保留基线）
├── stage4_iterative_optimization/  固定MLP迭代优化（PASS，方案晋升）
├── stage5_inhomogeneous_mlp/       非均匀与交替更新MLP（PASS，保留阶段4）
├── stage6_advanced_priors/         TGV与自适应图像先验（PASS，保留阶段4）
├── stage6a_mlic_reference/         虚拟MLIC真值与双口径重评（PASS）
├── stage6b_wepl_calibration/       独立WEPL标定与三场景复算（PASS，模型晋升）
├── stage7_detector_effects/        D1 Air/硅跟踪器与离线数字化（PASS）
└── stage8_compact_3d/              compact-3D输入与标定门控（READY）
```

每个阶段使用一个统一入口，并将该阶段的完整性检查直接集成在计算流程中，不再
额外创建`validate_stage*.py`。

## 当前数据状态

截至2026-07-30，阶段代码、QC、图表和总结均保留在本目录。为给Stage 8三维
重建腾出本地空间，results0716、S2、S3、MLP truth pilot及S6材料能量扫描的
大型数据已经迁入第一批冷归档；S1、S4、S5的Stage 6B正式三场景结果和Stage 7
正式结果仍在本地。归档相对路径、文件数量和校验信息见
[`../archive_batch1_20260730_record.md`](../archive_batch1_20260730_record.md)。
需要完整复算Stage 1--6时，应先按该记录恢复对应数据，不能让程序把“已归档”
误判为“从未生成”。

阶段3完成了过滤前80/10/10划分、稳健过滤、噪声校准、加权OS-SART及锁定测试
确认。没有候选达到预设晋升门槛，因此后续正式基线仍为局部3σ过滤和等权
OS-SART。详细结论见
`stage3_robust_weighting/qc/stage3_summary.md`。

阶段4已完成松弛调度、Huber数据损失、Huber-TV、18/36子集筛选和锁定测试，冻结
`λ0=0.25`、衰减`0.2`、quadratic数据项、固定`β=0.0125`、5 epoch和18子集。
36子集只改善`0.0537%`验证WEPL RMSE，未达到`0.2%`门槛。S1--S5测试WEPL均
未恶化，S2/S3水区标准差平均降低`42.58%`，最终决定为`PROMOTE_STAGE4`。
详细结论见`stage4_iterative_optimization/qc/stage4_summary.md`。

阶段5已完成`results0724_mlp_truth_pilot`的真实轨迹上限实验。72个角度中共有
222,901条可用轨迹；真值材料图驱动的非均匀MLP在验证集全部/强异质路径上的
平均改善仅为`0.006%/0.074%`，bootstrap 95%下限为`-0.194%`，未达到预注册
门槛。程序按设计停止Level 2/3，最终决定为`RETAIN_STAGE4_LEVEL1_FAIL`，后续
仍使用阶段4水MLP基线。详细结论见
`stage5_inhomogeneous_mlp/qc/stage5_summary.md`。

阶段6已完成14组高级先验预筛和S2/S4/S5的6次完整验证重建。TGV未通过预筛
MTF约束；边缘自适应TV和方向TV虽将S5 fMTF提高约3%--8%，却使S2水区标准差
分别增加`47.15%`和`27.47%`，材料与RSP指标也没有达到实质改善线。程序在
验证门槛正常停止，未打开锁定测试，最终决定为
`RETAIN_STAGE4_VALIDATION_FAIL`。详细结果见
`stage6_advanced_priors/qc/stage6_summary.md`。

阶段6A完成24-case、150--220 MeV虚拟MLIC扫描及200 MeV高统计补充。冻结的
Water/Aluminium MLIC-RSP分别为`0.999746/2.094511`。results0716迭代铝平台
相对MLIC仅低`0.136%`，但S4阶段4材料MAPE仍为`1.192%`，说明S4误差不能主要
归因于旧真值。完整结果见
`stage6a_mlic_reference/qc/stage6a_summary.md`。

阶段6B使用84个独立水板工况冻结Geant4一致射程表，锁定测试平均/最大误差为
`0.0461%/0.1657%`。S2/S3水均值门控通过，S4大材料柱MLIC-MAPE由
`1.1987%`降至`0.2551%`，且S5空间分辨率未退化，最终决定为
`PROMOTE_G4_WATER_CALIBRATED`。完整结果见
`stage6b_wepl_calibration/qc/stage6b_summary.md`。

阶段7已完成720角度六平面配对、8套10%筛选和3套全量5 epoch重建。相对理想
参考面，四层连续硅hit拟合使水区标准差增加`3.41%`、模体RMSE增加`2.21%`，
铝平台仅变化`0.047%`，说明冻结算法在连续硅hit下仍稳定。`0.2 mm`位置噪声与
`1%`出射能量噪声组合使水区标准差和RMSE分别增加`42.08%/42.73%`，但D1没有
物理能量探测器，该结果仅为离线参数化灵敏度分析。完整结果见
`stage7_detector_effects/qc/stage7_summary.md`。

后续主线依次为Stage 7B D1噪声稳健重建、Stage 7C通量敏感性、Stage 8
compact-3D体素重建和Stage 9单场景3D Gaussian可行性。Stage 7B/7C复用现有
D1数据，无需新增蒙卡；compact-3D原始ROOT当前保存在外部存储，正式运行前需
挂载并通过Stage 8 preflight。

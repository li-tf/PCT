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
├── stage7b_noise_robustness/       D1位置/能量噪声鲁棒重建（PASS，保留等权）
├── stage7c_fluence_sensitivity/    D1嵌套有效质子通量敏感性（PASS）
├── stage8b_low_fluence_adaptation/ 低通量角度分配、参数适配与能量加权（PASS，方案晋升）
└── stage8_compact_3d/              历史阶段编号；正式代码已迁至根目录pct3d_reconstruction
```

每个阶段使用一个统一入口，并将该阶段的完整性检查直接集成在计算流程中，不再
额外创建`validate_stage*.py`。

## 当前数据状态

截至2026-08-09，本轮企业实践已经结束，阶段代码、QC、图表和总结均保留在本目录。为给Stage 8三维
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

Stage 7B已完成。第一批从D1六平面ROOT按`(RunID, EventID)`在过滤前固定
80/10/10划分，完成噪声源分离、训练集噪声标定和10%候选筛选。解析逆方差、
经验逆方差、Huber及其组合均未改善等权quadratic基线；程序按预注册门槛跳过
80%正式双重建和锁定测试，测试集始终未打开。最终决定为`NO_PROMOTION`，
继续使用阶段4等权数据项。完整结果见
`stage7b_noise_robustness/qc/stage7b_summary.md`。

Stage 7C已完成单命令、可断点续跑全流程。它复用Stage 7的100%正式结果，在
局部3σ过滤后按`(RunID, EventID)`构造严格嵌套的50%、25%和10%子集，完整比较
理想参考面、连续硅hit和0.2 mm位置加1%出射能量噪声。组合噪声的25%和10%
另做两个抽样种子复核。所有通量使用阶段4冻结参数及各自的DDB-FDK初值；D1
没有DoseActor，因此只报告通量而不换算mGy。三种测量条件的推荐最低有效通量
均为25%，即`225 protons/mm²/projection`；10%均进入明显重建失稳。完整结果见
`stage7c_fluence_sensitivity/qc/stage7c_summary.md`。

Stage 8B已经完成。等总质子数下，`360角度×每角度20%`相对`720角度×每角度10%`将水区标准差和模体RMSE分别降低77.53%和49.22%。最终低通量专用配置采用360角度、每角度20%、0.5 mm低通初值上采样、18子集、初始松弛0.25、衰减0.1、Huber-TV `β=0.05`并在第2轮停止；锁定测试水区标准差降低58.25%，模体RMSE降低4.84%，但铝平台误差有所恶化。位置误差仍是主要退化来源，能量逆方差权重没有收益，正式数据项继续使用等权quadratic。完整结果见`stage8b_low_fluence_adaptation/qc/stage8b_summary.md`。

Stage 8在仓库根目录独立工程`pct3d_reconstruction/`完成首轮正式运行。三维数据处理、CPU/CUDA MLP一致性和严格伴随算子通过，但3轮重建时10--14 mm材料球MAPE为37.03%。随后Stage 8C完成坐标、旋转、求交、覆盖、匹配闭环和松弛策略诊断，确认首轮主要问题是严重欠收敛及松弛因子衰减过快。固定松弛0.15、18子集、30轮、`β=0`的全量复算在修正性测试中取得大材料球MAPE 0.5969%、铝球误差−0.8221%和测试WEPL RMSE 1.8909 mm，形成当前三维体素性能基线。严格合成闭环0.01 mm门槛仍未达到，测试集也不是全新盲测。Stage 9在本轮实践中没有启动。

本轮实践已经结束。跨阶段最终结论、正式算法和未来恢复研究时的候选路线统一见
[`../project_overview.md`](../project_overview.md)。

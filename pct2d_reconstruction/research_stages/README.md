# 研究阶段工作区

本目录集中保存阶段性研究代码、只读分析、QC、图表和技术总结，避免在
`pct2d_reconstruction/`顶层为每个实验单独新增目录。大型ROOT、pairs、DDB和
重建图像仍只写入仓库根目录的`data/`。

```text
research_stages/
├── stage1_material_calibration/    S6材料能量、有效RSP和Air WEPL标定（PASS）
├── stage2_diagnostic_phantoms/     S2--S5边界、材料与分辨率诊断（PASS）
├── stage3_robust_weighting/        稳健过滤、WEPL噪声与权重（PASS，保留基线）
└── stage4_iterative_optimization/  固定MLP迭代优化（进行中：子集筛选）
```

每个阶段使用一个统一入口，并将该阶段的完整性检查直接集成在计算流程中，不再
额外创建`validate_stage*.py`。

阶段3完成了过滤前80/10/10划分、稳健过滤、噪声校准、加权OS-SART及锁定测试
确认。没有候选达到预设晋升门槛，因此后续正式基线仍为局部3σ过滤和等权
OS-SART。详细结论见
`stage3_robust_weighting/qc/stage3_summary.md`。

阶段4已完成松弛调度、Huber数据损失和Huber-TV筛选，当前冻结
`λ0=0.25`、衰减`0.2`、quadratic数据项、固定`β=0.0125`和5 epoch。18/36
子集比较正在运行；测试集仍处于锁定状态，尚不能把这些验证集结果称为最终方法
提升。

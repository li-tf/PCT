# 二维pCT仿真、重建与评价流水线

本目录保存pCT实验的代码、配置、QC摘要、评价结果和报告。活动中的大型ROOT、
MHD、RAW以及重建检查点统一位于仓库根目录的`data/`；已经完成且短期不用的
历史数据按批次移至移动硬盘，不再把所有实验数据长期堆放在工作区。

## 目录职责

```text
pct2d_reconstruction/
├── current_research_summary.md  S1--S6、阶段0--7及当前最优算法总览
├── experiments/                 实验编号与代码/数据路径映射
├── simulation/                  Windows OpenGATE仿真包及仿真QC
├── preprocessing/               配对、3σ过滤和Schulte MLP DDB投影
├── analytic_reconstruction/     no-Hann DDB-FDK与200 MeV RSP真值
├── iterative_reconstruction/    GPU MLP OS-SART + Huber-TV
├── evaluation/                  冻结清单、固定划分和统一RSP/WEPL评价
├── research_stages/             阶段1起的研究性算法、评价与阶段报告
├── report/                      单次实验报告、跨阶段总结与最终PPT
└── archive_batch1_20260730_record.md
                                第一批冷数据结构、规模和恢复说明
```

当前本地重点保留的数据为：

```text
data/
├── simulation_data/
│   ├── results0717_s1/s4/s5...
│   └── results0728_stage6a/stage6b...
├── preprocessing_data/
│   ├── results0717_s1/s4/s5.../stage6b_calibrated/
│   └── results0718_d1.../stage7/full/
└── reconstruction_data/
    ├── results0717_s1/s4/s5.../stage6b_calibrated/
    └── results0718_d1.../stage7/full/
```

`experiments/experiment0716.json`将`simulation0716`、`results0716`和
`report0716`绑定为同一实验。Linux入口统一使用`--experiment 0716`，不依赖当前
工作目录推断数据位置。S1--S6是研究阶段使用的诊断数据，不冒充results0716正式
基线；D1通过Windows D盘只读挂载完成阶段7，compact-3D等待阶段8正式处理。

`results0716`、S2/S3、MLP真实轨迹pilot以及`test0707/0710/0713`已进入第一批
冷归档。目录结构、逐项字节数和恢复方法见
[`archive_batch1_20260730_record.md`](archive_batch1_20260730_record.md)。
下面的历史复现命令只有在相应数据恢复到原路径后才能运行。

## results0716历史冻结状态

| 阶段 | 状态 | 正式结果 |
|---|---|---|
| Windows OpenGATE仿真 | PASS | 720角度，每角度450,000个200 MeV质子 |
| primary-only配对 | PASS | 284,021,915条pairs |
| 3σ过滤 | PASS | 244,217,799条pairs，保留85.986% |
| Schulte MLP DDB投影 | PASS | 720幅`500×2×500 @ 0.5 mm`投影 |
| 解析重建 | PASS | no-Hann DDB-FDK，`2100×2100 @ 0.1 mm` |
| GPU迭代重建 | PASS | 全量、18子集、3 epoch、Huber-TV |
| 阶段0统一评价 | PASS | 固定90/10划分和五个检查点验证WEPL |
| report0716 | PASS | 仿真、预处理、解析与迭代完整报告 |

实验注册文件中的迭代`epochs=10`是原计划值；results0716实际完成并冻结的正式
结果为3 epoch，运行摘要是执行配置的权威来源。

## 研究路线当前状态

| 阶段 | 状态 | 冻结结论 |
|---|---|---|
| 1 材料能量标定 | PASS，历史口径 | 保留`I=78 eV`供历史复现；新正式重建改用Stage 6B标定模型 |
| 2 诊断模体 | PASS | 建立Air、边界、材料MAPE和MTF基线 |
| 3 稳健过滤与权重 | PASS，保留基线 | 局部3σ与等权优于或等效于候选方法 |
| 4 固定MLP迭代优化 | PASS，方案晋升 | 冻结`λ0=0.25`、衰减`0.2`、quadratic、固定Huber-TV `β=0.0125`、5 epoch和18子集；锁定测试通过 |
| 5 非均匀MLP | PASS，保留阶段4 | 真实轨迹上未证明当前非均匀MLP有稳定收益 |
| 6 高级先验 | PASS，保留阶段4 | TGV、自适应TV和方向TV未形成更优综合权衡 |
| 6A 虚拟MLIC真值 | PASS | 24-case能量扫描及200 MeV高统计完成；MLIC主参考和三个重点场景重评已冻结 |
| 6B 独立WEPL标定 | PASS，模型晋升 | 锁定测试和S2/S3门控通过，S4大柱MAPE降至0.255% |
| 7 探测器效应 | PASS | 连续硅hit仅使RMSE增加2.21%；0.2 mm位置与1%能量噪声组合使RMSE增加42.73% |
| 7B 噪声稳健重建 | PLANNED | 分离D1位置/能量噪声并标定异方差数据项 |
| 7C 通量敏感性 | PLANNED | 使用嵌套质子子集建立质量—通量曲线 |
| 8 紧凑三维 | READY | 接入compact-3D并执行三维体素算子验证 |
| 9 3D Gaussian | PLANNED | Stage 8之后开展单场景可行性研究 |

阶段性算法不会自动替换results0716成熟入口。只有完整训练—验证—锁定测试通过后，
才在阶段报告中决定是否推荐晋升。

## 主要命令

```bash
# 以下results0716命令需要先从第一批冷归档恢复数据
# 预处理（已有正式结果，通常不应重算）
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \
  --experiment 0716 --stage all --jobs 4

# no-Hann解析重建
.venv-gate/bin/python pct2d_reconstruction/analytic_reconstruction/run_analytic_reconstruction.py \
  --experiment 0716

# 精确复现results0716的3轮迭代基线
.venv-gate/bin/python pct2d_reconstruction/iterative_reconstruction/run_iterative_reconstruction.py \
  --experiment 0716 --epochs 3

# 只读复核阶段0冻结基线
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action verify

# 重现阶段1的S6材料—能量标定
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage1_material_calibration/run_material_energy_analysis.py \
  --force

# 执行阶段2的S2--S5诊断模体处理与评价
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage2_diagnostic_phantoms/run_stage2.py \
  --action all --jobs 4

# 复核阶段3最终决定
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage3_robust_weighting/run_stage3.py \
  --action report --datasets s1,s3,s5

# 复核Stage4正式总结（不重新运行GPU重建）
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action report --datasets s1,s2,s3,s4,s5

# 生成实验报告
.venv-gate/bin/python pct2d_reconstruction/report/build_report.py \
  --experiment 0716 --force

# 查看Stage 6B、7、8门控状态
.venv-gate/bin/python pct2d_reconstruction/research_stages/stage6b_wepl_calibration/run_stage6b.py --action status
.venv-gate/bin/python pct2d_reconstruction/research_stages/stage7_detector_effects/run_stage7.py --action status
.venv-gate/bin/python pct2d_reconstruction/research_stages/stage8_compact_3d/run_stage8.py --action status
```

各计算入口完成后直接写本阶段QC，不使用独立的`validate_stage*.py`。预处理和重建
入口默认拒绝覆盖已有正式结果；只有明确需要重算时才使用`--force`。阶段0评价的
`--force`只覆盖评价模块自身生成的清单、掩码和表格，不改写原始数据或重建结果。

## 推荐阅读顺序

1. `current_research_summary.md`：S1--S6、真实轨迹pilot、阶段0--7、当前最优算法及
   外部性能定位的阶段性总览；
2. `reconstruction_principles.md`：数据处理、MLP、DDB、解析和迭代重建原理；
3. `report/report0716/report0716_summary_report.md`：results0716完整实验报告；
4. `report/research_stages_summary/research_stages_summary.md`：
   基线之后阶段0--7的综合结论与当前进展；
5. `evaluation/baselines/results0716/baseline_summary.md`：冻结后的统一基线；
6. `research_stages/stage1_material_calibration/qc/results0717_s6_material_energy_scan/stage1_summary.md`：
   阶段1的材料能量、有效RSP与Air WEPL结论；
7. `research_stages/stage2_diagnostic_phantoms/qc/stage2_summary.md`：
   S2--S5的边界、材料定量与空间分辨率诊断；
8. `research_stages/stage3_robust_weighting/qc/stage3_summary.md`：
   稳健过滤、噪声模型和权重的完整负结果；
9. `research_stages/stage4_iterative_optimization/qc/stage4_summary.md`：
   固定MLP迭代参数优化和锁定测试结果；
10. `research_stages/stage5_inhomogeneous_mlp/qc/stage5_summary.md`：
   非均匀MLP真实轨迹上限实验及保留水MLP的决定；
11. `research_stages/stage6_advanced_priors/qc/stage6_summary.md`：
   TGV、自适应TV和方向TV的验证结果；
12. `pct_performance_benchmarks.md`：文献、商业系统与当前项目的性能参照；
13. `research_stages/stage7_detector_effects/qc/stage7_summary.md`：
   四层硅跟踪器、位置分辨率与参数化能量噪声结果；
14. 本地私有目录`report/final_presentations/`：最终进展汇报和成果总结PPT，
    因包含个人信息而整体排除在Git版本控制之外；
15. `archive_batch1_20260730_record.md`：冷归档范围与恢复说明；
16. `future_research_plan.md`：阶段0至阶段8研究路线与完成记录。

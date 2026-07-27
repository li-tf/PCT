# 二维pCT仿真、重建与评价流水线

本目录保存二维pCT实验的代码、配置、QC摘要、评价结果和报告。大型ROOT、MHD、
RAW以及重建检查点统一位于仓库根目录的`data/`；历史目录`test0713/`保持不变，
不作为新流程的运行目录。

## 目录职责

```text
pct2d_reconstruction/
├── experiments/                 实验编号与代码/数据路径映射
├── simulation/                  Windows OpenGATE仿真包及仿真QC
├── preprocessing/               配对、3σ过滤和Schulte MLP DDB投影
├── analytic_reconstruction/     no-Hann DDB-FDK与200 MeV RSP真值
├── iterative_reconstruction/    GPU MLP OS-SART + Huber-TV
├── evaluation/                  冻结清单、固定划分和统一RSP/WEPL评价
├── research_stages/             阶段1起的研究性算法、评价与阶段报告
└── report/                      各实验的正式报告与图表生成器
```

数据目录对应为：

```text
data/
├── simulation_data/results0716/
├── simulation_data/results0717_s1...s6/
├── simulation_data/results0724_mlp_truth_pilot/
├── preprocessing_data/results0716及results0717_s1...s5/
└── reconstruction_data/results0716及results0717_s1...s5/
```

`experiments/experiment0716.json`将`simulation0716`、`results0716`和
`report0716`绑定为同一实验。Linux入口统一使用`--experiment 0716`，不依赖当前
工作目录推断数据位置。S1--S6是研究阶段使用的诊断数据，不冒充results0716正式
基线；D1和compact-3D已在工作站完成，但大型ROOT尚未复制到当前`data/`。

## results0716当前状态

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
| 1 材料能量标定 | PASS | 保留`I=78 eV`主口径，同时用有效RSP解释能量相关差异 |
| 2 诊断模体 | PASS | 建立Air、边界、材料MAPE和MTF基线 |
| 3 稳健过滤与权重 | PASS，保留基线 | 局部3σ与等权优于或等效于候选方法 |
| 4 固定MLP迭代优化 | 进行中 | 已冻结`λ0=0.25`、衰减`0.2`、quadratic、固定Huber-TV `β=0.0125`和5 epoch；正在比较18/36子集 |

阶段性算法不会自动替换results0716成熟入口。只有完整训练—验证—锁定测试通过后，
才在阶段报告中决定是否推荐晋升。

## 主要命令

```bash
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

# Stage4必须按README分批运行；当前步骤为18/36子集比较
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage4_iterative_optimization/run_stage4.py \
  --action subset-screen --datasets s2,s4,s5 --device 0

# 生成实验报告
.venv-gate/bin/python pct2d_reconstruction/report/build_report.py \
  --experiment 0716 --force
```

各计算入口完成后直接写本阶段QC，不使用独立的`validate_stage*.py`。预处理和重建
入口默认拒绝覆盖已有正式结果；只有明确需要重算时才使用`--force`。阶段0评价的
`--force`只覆盖评价模块自身生成的清单、掩码和表格，不改写原始数据或重建结果。

## 推荐阅读顺序

1. `reconstruction_principles.md`：数据处理、MLP、DDB、解析和迭代重建原理；
2. `report/report0716/report0716_summary_report.md`：当前实验完整报告；
3. `evaluation/baselines/results0716/baseline_summary.md`：冻结后的统一基线；
4. `research_stages/stage1_material_calibration/qc/results0717_s6_material_energy_scan/stage1_summary.md`：
   阶段1的材料能量、有效RSP与Air WEPL结论；
5. `research_stages/stage2_diagnostic_phantoms/qc/stage2_summary.md`：
   S2--S5的边界、材料定量与空间分辨率诊断；
6. `research_stages/stage3_robust_weighting/qc/stage3_summary.md`：
   稳健过滤、噪声模型和权重的完整负结果；
7. `pct_performance_benchmarks.md`：文献、商业系统与当前项目的性能参照；
8. `future_research_plan.md`：阶段0至阶段8研究路线与完成记录。

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
├── research_stages/             阶段1起的研究性分析、评价与阶段报告
└── report/                      实验报告、图表和后续研究计划
```

数据目录对应为：

```text
data/
├── simulation_data/results0716/
├── preprocessing_data/results0716/
└── reconstruction_data/results0716/
```

`experiments/experiment0716.json`将`simulation0716`、`results0716`和
`report0716`绑定为同一实验。Linux入口统一使用`--experiment 0716`，不依赖当前
工作目录推断数据位置。

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
6. `future_research_plan.md`：阶段0至阶段8研究路线与完成记录。

# 阶段1：材料能量、WEPL与有效RSP标定

本目录分析`results0717_s6_material_energy_scan`的52组Water、Aluminium和Air
薄板仿真。入口直接读取工作站返回的ROOT与QC，只生成代码侧CSV、JSON、PNG和
Markdown，不修改或复制原始ROOT。

主要任务：

- 对`TrackID=1`主质子按EventID配对入口和出口状态；
- 核对ROOT、工作站metadata和已有汇总；
- 使用正式`I=78 eV`水Bethe--Bloch LUT计算逐质子WEPL；
- 同时报告原始分布、中位数与3-MAD稳健核心，避免核反应尾部支配结论；
- 评价Water LUT一致性、Aluminium有效RSP和Air单位长度WEPL；
- 建立供后续S1/S3/D1使用的能量相关Air WEPL插值模型；
- 将阶段结论写入`stage1_summary.md`。

运行：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage1_material_calibration/run_material_energy_analysis.py
```

默认拒绝覆盖已有完整产物。需要明确重算时：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage1_material_calibration/run_material_energy_analysis.py \
  --force
```

输出位于：

```text
pct2d_reconstruction/research_stages/stage1_material_calibration/
└── qc/results0717_s6_material_energy_scan/
    ├── input_manifest.json
    ├── root_integrity.csv
    ├── case_metrics.csv
    ├── water_lut_consistency.csv
    ├── aluminium_effective_rsp.csv
    ├── ionization_potential_sensitivity.csv
    ├── ionization_potential_objective.csv
    ├── material_wepl_fits.csv
    ├── air_wepl_model.json
    ├── chart_map.json
    ├── stage1_summary.json
    ├── stage1_summary.md
    └── figures/
```

S6不是CT重建数据，不执行80/10/10图像重建划分。置信区间使用EventID确定性分块
估计；正式CT数据仍须在DDB或迭代重建前划分训练、验证和锁定测试集。

默认`--root-backend auto`优先使用PyROOT读取数组，并用uproot检查树结构和分支。
当前环境中的`uproot 5.7.5 + NumPy 2.5`直接读这些数组明显更慢；如需诊断后端
差异，可显式指定`--root-backend pyroot`或`--root-backend uproot`。

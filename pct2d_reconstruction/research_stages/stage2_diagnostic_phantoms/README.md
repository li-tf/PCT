# 阶段2：诊断模体处理与评价

阶段2已于2026-07-23完成并通过验收。正式结论见
[`qc/stage2_summary.md`](qc/stage2_summary.md)。

本阶段统一处理四个100,000质子/角度的720角度pilot：

| 数据集 | 场景 | 主要问题 |
|---|---|---|
| S2 | Vacuum中的均匀水圆柱 | 水平台、FOV、Hann和支撑域 |
| S3 | Air中的均匀水圆柱 | 外部Air WEPL及边界效应 |
| S4 | Air中的五材料三半径定量模体 | 材料平台误差与径向趋势 |
| S5 | Air中的线对和SpineBone斜边 | 线对可视性、fMTF50和fMTF10 |

S2/S3另外执行3 epoch GPU迭代对照；S4/S5在本阶段完成no-Hann
DDB-FDK诊断，依据材料误差和MTF结果再决定是否增加通量和迭代重建。

统一入口：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage2_diagnostic_phantoms/run_stage2.py \
  --action all --jobs 4
```

支持`freeze`、`preprocess`、`project`、`analytic`、`iterative`、`holdout`、
`evaluate`和`all`。其中`holdout`只计算固定validation/test质子的WEPL指标，
不生成阶段总结；`evaluate`复用已有holdout结果并生成图表和总结。`--force`
只覆盖阶段2派生产物，原始ROOT和工作站仿真QC始终只读。

当前WSL2主机的Windows驱动最高支持CUDA 12.7。GPU环境固定为
`cupy-cuda12x==13.3.0`和`cuda-toolkit==12.6.3`；不要直接升级到链接CUDA
12.8/12.9的CuPy wheel，否则会出现`cudaErrorInsufficientDriver`。

大型pairs、DDB和MHD写入：

```text
data/preprocessing_data/results0717_s2_water_vacuum_pilot/
data/preprocessing_data/results0717_s3_water_air_pilot/
data/preprocessing_data/results0717_s4_material_calibration_air_pilot/
data/preprocessing_data/results0717_s5_resolution_air_pilot/
data/reconstruction_data/results0717_s2_water_vacuum_pilot/
data/reconstruction_data/results0717_s3_water_air_pilot/
data/reconstruction_data/results0717_s4_material_calibration_air_pilot/
data/reconstruction_data/results0717_s5_resolution_air_pilot/
```

代码侧只保留配置、表格、图像、输入清单和阶段总结：

```text
qc/
├── input_manifest.json
├── split_*.json
├── analytic_variant_metrics.csv
├── material_metrics.csv
├── slanted_edge_mtf.csv
├── line_pair_metrics.csv
├── radial_profiles.csv
├── holdout_wepl_metrics.csv
├── figures/
└── stage2_summary.md
```

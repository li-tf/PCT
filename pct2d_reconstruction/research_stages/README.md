# 研究阶段工作区

本目录集中保存阶段性研究代码、只读分析、QC、图表和技术总结，避免在
`pct2d_reconstruction/`顶层为每个实验单独新增目录。大型ROOT、pairs、DDB和
重建图像仍只写入仓库根目录的`data/`。

```text
research_stages/
├── stage1_material_calibration/    S6材料能量、有效RSP和Air WEPL标定（PASS）
└── stage2_diagnostic_phantoms/     S2--S5边界、材料与分辨率诊断（PASS）
```

每个阶段使用一个统一入口，并将该阶段的完整性检查直接集成在计算流程中，不再
额外创建`validate_stage*.py`。

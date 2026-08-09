# Stage 8迁移说明

正式三维代码已经从二维研究目录中分离，统一位于：

```text
pct3d_reconstruction/
```

本目录仅保留历史阶段编号和跳转说明，不再包含可执行Python入口或配置。正式
命令、数据布局、算法和QC说明见
[`pct3d_reconstruction/README.md`](../../../pct3d_reconstruction/README.md)。

首轮Stage 8已经完成。三维工程链和伴随算子通过，但3轮结果的大材料球MAPE为37.03%，当时状态为`PIPELINE PASS / PERFORMANCE FAIL`。Stage 8C随后完成系统诊断和全量复算，固定松弛因子0.15运行30轮后，大材料球MAPE降至0.5969%，测试WEPL RMSE为1.8909 mm，形成当前三维体素性能基线。Stage 8C测试属于修正性复核，严格合成闭环0.01 mm门槛仍未达到。

本轮企业实践已于2026-08-09结束，Stage 9没有启动。首轮结果见[`stage8_summary.md`](../../../pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8_summary.md)，最终Stage 8C结果见[`fixed015_test_summary.md`](../../../pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/fixed015_test_summary.md)，跨阶段结论见[`project_overview.md`](../../project_overview.md)。

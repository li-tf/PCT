# Stage 8迁移说明

正式三维代码已经从二维研究目录中分离，统一位于：

```text
pct3d_reconstruction/
```

本目录仅保留历史阶段编号和跳转说明，不再包含可执行Python入口或配置。正式
命令、数据布局、算法和QC说明见
[`pct3d_reconstruction/README.md`](../../../pct3d_reconstruction/README.md)。

截至2026-08-03，首轮Stage 8已经完成。三维工程链和伴随算子通过，但大材料球
MAPE为`37.03%`，状态为`PIPELINE PASS / PERFORMANCE FAIL`；Stage 9暂缓，
下一步先诊断坐标、路径覆盖和收敛性。正式结果见
[`stage8_summary.md`](../../../pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8_summary.md)。

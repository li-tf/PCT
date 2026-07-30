# Stage 8：紧凑三维pCT

状态：**READY，尚未执行正式三维预处理和重建**。compact-3D原始ROOT已经完成
工作站仿真，目前保存在外部存储；正式运行前需将其挂载为稳定的只读路径，并
为三维pairs、检查点和临时缓存预留本地活动空间。

冻结规格为`240×80×240 @ 0.5 mm`、360角度、三维Schulte MLP、三线性
8邻域严格配对转置、18子集和最多3 epoch。材料评价使用Stage 6A MLIC参考，
能量转换使用Stage 6B冻结模型。

输入与标定门控：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage8_compact_3d/run_stage8.py \
  --action preflight
```

正式实现前必须依次通过三维算子内积、球体方向、支撑域、显存和batch smoke
test。当前入口不会在算子尚未实现时误启动全量任务；正式处理预计CPU预处理
2--5小时、GPU重建3--8小时。Stage 8应使用Stage 6B的
`g4_water_calibrated`能量模型和Stage 4冻结算法作为二维先验，不在锁定测试
数据上重新调参。三维正式结果生成后，再决定哪些Stage 7筛选缓存可转入下一批
冷归档。

# 解析重建：no-Hann DDB-FDK

本目录只保留当前验证成熟的no-Hann DDB-FDK主链。输入不是普通二维X射线正弦图，
而是包含横向探测器坐标和MLP深度坐标的DDB投影；`pctfdk`使用与该DDB定义配套的
几何加权、Ramp滤波和距离驱动DDB反投影器。

`results0716`的大型DDB和解析重建已于2026-07-30移入第一批冷归档，代码侧QC和
`report0716`仍保留。运行本页命令前应按
[`archive_batch1_20260730_record.md`](../archive_batch1_20260730_record.md)
恢复原路径。当前正式三场景结果位于S1/S4/S5各自的
`stage6b_calibrated/`目录。

## results0716正式配置

- 720个角度：`0, 0.5, ..., 359.5°`；
- SID/SDD：1000/1110 mm；
- 输入投影：`500×2×500 @ 0.5×1×0.5 mm`；
- 输出网格：`2100×1×2100 @ 0.1×1×0.1 mm`；
- Hann参数：0，即no-Hann；
- 角度关系使用已验证的`r=R(θ)Fs`深度反射约定。

入口同时根据仿真材料和几何生成200 MeV参考RSP真值。真值采用8×8子像素采样，
用于圆柱和5 mm铝柱边界的部分容积计算；重建评价统一使用RSP，不把组成RED作为
正式误差口径。

## 正式结果

results0716解析重建的水区均值/标准差为`1.013718/0.009720`，模体RSP RMSE为
`0.045075`，铝平台恢复率为`98.8901%`，ROI CNR为`106.92`，铝柱10%–90%
边缘宽度中位数为`1.1367 mm`。几何生成耗时11.31 s，FDK耗时181.37 s。

## 命令和产物

```bash
.venv-gate/bin/python pct2d_reconstruction/analytic_reconstruction/run_analytic_reconstruction.py \
  --experiment 0716
```

- 重建：`data/reconstruction_data/results0716/analytic/recon/`；
- RSP真值：`data/reconstruction_data/results0716/analytic/truth/`；
- 几何：`data/reconstruction_data/results0716/analytic/geometry.xml`；
- QC：`analytic_reconstruction/qc/results0716/`。

入口会检查720幅投影、网格一致性、有限性和RSP关键指标。已有正式结果默认拒绝
覆盖；确认需要重建时加入`--force`。

本解析入口仍是快速基线和迭代初值生成器；当前最终定量结果采用Stage 6B
`g4_water_calibrated` WEPL和阶段4冻结的5 epoch迭代配置。

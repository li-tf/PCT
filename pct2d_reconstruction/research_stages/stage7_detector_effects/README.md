# Stage 7：Air与四层硅跟踪器效应

状态：**PASS（D1_DETECTOR_EFFECTS_CHARACTERIZED，2026-07-30）**。

本阶段直接读取D1的720组六平面ROOT，使用阶段6B冻结的
`g4_water_calibrated`模型和阶段4冻结的重建参数，不修改成熟预处理及重建代码。

## 测量层级

| 变体 | 位置/方向 | 能量 |
|---|---|---|
| `ideal_reference` | `z=−110/+110 mm`理想参考面 | 理想 |
| `continuous_hits` | 四层连续硅hit直线拟合 | 理想 |
| `strip_0p1/0p2/0p5mm` | 每层hit加入相应高斯位置分辨率后拟合 | 理想 |
| `energy_0p5/1/2pct` | 0.2 mm hit分辨率 | 出射能量加入相对高斯噪声 |

能量噪声只施加到出射能量，入射能量视为已知束流状态。出现`Eout≥Ein`的非物理
测量不会被截断成零WEPL，而是作为无效测量剔除并计入效率。

理想参考面使用其全部可配对事件；hit变体要求同一主质子遍历四个硅层及两个参考
面。因此图像变化同时包含硅散射、方向拟合误差和跟踪接受率损失。每个变体随后
独立执行局部3σ过滤、Geant4一致WEPL转换和圆柱外Air扣除。

## 计算预算

为避免8套配置全部进行昂贵的全量重建：

1. 8套变体均确定性抽取10%质子，完成720角度、3 epoch筛查；
2. 预注册的`ideal_reference`、`continuous_hits`和`energy_1pct`再使用全量质子、
   5 epoch重建；
3. 不根据测试图像临时挑选更好看的配置。

输出位于：

```text
data/preprocessing_data/results0718_d1_air_tracker_full/stage7/
data/reconstruction_data/results0718_d1_air_tracker_full/stage7/
```

正式运行实际生成的本地预处理数据约55 GB、重建结果约1.2 GB；其中`full/`
三套正式变体必须保留，`screen/`八套10%筛选用于追溯参数化灵敏度。约115 GB
原始ROOT仍位于Windows D盘，可通过下述路径只读挂载，不需要复制进仓库。

## 单命令运行

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage7_detector_effects/run_stage7.py \
  --action all \
  --data-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 \
  --device 0
```

另开终端查看跨任务状态：

```bash
watch -n 30 \
  ".venv-gate/bin/python pct2d_reconstruction/research_stages/stage7_detector_effects/run_stage7.py --action status"
```

`all`依次执行ROOT门控、六平面配对与数字化、DDB、no-Hann解析初值、GPU重建
和报告。已完成的角度、DDB、解析结果和完整变体会跳过；若在某个epoch中断，
该变体会从epoch 1重新开始。

`all`会在读取115 GB ROOT前先执行CuPy GPU预检。若WSL在Windows睡眠或驱动更新
后返回`cudaErrorInsufficientDriver`或`GPU access blocked`，先在PowerShell执行
`wsl --shutdown`，重新打开Debian后再运行同一命令。

本次完整执行预算约15--26小时，实际完成后可直接读取代码侧运行记录：

- ROOT读取、配对、拟合和过滤：1--3小时；
- 11组DDB及解析初值：1--3小时；
- 8套10%/3 epoch筛查：约2--4 GPU小时；
- 3套全量/5 epoch重建：约10--14 GPU小时；
- 汇总：数分钟。

不要使用`--force`，除非明确要删除并重算Stage 7自身的全部大数据产物。

## 已完成结果

720个角度、8套10%/3 epoch筛选和3套全量/5 epoch重建全部通过QC。全量结果为：

| 配置 | 水均值 | 水标准差 | 模体RMSE | 铝平台RSP | 中位CNR |
|---|---:|---:|---:|---:|---:|
| `ideal_reference` | 0.999447 | 0.001826 | 0.039868 | 2.069840 | 551.14 |
| `continuous_hits` | 0.999407 | 0.001888 | 0.040748 | 2.070807 | 516.90 |
| `energy_1pct` | 0.999739 | 0.002594 | 0.056903 | 2.044424 | 371.80 |

连续硅hit相对理想参考的RMSE只增加`2.21%`；`energy_1pct`同时包含`0.2 mm`
位置噪声，其RMSE增加`42.73%`。该能量结果不是物理能量探测器性能，只能作为
参数化灵敏度边界。详细解释和完整筛选表见
[`qc/stage7_summary.md`](qc/stage7_summary.md)。

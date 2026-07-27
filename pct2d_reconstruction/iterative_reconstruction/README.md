# GPU迭代重建：MLP OS-SART + Huber-TV

正式算法直接读取过滤后list-mode pairs，使用入口/出口位置、方向和能量重建，
不把DDB投影交给普通直线Joseph投影器。

## 算法

- `I=78 eV`水Bethe–Bloch LUT：入口/出口能量转换为目标WEPL；
- Schulte MLP：在半径100 mm水圆柱内计算弯曲路径；
- 0.1 mm路径采样和双线性像素权重：构成严格配对的正投影/转置反投影；
- 18子集OS-SART：行归一化、子集列归一化和逐epoch松弛衰减；
- no-Hann FDK初值、RSP非负约束和100 mm圆形支撑域；
- 每个epoch后执行Huber-TV近端更新。

## results0716实际执行配置

| 参数 | 数值 |
|---|---:|
| 每轮质子数 | 244,217,799（全量） |
| 图像网格 | `2100×2100 @ 0.1 mm` |
| MLP步长 | 0.1 mm |
| 子集 | 18 |
| epoch | **3** |
| 初始松弛因子/衰减 | 0.25 / 0.2 |
| Huber-TV权重/过渡点 | 0.0125 / 0.002 |
| GPU | RTX 4060 Laptop GPU |
| 总耗时 | 11,559.33 s（3 h 12 min 39 s） |

`experiment0716.json`中的10 epoch是原计划值，不是已完成基线。运行摘要中的3 epoch
是results0716正式执行配置；如需精确复现，必须显式传入`--epochs 3`：

```bash
.venv-gate/bin/python pct2d_reconstruction/iterative_reconstruction/run_iterative_reconstruction.py \
  --experiment 0716 --epochs 3
```

重建和每轮检查点写入
`data/reconstruction_data/results0716/iterative/recon/`，训练过程残差、正则化历史和
RSP指标写入`iterative_reconstruction/qc/results0716/`。第3轮水区标准差为
`0.002453`，模体RSP RMSE为`0.041956`，铝平台恢复率为`98.7113%`，ROI CNR为
`399.53`，边缘宽度中位数为`1.1179 mm`。

阶段0在固定10%验证集上另行得到第3轮WEPL RMSE `2.62098 mm`。该值是固定图像
检查点的独立验证残差，不等同于迭代日志中更新前的在线训练残差。验证结果位于
`evaluation/baselines/results0716/`。

已有重建默认拒绝覆盖。重新运行全量3轮会耗费约3小时，使用`--force`前应确认
确实需要替换正式检查点和迭代QC。

固定MLP下的参数优化位于
`research_stages/stage4_iterative_optimization/`，检查点写入各研究数据集的
`stage4/`目录，不会覆盖本目录的results0716正式基线。阶段4完成锁定测试前，
这里列出的3 epoch配置仍是正式方法。

# GPU迭代重建：MLP OS-SART + Huber-TV

正式算法直接读取过滤后list-mode pairs，使用入口/出口位置、方向和能量重建，
不把DDB投影交给普通直线Joseph投影器。

当前推荐的新数据入口是`run_best_reconstruction.py`。
`run_iterative_reconstruction.py`继续用于严格复现历史`results0716`，但其大型
输入和检查点已进入第一批冷归档，运行前需按
[`archive_batch1_20260730_record.md`](../archive_batch1_20260730_record.md)
恢复原路径。

## 算法

- 版本化水射程LUT：默认使用历史`I=78 eV` Bethe–Bloch模型；Stage 6B通过后
  可显式选择独立Geant4水板标定模型；
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
| 松弛调度 | `λ_k=0.25/[1+0.2(k-1)]` |
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
`stage4/`目录，不会覆盖本目录的results0716正式基线。阶段4已经通过S1--S5
锁定测试，当前推荐的新数据重建配置为5 epoch、18子集、`λ0=0.25`、衰减0.2、
quadratic数据项和固定`β=0.0125` Huber-TV。阶段5和阶段6均未产生可晋升方案。

Stage 6B随后把独立Geant4水板标定模型晋升为当前正式WEPL口径。S1铝柱、S4
多材料和S5线对卡的可复现输入分别保存在：

```text
data/preprocessing_data/results0717_s1_aluminium_air_full/stage6b_calibrated/
data/preprocessing_data/results0717_s4_material_calibration_air_pilot/stage6b_calibrated/
data/preprocessing_data/results0717_s5_resolution_air_pilot/stage6b_calibrated/
```

对应最终解析和5 epoch迭代结果位于三个数据集的
`data/reconstruction_data/.../stage6b_calibrated/`。

## 当前最优配置的通用入口

对其他已经完成primary配对、局部3σ过滤和no-Hann解析初值的二维数据，使用：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/iterative_reconstruction/run_best_reconstruction.py \
  --run-name my_dataset \
  --pairs-dir data/preprocessing_data/my_dataset/pairs_filtered \
  --initial-image data/reconstruction_data/my_dataset/analytic/recon/recon_ddb_nohann.mhd \
  --output-dir data/reconstruction_data/my_dataset/iterative_best \
  --runs 720 --angle-step-deg 0.5 --device 0
```

冻结参数保存在`best_reconstruction_config.json`。大型检查点写入用户指定的
`--output-dir`，QC写入`iterative_reconstruction/qc/best_runs/<run-name>/`。
若新模体不是半径100 mm、`2100² @ 0.1 mm`网格，可显式调整支撑半径、网格和
路径步长；这些属于几何适配，不会改变冻结的迭代调度和先验参数。

Vacuum数据保持默认的`--air-wepl-slope 0`。若两个参考面之间的圆柱外介质为Air，
还需显式传入阶段1标定值`--air-wepl-slope 0.00114710`；程序只扣除圆柱外路径，
不会把圆柱内材料当作Air。

当前正式研究结果在不改变其余冻结参数的情况下显式选择：

```bash
--wepl-model g4_water_calibrated \
--wepl-calibration \
  pct2d_reconstruction/research_stages/stage6b_wepl_calibration/qc/g4_water_calibrated.json
```

不传参数时仍严格使用`bb78`，所以历史命令不会被静默改变。运行摘要记录模型名、
能量范围和SHA-256，防止不同WEPL口径的检查点混用。

通用入口默认不计算实验特定真值指标；只有输入数据、真值和
`experiment*.json`确实对应时才使用`--with-truth-metrics`。它使用所有输入的
过滤后质子进行最终重建，不重新执行研究阶段的80/10/10参数选择。

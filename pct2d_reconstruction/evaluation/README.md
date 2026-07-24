# 阶段0：冻结基线与统一评价

本目录冻结`results0716`并为后续实验提供唯一的RSP和固定验证WEPL评价入口。
评价模块不改写仿真、预处理或重建结果；`--force`也只覆盖本模块生成的清单、
掩码和表格。

## 固定数据划分

现有过滤后pair没有保存EventID，因此质子身份固定为
`(RunID, filtered_row_index)`。使用种子`20260713`和`splitmix64-v1`规则，将
哈希模10等于0的质子分入验证集，其余为训练集。

- 全部质子：244,217,799；
- 训练集：219,790,828；
- 验证集：24,426,971（10.0021%）；
- 训练和验证集合均覆盖全部720个角度。

bit-packed验证掩码位于
`data/preprocessing_data/results0716/splits/baseline90_10/`，其补集即训练集。
重复生成的720个掩码逐字节一致，原始过滤后pair保持不变。

## 冻结基线与指标

冻结清单覆盖5,777个正式文件、67.77 GiB数据，并记录文件大小、修改时间、
SHA-256、MHD网格、计划/执行配置、Git状态、源码和既有QC哈希。正式基线为：

- no-Hann DDB-FDK；
- 全量、0.1 mm、18子集、3 epoch GPU MLP OS-SART + Huber-TV；
- 200 MeV参考RSP真值。

五个固定检查点共用同一验证集正投影：解析no-Hann、迭代初值、epoch 1、2、3。
第3轮验证WEPL RMSE为`2.62098 mm`，解析no-Hann为`2.65091 mm`。完整验证耗时
490.91 s，RTX 4060 Laptop GPU吞吐约49,758条质子/s。

MTF和路径误差已预留统一字段，但results0716没有专用MTF靶或真实蒙卡轨迹，
因此明确标记为不可用，不进行替代估算。

## 命令

```bash
# 日常只读复核：重新计算全部正式文件和划分哈希
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action verify

# 分步生成
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action freeze
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action split
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action metrics

# 明确需要重建全部评价产物时
.venv-gate/bin/python pct2d_reconstruction/evaluation/run_evaluation.py \
  --experiment 0716 --action all --force
```

`metrics`需要CUDA GPU；`freeze`和`verify`会完整读取约67.8 GiB数据。核心产物位于
`evaluation/baselines/results0716/`，最终验收结果位于
`evaluation/qc/results0716/evaluation_summary.json`。后续报告应直接读取统一CSV，
不得重新定义ROI，也不得把在线训练残差与固定验证残差混用。


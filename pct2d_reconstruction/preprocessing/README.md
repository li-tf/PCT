# 预处理：ROOT到Schulte MLP DDB投影

正式入口为`run_preprocessing.py`。它直接读取
`data/simulation_data/resultsXXXX/run_###/`中的720组入口/出口ROOT分片，不需要
生成额外的合并ROOT。

> 存储状态（2026-07-30）：`results0716`和S2/S3的ROOT及预处理结果已移入第一批
> 冷归档。以下历史命令和路径仍是权威复现接口，但运行前必须按
> [`archive_batch1_20260730_record.md`](../archive_batch1_20260730_record.md)
> 恢复对应数据。S1/S4/S5的Stage 6B正式输入和Stage 7全量数据仍在本机。

## 处理流程

1. **Primary-only配对**：公共`pctpairprotons --no-nuclear`先按
   RunID/EventID/TrackID匹配入口与出口，随后显式选择`TrackID=1`主质子，并把
   位置和方向外推到固定`z=-110/+110 mm`参考面。`--no-nuclear`只保证两侧
   TrackID相同，并不等于primary-only；在Air中，入口面前产生的次级粒子也可能
   穿过两个参考面；
2. **局部3σ过滤**：在`125×2 @ 2 mm`网格内联合检查能损与两个投影方向的
   散射角，去除网格外记录、异常散射和离群能损；
3. **DDB投影**：历史入口采用`I=78 eV`水Bethe–Bloch LUT将入口/出口能量转换为
   WEPL，用Schulte MLP计算弯曲路径，并生成`500×2×500 @ 0.5 mm`距离驱动
   投影。Stage 6B之后的正式研究结果显式使用
   `g4_water_calibrated`模型，不会静默改变历史`bb78`结果。

pair文件采用`N×5×3`的float32布局，保存入口/出口位置、方向和能量等重建字段。
当前格式没有继续保存EventID，因此阶段0固定划分使用
`(RunID, filtered_row_index)`作为稳定身份。

## results0716正式结果

| 阶段 | 输入 | 输出 | 本地耗时 |
|---|---:|---:|---:|
| Primary配对 | 720组ROOT | 284,021,915 | 1,573.47 s |
| 3σ过滤 | 284,021,915 | 244,217,799 | 18.84 s |
| Schulte MLP DDB | 244,217,799 | 720幅投影 | 928.55 s |

过滤后保留率为85.986%，所有输出仍为主质子；DDB投影没有物体内零计数像素，
WEPL方差未发现非有限值或负值。这些耗时来自当前本地存储和并行配置，不应直接
推广到机械硬盘或不同ROOT后端。

## 命令

```bash
# 完整流程
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \
  --experiment 0716 --stage all --jobs 4

# 分阶段执行
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \
  --experiment 0716 --stage pairing --jobs 4
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \
  --experiment 0716 --stage filtering --jobs 4
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \
  --experiment 0716 --stage projection --jobs 4
```

大型pairs和DDB数据写入`data/preprocessing_data/resultsXXXX/`；逐角度CSV和阶段
JSON摘要写入`preprocessing/qc/resultsXXXX/`。已有结果默认不覆盖，重算时必须
显式加入`--force`。`paircuts.py`和`projection.py`是内部算法模块，不是独立入口。

# 数据目录

本目录不保存代码或QC摘要：

- `simulation_data/resultsXXXX`：OpenGATE入口/出口ROOT数据；
- `preprocessing_data/resultsXXXX`：二维或三维pairs、过滤后pairs、划分及DDB投影；
- `reconstruction_data/resultsXXXX`：二维或三维解析/迭代MHD、RAW和检查点。

二维实验的映射、日志和QC位于`pct2d_reconstruction/`；Stage 8三维产物的代码侧
记录位于`pct3d_reconstruction/`。

## 2026-08-09结题时的外置存储状态

由于WSL虚拟磁盘位于Windows O盘，而O盘当前仅约134 GiB可用，以下大型预处理
数据已临时迁出；代码侧QC和总结仍保留在仓库中：

| 数据 | 临时位置 | 当前可用性 |
|---|---|---|
| S1铝柱预处理 | `D:\临时\preprocessing_data\results0717_s1_aluminium_air_full` | D盘连接时可读 |
| S4多材料、S5分辨率预处理 | `F:\临时\preprocessing_data\...` | F盘连接时可读 |
| D1的`stage7/`和`stage7b/` | `E:\preprocessing_data\results0718_d1_air_tracker_full\` | E盘为移动硬盘，当前断开 |

Stage 7C与Stage 8B正式结论及其代码侧QC已经保留，但若重新生成依赖Stage 7全通量图像的报告资产，需要重新连接E盘并恢复原目录映射。Stage 8C已经完成，正式三维pairs、检查点和QC仍位于本地，其原始compact-3D ROOT位于F盘。当前实践周期已经结束，没有正在运行的数据处理或重建任务；未来复算前必须先根据本文件和归档记录核对外部盘挂载、WSL内部空间及O盘宿主空间。

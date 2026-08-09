# 三维pCT重建

本目录是独立于`pct2d_reconstruction/`的三维list-mode重建工程。首个实验为
`results0718_compact_3d_pilot`：360个角度、每角度2,000,000个200 MeV质子，
半径50 mm、轴向长度30 mm的有限水圆柱内包含5个不贯穿轴向的材料球。

> **最终状态（2026-08-09）：本轮企业实践已经结束。Stage 8首轮工程链通过但欠收敛；Stage 8C固定松弛因子0.15、18子集、30轮、`β=0`的全量复算通过预设性能门槛，成为当前三维体素性能基线。当前没有正在运行的三维任务，Stage 9尚未启动。**

## 数据与坐标

原始ROOT只读保存在：

```text
/mnt/f/临时/results0718_compact_3d_pilot
```

F盘上的ROOT分支直接随机读取很慢，因此预处理开始时会将86.4 GiB ROOT顺序暂存
到本机，完成SHA-256和pairs生成后自动删除缓存。正式pairs和检查点写入：

```text
data/preprocessing_data/results0718_compact_3d_pilot/stage8/
data/reconstruction_data/results0718_compact_3d_pilot/stage8/
```

重建采用0°仿真时的scanner坐标，物理坐标顺序为`(x,y,z)`，NumPy内部数组为
`(z,y,x)`。输出网格为`240×80×240 @ 0.5 mm`，即
`120×40×120 mm³`。

本机WSL虚拟磁盘实际由Windows O盘承载，因此空间判断同时检查Linux文件系统和
`/mnt/o`对应的宿主盘余量；不能只根据WSL内部`df`显示的虚拟可用空间判断。
2026-08-01首次运行已经完成86.4 GiB暂存，断点续跑会复用该缓存，预处理成功后
自动删除。删除后空间可被WSL继续复用，但Windows端VHDX文件不一定立即缩小。

## 算法

- primary-only入口/出口EventID配对；
- 过滤前固定80%/10%/10%训练—验证—测试划分；
- 训练集拟合的二维局部网格能损和双方向散射3σ过滤；
- Stage 6B `g4_water_calibrated` WEPL及有限圆柱外Air校正；
- 两个横向方向的Schulte水MLP；
- 0.5 mm路径采样和三线性8邻域严格配对转置；
- 18子集GPU OS-SART、3 epoch、非负与有限圆柱支撑域；
- 小样本筛选`beta=0/0.003/0.006/0.0125`的三维Huber-TV。

首版使用有限水圆柱均匀初值，不实现三维DDB-FDK。测试集只在正则化和epoch
冻结后打开一次。

## 运行

Stage 7C结束后执行：

```bash
.venv-gate/bin/python \
  pct3d_reconstruction/run_stage8.py \
  --action all \
  --raw-root '/mnt/f/临时/results0718_compact_3d_pilot' \
  --jobs 4 \
  --device 0
```

另开终端：

```bash
watch -n 30 \
  ".venv-gate/bin/python pct3d_reconstruction/run_stage8.py --action status"
```

`all`依次执行`preflight → prepare → operator-smoke → screen → confirm →
report`。中断后重新执行同一命令会跳过已通过QC的角度和epoch。正常续跑不要使用
`--force`。

### 2026-08-01断点说明

首次正式运行已完成全部ROOT暂存，并产生349个通过完整性检查的角度；角度87--97
因中断留下11组零字节JSON/MHD/NPZ。当前版本会拒绝这些不完整检查点，只重算
对应11个角度，并对所有新产物使用原子写入。恢复时直接重新执行原`--action all`
命令，不要手工删除正确结果，也不要使用`--force`。

## 验证与产物

CPU测试：

```bash
PYTHONPATH=pct3d_reconstruction \
  .venv-gate/bin/python -m unittest pct3d_reconstruction/test_stage8.py -v
```

代码侧QC位于：

```text
pct3d_reconstruction/qc/results0718_compact_3d_pilot/
```

最终生成三维RSP、真值、正交切片、材料球指标、边缘宽度、验证/测试WEPL、
运行资源和`stage8_summary.md`。Stage 8的PASS表示完整三维链通过，不等价于
全部性能目标均达标。

## Stage 8C诊断与重新计算

Stage 8首轮结果通过了数据完整性、CPU/CUDA MLP一致性和严格伴随测试，但3轮时材料球对比度明显不足。Stage 8C在独立输出目录中依次完成坐标与旋转、有限圆柱求交、路径覆盖、常数投影、伴随关系、匹配合成闭环和松弛调度诊断。基础算子门控全部通过，固定松弛因子0.15在五球合成数据上明显优于原衰减策略，说明首轮主要问题是严重欠收敛以及松弛因子过快衰减。

### 冻结配置与结果

| 项目 | Stage 8C选择 |
|---|---|
| 数据 | 360角度；443,653,707条过滤后质子；80%/10%/10%划分 |
| 网格与路径步长 | `240×80×240 @ 0.5 mm`；0.5 mm |
| 路径与算子 | 两方向Schulte水MLP；三线性八邻域严格转置 |
| 求解器 | GPU OS-SART；18子集；固定松弛0.15；30轮 |
| 数据项与先验 | 等权quadratic；`β=0`，不使用三维TV |
| 约束 | 非负与有限圆柱支撑域 |

修正性测试WEPL RMSE为1.890864 mm；水区偏差和标准差为+0.0403%和0.6305%；10--14 mm大材料球MAPE为0.5969%；6 mm铝球误差为−0.8221%；Air球绝对RSP误差为0.003249。全部预设性能门槛通过。测试分区在Stage 8历史流程中已经查看，因此这是固定配置下的修正性复核，不是从未查看的全新盲测。

严格无噪声合成闭环仍保留一个未决项：固定0.15第30轮的匹配WEPL RMSE为0.09289 mm，没有达到预注册的0.01 mm数值门槛。因而本结果应称为“通过内部性能门槛的三维体素基线”，不能表述为数值收敛与跨场景泛化均已完全证明。

### 复现与证据

Stage 8C的QC位于`qc/results0718_compact_3d_pilot/stage8c/`，检查点位于`data/reconstruction_data/results0718_compact_3d_pilot/stage8c/`。已完成动作保留以下复现入口：

```bash
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action diagnose --device 0
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action closure --device 0
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action convergence --device 0
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action full-fixed015 --device 0
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action test-fixed015 --device 0
.venv-gate/bin/python pct3d_reconstruction/run_stage8c.py --action report
```

正常情况下这些命令会复用通过QC的结果；不要使用`--force`。正式结论见[`fixed015_test_summary.md`](qc/results0718_compact_3d_pilot/stage8c/fixed015_test_summary.md)，完整跨阶段解释见[`project_overview.md`](../pct2d_reconstruction/project_overview.md)。本轮实践结束后Stage 9仅作为未来候选方向保留。

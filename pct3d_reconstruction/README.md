# 三维pCT重建

本目录是独立于`pct2d_reconstruction/`的三维list-mode重建工程。首个实验为
`results0718_compact_3d_pilot`：360个角度、每角度2,000,000个200 MeV质子，
半径50 mm、轴向长度30 mm的有限水圆柱内包含5个不贯穿轴向的材料球。

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

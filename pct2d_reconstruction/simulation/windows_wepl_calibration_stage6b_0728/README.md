# Stage 6B独立WEPL标定仿真

本包用于建立与当前OpenGATE输运配置一致的水射程—能量关系，解决`I=78 eV`
简化LUT相对Geant4/MLIC约`+1.4%`的水WEPL偏差。它不使用S1--S5重建结果拟合，
因此可作为独立标定数据。

## 当前状态

84个工况的工作站仿真、单调射程拟合、锁定测试及S1--S5复算均已完成。测试集
平均/最大绝对相对偏差为`0.0461%/0.1657%`，模型已晋升为
`g4_water_calibrated`。冻结模型与完整结论见
[`../../research_stages/stage6b_wepl_calibration/`](../../research_stages/stage6b_wepl_calibration/)。
历史`bb78`模型保留用于复现旧结果，不再作为新正式重建的默认科学口径。

## 场景

单能质子依次穿过入口参考面、已知厚度水板和出口参考面。两个参考面记录同一
primary质子的入射/出射动能；水板物理列表为`QGSP_BIC_EMZ`，最大步长1 mm。
世界为Vacuum，避免把Air能损混入水标定。

| 参数 | 设置 |
|---|---|
| 入射能量 | 30--230 MeV，10 MeV间隔 |
| 水厚度 | 当前BB78标称射程的10%、30%、50%、70%，四舍五入至0.5 mm |
| 质子数 | 100,000/case，共84 case、840万质子 |
| 训练能量 | 30、50、…、230 MeV |
| 验证能量 | 40、80、120、160、200 MeV |
| 测试能量 | 60、100、140、180、220 MeV |
| 随机种子 | `2026072800 + case_index` |

输出目录默认为：

```text
D:\OpenGATE\data\simulation_data\results0728_stage6b_wepl_calibration\
  e030_f10\PhaseSpaceIn.root
  e030_f10\PhaseSpaceOut.root
  ...
```

日志、配置快照、进度和逐case metadata保存在本包`qc/full/`，不放入数据目录。

## 运行

复制整个文件夹到`D:\OpenGATE`后，在已激活`opengate_env`的PowerShell中：

```powershell
cd D:\OpenGATE\windows_wepl_calibration_stage6b_0728
.\00_check_environment.bat
.\01_smoke_test.bat
.\02_run_full.bat 12
```

另开终端查看总体进度：

```powershell
.\03_show_status.bat
```

重复运行完整命令会跳过配置哈希、质子数和ROOT均匹配的case，并重试失败项。
预计12进程工作站耗时约0.5--1.5小时，ROOT约1--5 GB。

完成后将整个`results0728_stage6b_wepl_calibration`和本包`qc/full`复制回当前
电脑，再运行Stage 6B标定分析。测试能量在模型冻结前不会用于拟合或调参。

# Windows二维pCT夜间多场景仿真包

本文件夹可直接复制到Windows工作站：

```text
D:\OpenGATE\windows_overnight_simulations_0716
```

它不会修改已有的`results0716`。大型ROOT写入：

```text
D:\OpenGATE\data\simulation_data\<output_name>
```

日志、配置快照、完成标志、manifest和汇总保存在本文件夹的`qc\`中。

该包服务于`future_research_plan.md`的阶段1、阶段2和阶段7前置仿真：S6检查能量相关RSP/
WEPL一致性，S2--S5提供均匀水、Air、材料定量和空间分辨率诊断场景，S1给出
results0716的Air配对实验。它尚未加入真实探测器材料、有限分辨率或电子学噪声；
这些效应应在现有理想探测器场景完成分析后分级加入。

## 仿真内容

| 场景 | 角度×质子/角度 | 外部介质 | 目的 | 输出目录 |
|---|---:|---|---|---|
| S1 原25根铝柱 | 720×450,000 | Air | 与results0716 Vacuum完整配对 | `results0717_s1_aluminium_air_full` |
| S2 均匀水圆柱 | 720×100,000 | Vacuum | 隔离解析边界伪影 | `results0717_s2_water_vacuum_pilot` |
| S3 均匀水圆柱 | 720×100,000 | Air | 与S2分离Air影响 | `results0717_s3_water_air_pilot` |
| S4 大材料柱 | 720×100,000 | Air | 材料平台、部分容积和径向误差 | `results0717_s4_material_calibration_air_pilot` |
| S5 线对和斜边 | 720×100,000 | Air | 径向/切向MTF和线对分辨率 | `results0717_s5_resolution_air_pilot` |
| S6 材料薄板 | 52组×100,000 | Vacuum背景 | Water/Aluminium/Air能量—厚度标定 | `results0717_s6_material_energy_scan` |

S4使用Air、Lung、A150_Tissue_Plastic、SpineBone和Aluminium五种材料，在
30、60和85 mm三个半径放置15 mm柱，并在中心保留一个5 mm铝柱。S5包含
0.5、0.75、1.0、1.5、2.0和3.0 mm六组铝线对及五个SpineBone斜边目标。

所有CT场景保留720角度。S2–S5只降低通量，不降低角度数，避免角度欠采样干扰
边界和MTF判断。

## S1–S5共用的二维CT设置

S1–S5均继承`scenarios\base_ct.json`。除各场景明确覆盖的内容外，下面的束流、
扫描几何、记录面和物理模型完全相同。

| 参数 | 设置 |
|---|---:|
| 质子源 | 200 MeV单能质子 |
| 投影数 | 720，角度为0、0.5、…、359.5° |
| 旋转轴 | 扫描器\(y\)轴 |
| 水圆柱 | 半径100 mm、轴向长度400 mm |
| 源位置 | `z=-1060 mm` |
| 有效焦点 | `z=-1000 mm` |
| 源平面尺寸 | `15×0.12×10⁻⁶ mm³` |
| 等中心束流覆盖 | `250×2 mm²` |
| 入口/出口参考面 | `z=-110/+110 mm` |
| 参考面尺寸 | `400×400×10⁻⁶ mm³` |
| 世界尺寸 | `4000×4000×4000 mm³` |
| 物理列表 | `QGSP_BIC_EMZ` |
| 水和普通插入物最大step | 1 mm |
| 单个角度的OpenGATE线程数 | 1 |

源到有效焦点的距离为60 mm，源的两个横向尺寸继续传播到等中心后按`1000/60`
放大，因此形成`250×2 mm²`的扇束。第二个方向只有2 mm，所以这些场景是准二维
仿真，不是完整三维锥束CT。

入口和出口面使用`PhaseSpaceActor`理想记录质子状态，输出字段为：

```text
RunID, EventID, TrackID, KineticEnergy, PreGlobalTime, Position, Direction
```

两个参考面不是有厚度硅跟踪器，也没有条带/像素量化、电子学噪声、探测效率或物理
能量探测器。因此S1–S5用于研究介质、模体和重建算法，不代表真实探测器性能。

每个CT场景的数据目录都包含720个角度子目录。例如S2为：

```text
D:\OpenGATE\data\simulation_data\
└── results0717_s2_water_vacuum_pilot\
    ├── run_000\
    │   ├── PhaseSpaceIn.root
    │   └── PhaseSpaceOut.root
    ├── run_001\
    ├── ...
    └── run_719\
```

对应QC不放在数据目录，而位于：

```text
D:\OpenGATE\windows_overnight_simulations_0716\
└── qc\s2_water_vacuum_pilot\
    ├── scenario_config.json
    ├── base_ct.json
    ├── launcher_summary.json
    ├── result_manifest.csv
    ├── logs\run_000.stdout.log / run_000.stderr.log
    └── runs\run_000\
        ├── completed.flag
        ├── run_metadata.json
        └── protonct.txt
```

## S1：原25根铝柱模体的Air完整通量扫描

配置文件：`scenarios\s1_aluminium_air_full.json`。

### 目的

S1是`results0716`的直接配对实验。原水圆柱、25根铝柱、200 MeV、720角度和论文
通量全部保留，主要变化只有世界材料由Vacuum改为Air。它用于回答：

1. 两个固定参考面之间的外部Air能损会给WEPL和RSP带来多大偏差；
2. Air中的能损与散射是否改变水区均值、铝平台和外围伪影；
3. 对Air路径校正后，S1能否恢复到`results0716`的Vacuum基线。

### 场景和参数

| 参数 | 设置 |
|---|---:|
| 外部介质 | Air |
| 铝柱数 | 25 |
| 铝柱直径/长度 | 5/400 mm |
| 铝柱中心半径 | 0、5、9、13、…、97 mm |
| 相邻柱方位角增量 | 139° |
| 每角度质子数 | 450,000 |
| 总入射质子数 | 324,000,000 |
| 随机种子 | `20260720 + angle_index` |

铝柱按照上述半径和139°角度增量形成螺旋式分布，与`results0716`几何一致。

### 输出

```text
数据：D:\OpenGATE\data\simulation_data\
      results0717_s1_aluminium_air_full\run_000 ... run_719

QC：  D:\OpenGATE\windows_overnight_simulations_0716\
      qc\s1_aluminium_air_full\
```

入口和出口能量定义在`z=-110/+110 mm`。水圆柱之外但仍处于两个参考面之间的Air
能损也包含在WEPL中，后续必须扣除已知Air路径贡献，或把Air作为固定背景加入前向模型。

## S2：Vacuum中的均匀水圆柱

配置文件：`scenarios\s2_water_vacuum_pilot.json`。

### 目的

S2删除全部铝柱，只保留Vacuum中的均匀水圆柱，用于隔离水圆柱边界问题：

- 如果仍出现近似圆对称的外围圆环，说明它不是铝柱产生的；
- 可单独研究水—Vacuum阶跃、Ramp长程响应、有限DDB视野和解析支撑域；
- 可测量均匀水RSP是否随半径或方位变化。

### 场景、参数和输出

| 参数 | 设置 |
|---|---:|
| 外部介质 | Vacuum |
| 模体 | 半径100 mm、长度400 mm的均匀Water圆柱 |
| 插入物 | 无 |
| 每角度质子数 | 100,000 |
| 总入射质子数 | 72,000,000 |
| 随机种子 | `20261720 + angle_index` |

```text
数据：D:\OpenGATE\data\simulation_data\
      results0717_s2_water_vacuum_pilot\run_000 ... run_719

QC：  D:\OpenGATE\windows_overnight_simulations_0716\
      qc\s2_water_vacuum_pilot\
```

通量低于S1，但仍保留720个角度，避免角度欠采样与边界伪影混在一起。

## S3：Air中的均匀水圆柱

配置文件：`scenarios\s3_water_air_pilot.json`。

### 目的

S3与S2形成配对：水圆柱、角度数、每角度质子数和记录面均相同，只把外部介质从
Vacuum改为Air。因此`S3-S2`主要反映Air能损和Air散射的影响，不受铝柱干扰。
S3还可以验证S6得到的Air WEPL修正是否能消除均匀水的RSP偏移。

### 场景、参数和输出

| 参数 | 设置 |
|---|---:|
| 外部介质 | Air |
| 模体 | 半径100 mm、长度400 mm的均匀Water圆柱 |
| 插入物 | 无 |
| 每角度质子数 | 100,000 |
| 总入射质子数 | 72,000,000 |
| 随机种子 | `20262720 + angle_index` |

```text
数据：D:\OpenGATE\data\simulation_data\
      results0717_s3_water_air_pilot\run_000 ... run_719

QC：  D:\OpenGATE\windows_overnight_simulations_0716\
      qc\s3_water_air_pilot\
```

## S4：多材料和径向位置标定模体

配置文件：`scenarios\s4_material_calibration_air_pilot.json`。

### 目的

`results0716`只有水和直径5 mm铝柱，小柱平台容易受到部分容积影响。S4使用较大的
15 mm材料柱，并把相同材料重复放在三个半径，用于评价：

- 多材料RSP误差和MAPE，而不再只看Aluminium；
- 同一材料的RSP是否随半径变化；
- 固定均匀水MLP在非均匀材料中的模型偏差；
- 15 mm材料平台与中心5 mm铝柱之间的部分容积差异；
- Air校正后是否还存在材料相关系统误差。

### 材料布局

每个圆环都包含`Air、Lung、A150_Tissue_Plastic、SpineBone、Aluminium`五种材料，
相邻材料相隔72°。下表的角度定义在模体自身的二维截面局部坐标中；OpenGATE再用
统一的初始旋转把该截面放到扫描器\(x-z\)重建平面，并绕扫描器\(y\)轴采集：

| 圆环半径 | 柱直径 | Air角度 | Lung角度 | A150角度 | SpineBone角度 | Aluminium角度 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 mm | 15 mm | 8° | 80° | 152° | 224° | 296° |
| 60 mm | 15 mm | 32° | 104° | 176° | 248° | 320° |
| 85 mm | 15 mm | 56° | 128° | 200° | 272° | 344° |

圆柱中心另放置一根直径5 mm的Aluminium柱。全部材料柱沿准二维轴向贯穿400 mm。
圆柱内部的Air柱表示低密度空腔，它与圆柱外部Air的路径贡献必须分别评价。

### 其他参数和输出

| 参数 | 设置 |
|---|---:|
| 外部介质 | Air |
| 15 mm材料柱 | 15根 |
| 中心5 mm铝柱 | 1根 |
| 每角度质子数 | 100,000 |
| 总入射质子数 | 72,000,000 |
| 随机种子 | `20263720 + angle_index` |

```text
数据：D:\OpenGATE\data\simulation_data\
      results0717_s4_material_calibration_air_pilot\run_000 ... run_719

QC：  D:\OpenGATE\windows_overnight_simulations_0716\
      qc\s4_material_calibration_air_pilot\
```

## S5：线对和斜边空间分辨率模体

配置文件：`scenarios\s5_resolution_air_pilot.json`。

### 目的

S5用于建立比“肉眼观察铝柱”或单一10%–90%边缘宽度更标准的空间分辨率评价：

- Aluminium线对用于判断不同尺寸结构是否可分辨；
- SpineBone斜边用于计算ESF、LSF、`MTF50`和`MTF10`；
- 多个位置与方向用于比较中心/外围、径向/切向分辨率；
- 保留720角度，避免把角度欠采样误认为系统分辨率。

### Aluminium线对

每组包含4根平行铝条，条宽等于间隙宽度，条长10 mm：

| 条宽/间隙 | 理论频率 | 模体局部截面中心 | 条数 |
|---:|---:|---:|---:|
| 0.50 mm | 1.000 lp/mm | `(-60,-55) mm` | 4 |
| 0.75 mm | 0.667 lp/mm | `(-30,-55) mm` | 4 |
| 1.00 mm | 0.500 lp/mm | `(0,-55) mm` | 4 |
| 1.50 mm | 0.333 lp/mm | `(30,-55) mm` | 4 |
| 2.00 mm | 0.250 lp/mm | `(58,-50) mm` | 4 |
| 3.00 mm | 0.167 lp/mm | `(55,-25) mm` | 4 |

线对频率按`1/(2×线宽)`计算。细线目标的Geant4最大step自动限制为不超过线宽一半，
例如0.5 mm线宽使用不超过0.25 mm的step。

### SpineBone斜边

每个目标为`15×15 mm²`的SpineBone方块：

| 目标 | 模体局部截面中心 | 旋转角 |
|---:|---:|---:|
| 1 | `(0,25) mm` | 5° |
| 2 | `(40,35) mm` | 50° |
| 3 | `(-40,35) mm` | -40° |
| 4 | `(72,30) mm` | 28° |
| 5 | `(-72,30) mm` | -28° |

倾斜边缘避免与重建像素网格完全对齐，并允许测量不同位置和方向的MTF。

### 其他参数和输出

| 参数 | 设置 |
|---|---:|
| 外部介质 | Air |
| 每角度质子数 | 100,000 |
| 总入射质子数 | 72,000,000 |
| 随机种子 | `20264720 + angle_index` |

```text
数据：D:\OpenGATE\data\simulation_data\
      results0717_s5_resolution_air_pilot\run_000 ... run_719

QC：  D:\OpenGATE\windows_overnight_simulations_0716\
      qc\s5_resolution_air_pilot\
```

## S6：材料—能量—厚度薄板扫描

配置文件：`material_scan_config.json`。

S6不是CT扫描，不旋转模体，也不生成720个投影。它让平行单能质子束穿过已知材料
和厚度的平板，用于建立能量响应与WEPL标定。

### 目的

- 检查Water和Aluminium的有效RSP是否随入射能量及厚度变化；
- 检查使用水`I=78 eV` Bethe–Bloch LUT时是否产生材料相关WEPL偏差；
- 测量20、220、1000和2000 mm Air路径能损，为Air场景提供校正；
- 区分铝平台固定误差来自材料物理、WEPL转换还是图像重建；
- 统计各材料和厚度下的primary存活率与能损分布。

### 能量、材料和厚度

入射能量统一为`150、180、200、220 MeV`：

| 材料 | 厚度 | 最大step | case数 |
|---|---|---:|---:|
| Water | 5、10、20、50、100 mm | 1 mm | 20 |
| Aluminium | 5、10、20、50 mm | 1 mm | 16 |
| Air | 20、220、1000、2000 mm | 10 mm | 16 |
| 合计 | 13种材料—厚度组合×4能量 | — | 52 |

每个case包含100,000个质子，共5,200,000个；随机种子为
`20265720 + case_index`。

### 薄板几何

| 参数 | 设置 |
|---|---:|
| 世界介质 | Vacuum |
| 薄板横向尺寸 | `100×20 mm²` |
| 束流 | 沿`+z`的平行单能束 |
| 源尺寸 | `10×0.1×10⁻⁶ mm³` |
| 入口/出口面 | 分别距薄板表面0.5 mm |
| 参考面尺寸 | `120×20×10⁻⁶ mm³` |
| 物理列表 | `QGSP_BIC_EMZ` |

源位于入口参考面上游5 mm。每个case输出两个理想相空间ROOT。

### 输出

case目录名由材料、能量和厚度组成，例如：

```text
D:\OpenGATE\data\simulation_data\
└── results0717_s6_material_energy_scan\
    ├── water_e150_t0005\
    │   ├── PhaseSpaceIn.root
    │   └── PhaseSpaceOut.root
    ├── water_e150_t0010\
    ├── ...
    ├── aluminium_e220_t0050\
    └── air_e220_t2000\
```

对应QC和汇总位于：

```text
D:\OpenGATE\windows_overnight_simulations_0716\
└── qc\s6_material_energy_scan\
    ├── launcher_summary.json
    ├── material_scan_summary.csv
    ├── cases\<case_id>\
    │   ├── completed.flag
    │   ├── case_metadata.json
    │   └── protonct.txt
    └── logs\
```

`material_scan_summary.csv`包含入口/出口primary数量、存活率、能损统计和使用
`I=78 eV`水LUT计算的WEPL。

## 运行命令

打开已激活`opengate_env`的PowerShell，进入本目录：

```powershell
cd D:\OpenGATE\windows_overnight_simulations_0716
```

依次运行：

```powershell
.\00_check_environment.bat
.\01_smoke_test_all.bat
.\02_run_overnight.bat 12
```

另开一个PowerShell可随时查看状态：

```powershell
.\03_show_status.bat
```

`02_run_overnight.bat`按`S6 → S1 → S2 → S3 → S4 → S5`顺序执行，场景之间
串行、场景内部最多12个单线程进程并行。预计总时间4–6小时、ROOT约75–85 GB；
脚本启动前要求目标盘至少有90 GB可用空间，建议实际预留100 GB以上。

是否完成以`qc\<scenario>\launcher_summary.json`、逐任务`completed.flag`和
`03_show_status.bat`为准，不应仅根据输出目录存在与否判断。拷回当前电脑时只需
迁移各`results0717_*`数据目录和本包`qc/`，不要把ROOT混入代码目录。

## 断点续跑

每个CT角度和每个材料扫描组合都有独立`completed.flag`。中断后重新运行同一命令：

```powershell
.\02_run_overnight.bat 12
```

已完成且配置、质子数一致的任务会跳过，未完成或失败的任务会重试。不要在同一
输出目录下手工混入其他配置生成的ROOT。

## Air数据的重要说明

Air场景的`E_in/E_out`定义在固定`z=-110/+110 mm`参考面。因此两参考面之间、
水圆柱以外的Air能损包含在测量WEPL中。后续重建不能把这部分WEPL全部归入水
圆柱，必须先扣除已知Air路径贡献，或把Air作为固定背景加入前向模型。

S6会自动生成：

```text
qc\s6_material_energy_scan\material_scan_summary.csv
```

其中包含primary存活率、能损分布以及与当前PCT一致的`I=78 eV`水LUT WEPL，
可用于建立外部Air校正和分析铝的能量相关有效RSP。

## 如果环境检查失败

确认当前PowerShell已经激活OpenGATE环境，然后执行：

```powershell
python -m pip install -r requirements-windows.txt
```

安装完成后重新运行环境检查和smoke test。只有全部smoke test显示PASS后才运行
夜间正式任务。

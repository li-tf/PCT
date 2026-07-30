# PCT 第一批冷数据归档记录

记录日期：2026-07-30  
归档目录名称：`PCT_archive_batch1_20260730`  
移动硬盘最终位置：移动完成后补充  
归档状态：已在PCT项目根目录完成集中整理，等待移动到移动硬盘。

## 1. 归档规模

- 数据文件数：`68,187`
- 数据总字节数：`174,088,230,916 bytes`
- 数据容量：约`162.13 GiB`
- 加上归档目录中的`README.md`和`archive_manifest.tsv`后，共`68,189`个文件
- 文件系统显示占用：约`163 GB`

## 2. 归档原则

本批内容均为已经完成、短期不参与Stage 7数据加权或Stage 8三维重建的历史
实验与中间数据。

以下内容没有归档，继续保留在本机：

- `pct2d_reconstruction/`全部代码、QC、报告和配置；
- Stage 6B独立WEPL标定和Stage 6A虚拟MLIC真值；
- S1铝柱、S4多材料和S5线对卡的`stage6b_calibrated`正式输入；
- S1、S4和S5的最终解析及迭代重建结果；
- Stage 7三个全量正式方案及对应重建结果；
- 后续Stage 8所需的活动空间。

## 3. 移动后的归档结构

```text
PCT_archive_batch1_20260730/
├── README.md
├── archive_manifest.tsv
├── test0707/
├── test0710/
├── test0713/
└── data/
    ├── simulation_data/
    │   ├── results0716/
    │   ├── results0717_s2_water_vacuum_pilot/
    │   ├── results0717_s3_water_air_pilot/
    │   ├── results0717_s6_material_energy_scan/
    │   └── results0724_mlp_truth_pilot/
    ├── preprocessing_data/
    │   ├── results0716/
    │   ├── results0717_s2_water_vacuum_pilot/
    │   ├── results0717_s3_water_air_pilot/
    │   └── results0724_mlp_truth_pilot/
    └── reconstruction_data/
        ├── results0716/
        ├── results0717_s2_water_vacuum_pilot/
        └── results0717_s3_water_air_pilot/
```

## 4. 逐项清单

| 原始相对路径 | 文件数 | 字节数 | 约合容量 |
|---|---:|---:|---:|
| `test0707` | 2,226 | 1,341,248,091 | 1.25 GiB |
| `test0710` | 2,916 | 13,685,508,778 | 12.75 GiB |
| `test0713` | 4,579 | 17,709,842,878 | 16.49 GiB |
| `data/simulation_data/results0716` | 1,440 | 39,487,636,674 | 36.78 GiB |
| `data/preprocessing_data/results0716` | 5,040 | 33,165,594,374 | 30.89 GiB |
| `data/reconstruction_data/results0716` | 17 | 141,376,098 | 134.83 MiB |
| `data/simulation_data/results0717_s2_water_vacuum_pilot` | 1,440 | 8,864,944,059 | 8.26 GiB |
| `data/preprocessing_data/results0717_s2_water_vacuum_pilot` | 25,200 | 21,360,814,225 | 19.89 GiB |
| `data/reconstruction_data/results0717_s2_water_vacuum_pilot` | 385 | 2,751,155,626 | 2.56 GiB |
| `data/simulation_data/results0717_s3_water_air_pilot` | 1,440 | 9,124,059,550 | 8.50 GiB |
| `data/preprocessing_data/results0717_s3_water_air_pilot` | 23,040 | 20,178,286,107 | 18.79 GiB |
| `data/reconstruction_data/results0717_s3_water_air_pilot` | 71 | 564,757,327 | 538.59 MiB |
| `data/simulation_data/results0724_mlp_truth_pilot` | 216 | 4,711,585,303 | 4.39 GiB |
| `data/preprocessing_data/results0724_mlp_truth_pilot` | 73 | 561,671,252 | 535.65 MiB |
| `data/simulation_data/results0717_s6_material_energy_scan` | 104 | 439,750,574 | 419.38 MiB |
| **合计** | **68,187** | **174,088,230,916** | **162.13 GiB** |

## 5. 恢复方法

假设移动硬盘挂载点为`/media/ltf/PCT_DISK`，在PCT项目根目录执行：

```bash
ARCHIVE=/media/ltf/PCT_DISK/PCT_archive_batch1_20260730

rsync -a --info=progress2 "$ARCHIVE/data/" data/
rsync -a --info=progress2 "$ARCHIVE/test0707" ./
rsync -a --info=progress2 "$ARCHIVE/test0710" ./
rsync -a --info=progress2 "$ARCHIVE/test0713" ./
```

恢复后应核对：

- 数据文件总数为`68,187`；
- 数据总字节数为`174,088,230,916`；
- 归档清单中的15个原始相对路径均已恢复；
- 历史QC中记录的原始相对路径重新有效。

## 6. 移动硬盘校验

移动完成后应在本记录中补充：

- 移动硬盘卷标或挂载点；
- 归档文件夹实际位置；
- 移动完成日期；
- 移动硬盘端文件数和总字节数；
- 是否与本记录一致；
- 是否保留第二份备份。


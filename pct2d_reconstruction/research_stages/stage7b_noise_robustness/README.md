# Stage 7B：位置与能量噪声鲁棒性

本阶段只读取D1的六平面ROOT，在过滤前按`(RunID, EventID)`固定划分
80%训练、10%验证和10%测试。筛选阶段不会读取测试集。代码和结果与阶段7
隔离，不会覆盖成熟重建链。

## 最终结果

阶段已于2026-07-30完成，状态为`NO_PROMOTION`。组合噪声下等权quadratic的
验证ideal-WEPL RMSE为`3.73757 mm`；解析逆方差、经验逆方差、Huber 1.5、
Huber 2.5及经验逆方差与Huber组合分别得到`3.80745`、`3.81179`、`3.76075`、
`3.74920`和`3.82484 mm`，均未通过晋升门槛。

因此程序跳过80%正式双重建及锁定测试，测试集始终未打开，最终继续使用阶段4
冻结的等权quadratic数据项。详细结论见`qc/stage7b_summary.md`。

## 复现命令

第一批：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage7b_noise_robustness/run_stage7b.py \
  --action screen \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

检查`qc/screen_summary.md`后执行第二批：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage7b_noise_robustness/run_stage7b.py \
  --action confirm \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

进度：

```bash
watch -n 30 \
  ".venv-gate/bin/python pct2d_reconstruction/research_stages/stage7b_noise_robustness/run_stage7b.py --action status"
```

程序按角度、DDB、候选、epoch和subset保存检查点；中断后重新执行相同命令。
`--force`只会删除并重算Stage 7B自己的产物。

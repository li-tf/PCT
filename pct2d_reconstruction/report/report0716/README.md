# report0716

本目录对应`simulation0716 / results0716`，保存完整中文实验报告、英文图表、
数据表和报告QC。跨实验研究路线已移至
`pct2d_reconstruction/future_research_plan.md`。

## 当前内容

- 720角度、论文通量Windows OpenGATE仿真及设备/时间统计；
- primary-only配对、3σ过滤和Schulte MLP DDB投影；
- results0716 no-Hann解析重建，并与test0713 no-Hann作统一RSP对照；
- results0716全量0.1 mm、3 epoch GPU迭代重建；
- 200 MeV参考RSP真值、误差图、边界剖面、epoch曲线和关键指标；
- 指向跨阶段总结和精简后续计划的链接。

迭代部分只使用results0716自身结果，不把test0713迭代结果混入比较。阶段0固定
训练/验证划分和验证WEPL结果保存在`pct2d_reconstruction/evaluation/`。历史
完成记录现已合并到根目录`current_research_summary.md`，计划文件不再保存阶段
工作结果。

## 生成命令

```bash
.venv-gate/bin/python pct2d_reconstruction/report/build_report.py \
  --experiment 0716 --force
```

生成器读取已有QC和重建结果，不重新执行OpenGATE、预处理、FDK或GPU迭代。它会
更新：

- `report0716_summary_report.md`；
- `assets/`中的8幅正式图；
- `tables/`中的协议、数量、时间和RSP指标CSV；
- `qc/report_summary.json`中的本地链接及产物检查结果。

当前报告QC为PASS，8幅图片链接均有效。阶段0--8的最新总结见
[`current_research_summary.md`](../../current_research_summary.md)；后续研究安排见
[下一阶段研究计划](../../future_research_plan.md)。

## 数据归档状态

本报告、图表、表格和QC仍完整保留，但其results0716与test0713大型源数据已于
2026-07-30迁入第一批冷归档。阅读现有报告不受影响；重新执行生成器或追溯
MHD/ROOT时，应先按
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)
恢复原路径。当前对外汇报用的最终PPT统一保存在
本地私有目录`report/final_presentations/`；该目录包含个人信息，已整体排除
在Git版本控制之外。

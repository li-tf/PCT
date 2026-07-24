# report0716

本目录对应`simulation0716 / results0716`，保存完整中文实验报告、英文图表、数据表、
报告QC和后续研究路线。

## 当前内容

- 720角度、论文通量Windows OpenGATE仿真及设备/时间统计；
- primary-only配对、3σ过滤和Schulte MLP DDB投影；
- results0716 no-Hann解析重建，并与test0713 no-Hann作统一RSP对照；
- results0716全量0.1 mm、3 epoch GPU迭代重建；
- 200 MeV参考RSP真值、误差图、边界剖面、epoch曲线和关键指标；
- 阶段0至阶段8后续研究计划。

迭代部分只使用results0716自身结果，不把test0713迭代结果混入比较。阶段0固定
训练/验证划分和验证WEPL结果保存在`pct2d_reconstruction/evaluation/`；完成记录已
写入根目录`pct2d_reconstruction/future_research_plan.md`的`4.4 工作结果`。

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

当前报告QC为PASS，8幅图片链接均有效。后续研究安排和阶段完成记录见
[二维质子CT下一阶段研究计划](../../future_research_plan.md)。

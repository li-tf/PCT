# research_stages阶段总结

本目录汇总`pct2d_reconstruction/research_stages/`的阶段性研究结果，与
`report0716/`的单次实验总结分开维护。

当前主文档为：

- [research_stages_summary.md](research_stages_summary.md)

配套表格：

- [stage_status.csv](tables/stage_status.csv)：阶段状态、数据来源和当前决策；
- [key_results.csv](tables/key_results.csv)：已冻结或已通过筛选的核心数值。

报告直接引用各阶段QC目录中的可复现图片，不复制ROOT、pairs、DDB、MHD或重建
检查点。阶段0--7均已有正式结论；阶段4在锁定测试后晋升，阶段5的真实轨迹
上限实验和阶段6高级先验验证均未达到晋升门槛，因此继续保留阶段4水MLP与固定
Huber-TV。阶段7证明连续硅hit下冻结算法仍稳定，并量化了离线位置/能量噪声
边界；阶段8尚未形成正式结果。

当前正式结论以Stage 6B校准后的S1/S4/S5三场景结果和Stage 7三套全量结果为
主。results0716、S2、S3、MLP truth pilot及S6原始大数据已进入第一批冷归档，
但各阶段代码侧QC和本汇总均保留；恢复方法见
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)。
最终进展汇报和总结汇报只保存在本地私有目录`report/final_presentations/`，
不进入Git版本控制。

后续更新原则：

1. 每个阶段完成后先更新该阶段自己的`stage*_summary.md`和
   `future_research_plan.md`；
2. 再把正式状态、核心指标和阶段决策同步到本报告；
3. 开发集、验证集和锁定测试结果必须明确区分；
4. 负结果同样保留，不能只汇报胜出候选。

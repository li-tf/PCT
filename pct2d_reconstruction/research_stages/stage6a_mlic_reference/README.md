# 阶段6A：虚拟MLIC参考

本阶段只读使用虚拟多层电离室深度剂量结果和已有重建，不重新运行OpenGATE、
预处理或GPU重建。它冻结200 MeV高统计MLIC-RSP，并使用相同ROI重新评价
results0716/S1、S4和S5。

阶段6A已经完成并通过验收。200 MeV高统计虚拟MLIC冻结了独立材料RSP真值；
其中Water和Aluminium分别为`0.999746`和`2.094511`。results0716大型重建数据
已进入第一批冷归档，S1、S4、S5正式三场景数据仍在本地，因此已有总结可直接
阅读，但重新评价results0716前需要先恢复数据。归档记录见
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)。

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6a_mlic_reference/run_stage6a.py \
  --action all
```

只读复核：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6a_mlic_reference/run_stage6a.py \
  --action verify
```

全部产物位于`qc/`，核心结论见`qc/stage6a_summary.md`。

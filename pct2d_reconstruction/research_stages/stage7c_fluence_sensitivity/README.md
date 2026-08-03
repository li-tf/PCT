# Stage 7C：D1有效质子通量敏感性

本阶段不重新运行OpenGATE。它在Stage 7局部3σ过滤后的正式pairs上，以
`(RunID, EventID)`构造严格嵌套的100%、50%、25%和10%有效质子子集，比较：

- 理想入口/出口参考面；
- 四层连续硅hit拟合；
- 0.2 mm位置误差与1%出射能量噪声。

所有低通量点使用独立DDB-FDK初值和阶段4冻结的5轮等权OS-SART，不重新调整
正则化、松弛因子或停止轮数。组合噪声25%和10%使用三个抽样种子。

D1没有DoseActor，因此结果只能称为“通量敏感性”，不能直接换算为mGy。

## 正式结果

Stage 7C已经完成，状态为`FLUENCE_SENSITIVITY_CHARACTERIZED`。三种测量条件的
推荐最低有效通量均为25%，即`225 protons/mm²/projection`；组合噪声25%经
三个随机种子复核均通过。10%通量在三种条件下均失稳，水区标准差升至约
`8.4%--10.6%`。完整数值和图见[`qc/stage7c_summary.md`](qc/stage7c_summary.md)。

## 复现命令

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage7c_fluence_sensitivity/run_stage7c.py \
  --action all \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

另开终端：

```bash
watch -n 30 \
  ".venv-gate/bin/python pct2d_reconstruction/research_stages/stage7c_fluence_sensitivity/run_stage7c.py --action status"
```

程序支持断点续跑。`--force`只重算Stage 7C自身产物。已有正式结果通常不应
重算；复算前需确认D1原始ROOT挂载和约50--60 GB额外空间。

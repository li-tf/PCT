# 阶段6：固定MLP下的高级图像先验

本目录是独立研究实现，不修改成熟的`preprocessing/`、
`analytic_reconstruction/`、`iterative_reconstruction/`或阶段4结果。阶段6
固定局部3σ、等权数据、quadratic损失、水Schulte MLP、`λ0=0.25`、衰减0.2、
18子集和5 epoch，只比较图像域先验。

## 当前状态（2026-07-28）

阶段6已完成，流程状态为PASS，科学决策为
`RETAIN_STAGE4_VALIDATION_FAIL`。TGV在预筛中未通过MTF约束；自适应TV与方向
TV完成S2/S4/S5正式验证，但水区噪声分别明显回升，材料和RSP指标没有达到实质
改善门槛。因此测试集未打开，后续继续使用阶段4固定Huber-TV。详细数值见
`qc/stage6_summary.md`。

S2/S3和早期阶段6大型检查点已进入第一批冷归档，代码侧选择结果与负结果总结
仍保留。完整复算需先按
[`../../archive_batch1_20260730_record.md`](../../archive_batch1_20260730_record.md)
恢复对应数据；当前正式复用应直接采用Stage 4冻结配置。

## 候选方法

- 阶段4固定`β=0.0125` Huber-TV：比较基线，不重复调参；
- 二阶Huber-TGV：减少普通TV在缓变区域产生的阶梯效应；
- 固定边缘自适应Huber-TV：由no-Hann初值生成空间权重；
- 弱方向TV：分别按初值的横向和纵向梯度减弱跨边缘平滑。

初值只用于生成固定引导权重，不在迭代中更新，避免形成反馈回路。路径概率图和
深度学习先验不属于本阶段范围。

## 单命令运行

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6_advanced_priors/run_stage6.py \
  --action all --datasets s1,s2,s3,s4,s5 --jobs 4 --device 0
```

该命令自动完成：

1. 算子伴随关系、GPU先验和两角度真实数据smoke test；
2. 14组近端候选预筛，每个方法族最多保留一个；
3. S2/S4/S5完整5 epoch验证重建；
4. 冻结唯一候选后执行S1--S5锁定测试；
5. 自动生成指标、决策与`stage6_summary.md`。

如果没有候选通过预筛或验证门槛，程序会正常生成保留阶段4的负结果并结束。
意外中断后重新执行同一命令即可从已完成epoch继续；正常续跑不要添加`--force`。

## 进度

另开终端执行：

```bash
watch -n 30 '.venv-gate/bin/python pct2d_reconstruction/research_stages/stage6_advanced_priors/run_stage6.py --action status --datasets s1,s2,s3,s4,s5'
```

状态入口只读取原子化`qc/progress.json`，不加载ROOT、不占用GPU。它显示当前阶段、
数据集、候选、epoch、subset、质子处理速率、当前任务ETA和全流程粗略ETA。

## 输出与保护

- 代码侧：本目录`qc/`保存进度、候选表、验证/测试指标和总结；
- 大型检查点：各数据集`data/reconstruction_data/.../stage6/`；
- 测试集在`frozen_candidate.json`生成前保持关闭；
- 配置哈希不一致时拒绝混写；
- `--force`只删除阶段6自身的QC和检查点。

轻量验证命令：

```bash
.venv-gate/bin/python -m unittest \
  pct2d_reconstruction/research_stages/stage6_advanced_priors/test_stage6.py

.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage6_advanced_priors/run_stage6.py \
  --action smoke --datasets s2 --device 0
```

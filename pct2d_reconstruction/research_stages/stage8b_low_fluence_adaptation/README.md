# Stage 8B：低通量适配

本阶段使用D1组合数字化条件研究冻结高通量算法在25%以下的失稳与适配方法，不重新运行OpenGATE，也不修改成熟的预处理和重建代码。

> **最终状态（2026-08-09）：PASS并晋升低通量专用配置。以下命令保留用于复现，不表示当前仍在计算。**

三个任务在正式研究中依次完成；复现时仍应按同一顺序执行并检查决策文件。

## 最终结论

最终采集条件为360角度、1°间隔、每角度20%通量，总质子数约等于原720角度全通量的10%。冻结重建参数为0.5 mm低通初值上采样至0.1 mm、18子集、初始松弛因子0.25、衰减0.1、Huber-TV `β=0.05`并在第2轮停止。锁定测试相对冻结高通量参数将水区标准差降低58.25%、模体RMSE降低4.84%，CNR由约52.0提高到132.6；边缘宽度只增加0.44%，但铝平台误差由−0.644%扩大至−2.663%，因此该方案是低通量下的综合折衷。

噪声分解再次表明0.2 mm位置误差是主要退化来源，当前1%出射能量高斯噪声没有形成可分辨贡献。温和和中等逆方差权重均未优于等权，完整逆方差还违反有效样本量门槛，因此正式数据项继续使用等权quadratic。详细证据见[`qc/stage8b_summary.md`](qc/stage8b_summary.md)和[`qc/stage8b_decision.json`](qc/stage8b_decision.json)。

## 任务1：定位转折点

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_stage8b.py \
  --action transition \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

任务1比较25%、20%、17.5%、15%、12.5%和10%。该实验确认720角度下25%通过、20%失败，但后续剂量讨论改变了任务2的开发条件：不再直接使用720角度×20%。

## 等名义剂量角度—通量对照

任务1完成后，先不调参，比较`360角度×每角度20%`与已有的`720角度×每角度10%`：

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_angular_fluence_baseline.py \
  --action run --jobs 4 --device 0
```

该入口选取原始偶数RunID并重新编号为`0...359°`，所以角度间隔为1°。两组实际质子数分别为17,343,126和17,350,551，只差0.043%；名义总扫描通量相同，均为720角度全通量的10%。两组都使用冻结的DDB-FDK初值和5轮OS-SART参数。

结果显示，`360角度×20%`相对`720角度×10%`将水区标准差从0.103110降至0.023171，将模体RSP RMSE从0.123393降至0.062659。改善从DDB-FDK初值阶段已经出现，说明当前低通量瓶颈主要是单角度DDB统计不足，而不是角度数量不足。该结果仍未通过水区标准差≤1%和CNR≥100门槛，因此被正式冻结为任务2的开发条件，而不是最终合格方案。

实时状态：

```bash
watch -n 30 \
  ".venv-gate/bin/python pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_angular_fluence_baseline.py --action status"
```

## 任务2：低通量参数优化

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_stage8b.py \
  --action optimize \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

任务2固定使用`360角度×每角度20%`，即1°角间隔、总扫描通量为720角度全通量的10%。程序顺序筛选停止epoch、松弛因子、衰减、Huber-TV和多尺度初值；开发、验证和锁定测试均采用相同的360角度采集定义。旧任务1的720角度结果只作为历史证据，不再决定任务2通量。

## 任务3：噪声分解与能量加权

```bash
.venv-gate/bin/python \
  pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_stage8b.py \
  --action noise-weighting \
  --raw-root '/mnt/d/临时/results0718_d1_air_tracker_full' \
  --jobs 4 --device 0
```

只有能量噪声贡献达到预注册门槛时，程序才运行温和逆方差候选。

## 实时状态

```bash
watch -n 30 \
  ".venv-gate/bin/python pct2d_reconstruction/research_stages/stage8b_low_fluence_adaptation/run_stage8b.py --action status"
```

所有计算支持断点续跑。正常运行不要使用`--force`。

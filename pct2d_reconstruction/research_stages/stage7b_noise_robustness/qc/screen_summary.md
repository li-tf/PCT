# 阶段7B第一批：噪声准备与候选筛选

状态：**NO_PROMOTION**。

数据在过滤前按`(RunID, EventID)`固定划分为80%训练、10%验证和10%测试；
本批未读取测试指标。

## 噪声源分离

| 条件 | 水标准差 | 模体RMSE | 验证理想WEPL RMSE/mm |
|---|---:|---:|---:|
| continuous | 0.112916 | 0.131178 | 2.76817 |
| position_0p2mm | 0.113066 | 0.132503 | 3.80307 |
| energy_1pct | 0.112923 | 0.131258 | 2.76393 |
| combined_0p2mm_1pct | 0.113099 | 0.132293 | 3.73757 |

## 组合噪声候选

| 方法 | 验证理想WEPL RMSE/mm | p99/mm | 是否合格 |
|---|---:|---:|---:|
| equal_quadratic | 3.73757 | 11.52630 | True |
| analytic_invvar | 3.80745 | 12.14821 | False |
| empirical_invvar | 3.81179 | 12.18839 | False |
| huber_z1p5 | 3.76075 | 11.75042 | False |
| huber_z2p5 | 3.74920 | 11.63070 | False |
| empirical_huber_z2p5 | 3.82484 | 12.31482 | False |

冻结候选：`equal_quadratic`。

只有冻结候选不是等权基线时，第二批才执行两套80%数据正式重建。

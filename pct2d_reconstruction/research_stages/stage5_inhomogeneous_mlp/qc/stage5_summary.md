# 阶段5：非均匀MLP与迭代MLP

## 执行状态

- 最终决定：`RETAIN_STAGE4_LEVEL1_FAIL`
- Level 1真实轨迹门槛：`FAIL`
- Level 2固定非均匀MLP：`未执行`
- Level 3交替更新MLP：`未执行`
- 锁定测试：`未执行`

## 路径上限实验

真实轨迹pilot使用72个角度和五种异质材料。验证集整体路径改善为
`0.006%`，强异质质子改善为
`0.074%`，配对bootstrap
95%下限为`-0.194%`。

![Path error comparison](assets/path_error_comparison.png)

![Path error by depth](assets/path_error_by_depth.png)


## 结论

自动门控严格遵循预注册阈值。Level 1失败表示真实材料先验本身未证明稳定路径
收益，因此Level 2固定非均匀MLP、Level 3交替更新MLP和锁定测试均按设计跳过。
本阶段保留阶段4水MLP基线。

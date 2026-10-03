# TODO-001：融合方向与实测性能的平衡

**状态：原有的强制单 kernel 和融合类型退出限制已修正，后续重点是真实 Triton Ascend 任务验证。**

## 当前规则

- 支持单 kernel、多 kernel、部分融合和按 shape 路由。Stage9 根据实际数据流、硬件能力及测量判断，不因 HBM 中间结果或偏离初始方案强制修改。
- Jev 概率用于选择初始候选，不能替代精度与性能证据。Stage3 可以提出新方案，但要记录选择依据并完成自测和正式评测。
- 退出由程序按全部 case 是否达标、x/y 有效窗口及迭代上限控制；Stage9 的意见不直接决定退出。
- 全部 case 达标优先，同类实现按同口径 avg_speedup 选择最好；慢 case 仍在改善时，不能只因平均提升较小就换方案。
- 人工指导按已实现的人类参与入口交给 Stage9，形成明确 P0 后转 Stage3。

## 还需要验证

1. 在真实 Triton Ascend 任务中，确认 Stage9 能保留有收益的多 kernel 方案，也能在有证据时建议改变融合边界。
2. 观察各 case 的收益与退步，核对 Stage3 是否执行了结构化任务中的目标、文件范围和验收要求。
3. 对比不同方案时保持框架、硬件、case、基准及计时口径一致；尚未重测的历史成绩不能进入当前最佳记录。

此前旧框架报告中的 4.56x 与 2.87x 存在计时口径差异，不能据此证明某种融合方式更快，也不是 Triton 成绩。原始材料由用户另存备份。

规则细节见[语义退出方案](../docs/workflow_fusion_selection_semantic_exit_design.md)、[性能判断规则](../docs/workflow_performance_decision_rules.md)和[Stage9 review](../docs/workflow_stage9_review.md)。

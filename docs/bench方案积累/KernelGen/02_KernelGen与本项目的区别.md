# KernelGen 与本项目的区别

核对日期：2026-09-25。比较公开材料与本项目当前代码，不作未实测的性能排名。

## 共同点

两者都在做“生成 → 实测 → 分析 → 修改”的算子开发闭环。KernelGen 的公开优化 Skill 也记录每轮修改和成绩，维护精度通过版本、历史最佳版本，并结合 profiling 调优。因此，“有循环、有反馈、有最佳版本”本身不是我们的独有特点。[优化 Skill](github/skills-main/skills/kernelgen-flagos/kernelgen-optimize.md)

## 主要区别

**1. 关注范围不同。** KernelGen 覆盖算子生成、优化、跨芯片迁移和入库，也涉及融合算子，例如公开 Skill 的 `vendor-fused` 接入；我们重点管理给定融合需求的方案选择、多 case 迭代和交付。区别在融合决策如何进入开发闭环，不能写成“它不做融合”。[融合接入 Skill](github/skills-main/skills/kernelgen-flagos/kernelgen-specialize-for-flaggems.md)、[本项目 README](../../../README.md)

**2. 我们把融合方案选择单独管理。** Stage1.5 用 Jev 给融合方法评分，筛选 Top N；后续结合实测与选择依据决定是否换方案。KernelGen 公开流程有检索参考实现和生成优化，但未见与我们相同的“融合方法概率库＋逐轮方案审查”机制。其后端是否另有类似实现，公开资料不足以判断。[融合选择代码](../../../lib/fusion_selection.py)

**3. 我们的退出与回退条件更贴合当前任务。** 其公开 Skill 默认目标加速比 1.2、最多优化 10 轮，达目标或到上限结束。我们区分“全部 case ≥1”和“仍有慢 case”，用 x/y 次有效迭代的最佳平均加速比提升判断停滞，再决定交付或审查方案。最佳记录同时保存代码、实现方案、融合方案、HAP、平均加速比和评测证据。[其退出步骤](github/skills-main/skills/kernelgen-flagos/kernelgen-optimize.md)、[我们的判断规则](../../workflow_performance_decision_rules.md)

**4. 人工参与的组织方式不同。** KernelGen 网页支持人工改代码、重测和选版本。我们把人的方向意见送给 Stage9，作为 P0 转交 Stage3；场景2停滞多次触发后，还有问题文档、选项、等待和反馈重新注入的程序流程。[其人工操作](https://docs.flagos.io/projects/kernelgen/en/latest/web_user_guide/view-genertion-results-and-select-optimal-triton-kernel.html)、[我们的人机方案](../../workflow_human_review_design.md)

**5. 当前能审查的实现层次不同。** KernelGen 平台公开了 Skills、工具和样例，核心生成、优化仍依赖远端服务；我们把路由、持久化、窗口计数和人工等待放在本地程序中。其配套 KernelGenBench **另有开源 Agent 执行和实测反馈代码、Ascend 适配**，不能漏掉，也不能拿它代替平台后端。公开材料不足以判断平台内部是否有与我们相同的机制。[平台接入](github/skills-main/skills/kernelgen-flagos/SKILL.md)、[Bench 的 Agent 实现](github/KernelGenBench-main/agent_bench/methods/normal_cc/method.py)、[我们的主程序](../../../orchestrator.py)

## 对我们的启发

优先借鉴两件事：一是把需求、约束、各 case 成绩、瓶颈和历史修改组织成固定输入；二是补充评测覆盖和开发成本统计，观察换 shape、换任务后是否仍有效，以及花了多少轮、时间和 Token。

**我们的差异应由实验说明：融合选择、语义退出和人工指导，是否让成功率更高、慢 case 更少、试错成本更低。** 仅比较流程图和节点数量，无法证明效果。其跨芯片编译和算子库生态也不是增加几个 Stage 就能替代的。

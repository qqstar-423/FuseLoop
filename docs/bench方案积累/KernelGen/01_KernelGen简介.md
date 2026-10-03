# KernelGen 简介

核对日期：2026-09-25。复核官方仓库及发布分支、文档、两篇论文和你提供的两张 PPT；未运行芯片实测。

## 它在做什么

**KernelGen 把“写算子、测正确性、调性能、适配芯片”串成自动迭代流程，主要生成 Triton／Triton 扩展语言算子。** 要解决的是：算子开发依赖专家，而且换芯片后往往还要重新适配、调优。可通过网页或接入 Agent 的工具使用。[官方项目](https://github.com/flagos-ai/KernelGen)

几个名字的关系是：

| 名称 | 干什么 |
|---|---|
| FlagOS | 整套跨芯片 AI 软件体系，KernelGen 是其中一部分。 |
| KernelGen | 自动生成、优化和迁移算子。 |
| FlagTree | 多芯片编译器，把算子编译到不同硬件。 |
| FlagGems | 算子库，接收、维护和提供可用算子。 |
| KernelGenBench | 评测模型和 Agent 的算子生成能力。 |

以上分工见[官方组织说明](https://github.com/flagos-ai)和[各项目职责](https://github.com/flagos-ai/community/blob/main/sigs/README.md)。

## 怎样完成一次开发

输入算子定义、shape、精度与性能要求、目标芯片 → 检索参考实现 → 生成代码 → 编译运行、验证精度 → 根据耗时和 profiling 修改 → 保留较好版本 → 适配目标芯片、接入算子库。

它也允许人修改代码、重测和选择历史版本，并非只能全自动运行。[生成流程](https://docs.flagos.io/projects/kernelgen/en/latest/KernelGen_overview/workflow.html)、[版本选择](https://docs.flagos.io/projects/kernelgen/en/latest/web_user_guide/view-genertion-results-and-select-optimal-triton-kernel.html)

## 做得怎么样

**值得关注的是配套完整：生成工具有编译器、算子库、评测和应用接入来承接。** 不过，宣传数字要区分范围：

- [ppt1](ppt1.jpg)称算子库有 **800 余个算子，超过半数由 KernelGen 生成**，工具支持 9 款芯片。这说明其公布的应用规模；不等于每个算子在全部芯片上都经过同等性能验证。“全球最大”也是材料中的说法，本次未独立核实。
- 同页列出的加速倍数和 KernelBench 百分比，缺少完整评测条件，不能据此认定整体优于其他方法。
- [ppt2](ppt2.jpg)展示的是 **FlagOS 整体的模型适配进展**，不能全算成 KernelGen 单个工具的成果。

## 下载了什么

三个官方仓库已保留 ZIP 并解压到 `github/`：

- [KernelGen](github/KernelGen-main/README.md)：平台文档、接入说明、优化分析、工具和样例。
- [Skills](github/skills-main/README.md)：KernelGen 接入步骤，以及独立的 [TLE 开发 Skill](github/skills-main/skills/tle-developer-flagos/SKILL.md)。后者提供本地开发流程，已有 Performance Record、Keep/Revert、Lessons Entry [模板](github/skills-main/skills/tle-developer-flagos/references/workflow-templates.md)，即性能记录、保留／回退和经验总结；它不等于远端 KernelGen 服务。
- [KernelGenBench](github/KernelGenBench-main/README.md)：评测框架、任务、测试，以及**实际的 Agent 启动、验证反馈循环和 Ascend 适配代码**，不只是评测说明。

**不能笼统说“KernelGen 没有源码”，要区分平台与评测仓库。** 平台的生成、优化、特化 Skill 调用需要 Token 的远端 MCP；复核主分支、2.0／2.1／2.2 相关分支、发布附件和 [Gitee 镜像](https://gitee.com/flagos-ai/KernelGen)，仍未找到可本地部署的完整生成服务后端。[官方 Skill 接入说明](https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos)

KernelGenBench 则公开了 `agent_bench/methods/normal_cc`、`normal_opencode` 和 `tools/verify_single.py`：启动外部 Agent，让它写代码、运行验证、根据错误修改，再返回精度和加速比。`sota_agents/` 还收录了 AutoKernel、AKO4ALL、cuda-optimized-skill；**这些是被测方法及评测接入，不是 KernelGen 平台后端。** [Agent 代码](github/KernelGenBench-main/agent_bench/methods)、[验证入口](github/KernelGenBench-main/agent_bench/tools/verify_single.py)、[第三方来源](github/KernelGenBench-main/THIRD_PARTY_NOTICES.md)

**按最新明确的 RQ1 口径，KernelGen 可以作为在线平台对比对象。** 只需提交任务并取得算子实现，再在我们的 910 环境统一评测；不要求公开其后端，也不要求平台自带 910。官方文档明确支持自定义需求、在提示词中指定生成设备，以及下载代码。已有 Ascend／910B3 材料可作为适配参考，但生成代码能否在指定机器正确、高效运行，要由实验判断。[提交需求](https://docs.flagos.io/projects/kernelgen/en/latest/web_user_guide/generate-triton-kernels-through-your-operator-definitions.html)、[下载实现](https://docs.flagos.io/projects/kernelgen/en/latest/web_user_guide/view-genertion-results-and-select-optimal-triton-kernel.html)、[910B3 示例](github/skills-main/skills/kernelgen-flagos/kernelgen-specialize.md)

两篇论文在 `papers/`，见[论文与开发主线](03_FlagOS论文与开发主线.md)。下载来源、版本和校验值见[下载记录](下载记录.json)。

# 沐曦股份联合上海人工智能实验室发布高性能GPU算子生成系统Kernel-Smith



近日，上海人工智能实验室和[沐曦股份](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=沐曦股份&zhida_source=entity)联合发布了**高性能GPU算子生成系统—— Kernel-Smith**。



Kernel-Smith创新性地将“稳定评估驱动的进化智能体”与“面向进化的后训练范式”深度融合，依托上海人工智能实验室书生大模型[Intern-S1-Pro](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=Intern-S1-Pro&zhida_source=entity)的基座能力进行深度定制化训练，让大模型真正化身为“[算子优化](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=算子优化&zhida_source=entity)大师”。



**在双方的合作中，沐曦团队全程深度参与。**联合创始人、[CTO](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=CTO&zhida_source=entity)兼首席软件架构师[杨建](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=杨建&zhida_source=entity)、研发副总裁黄向军、AI部门的武亚光、董华楠、[谢佳](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=谢佳&zhida_source=entity)形均参与了该项目并做出贡献。

![img](https://pic3.zhimg.com/v2-a060dcce105338b84d7db4c799b1f8bc_1440w.jpg)

沐曦自研软件栈MXMACA展现出了深度兼容能力。在MXMACA后端测试中，Kernel-Smith 表现出色。研究团队在四类常用算子上对比了不同模型生成高性能 MXMACA 算子的能力，结果显示，Kernel-Smith-MACA-30B 的平均加速比超过了 [DeepSeek](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=DeepSeek&zhida_source=entity)-v3.2 和 [Qwen3](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=Qwen3&zhida_source=entity)-235B-2507 等大参数量的开源模型，而 Kernel-Smith-MACA-235B 取得了进一步性能提升，验证了 Kernel-Smith 框架支持异构平台的能力。

![img](https://picx.zhimg.com/v2-67eafc6bfe910285281981e7cab14939_1440w.jpg)

目前，Kernel-Smith 自动生成的高性能算子已在实际场景中得到应用：不仅加速了 DeepSeek 新架构 Engram，并合入 DLBlas；还落地主流生产级推理引擎 SGLang 和 LMDeploy，实现了大模型自动生成算子从实验室受控评估，到前沿模型研发与生产级部署的双重跨越。



**技术报告链接**：

*[https://arxiv.org/pdf/2603.28342](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2603.28342)*



**在线体验链接**：

*[https://chat.intern-ai.org.cn/k](https://link.zhihu.com/?target=https%3A//chat.intern-ai.org.cn/kernel-smith)*

## **算子开发的两大挑战**



在当今的大模型时代，高性能 GPU 算子（Kernel）是将硬件算力转化为实际吞吐量的核心引擎。无论是支撑 Megatron、vLLM、LMDeploy 等底层系统，还是驱动 AI for Science (AI4S) 的复杂科学计算，高效的算子实现都是释放硬件潜能的重中之重。



然而，尽管大模型的编程能力日益强大，但让其自主生成[高性能算子](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=2&q=高性能算子&zhida_source=entity)并稳定应用于真实生产环境，仍是一个未被全面攻克的行业难题。



当前的算子开发高度依赖工程师的经验。一个高效的算子往往需要在众多融合模式、Tiling 策略等实现方案中反复搜索与调试，而现有基于 LLM 的算子生成系统，多依赖多轮对话或基于历史的 Agent 循环，这带来了两大挑战：



**路径依赖与试错成本高**： 传统的 Debug 过程容易让模型“锚定”在早期的错误决策上，限制了探索的多样性。



**“写对”不等于“跑得快”**： 功能上的正确性与极致的性能是两种完全不同的能力。模型不仅需要一次性生成正确的代码，更需要具备在测试阶段通过持续迭代，不断提升算子性能的能力。



## **两大创新设计**



为了解决这些问题，研究团队创新地提出一个统一的进化智能体与强化训练框架。



**核心设计一：构建稳定评估驱动的进化智能体。**



进化搜索天然适合算子优化，因为它可以通过维护一个候选程序池，在多轮迭代中不断累积性能增益。然而，这一过程对“评测方差”极其敏感：如果 GPU 运行时间测量存在噪声，智能体可能会误删极具潜力的算子，或者保留次优解，这种错误会在代际之间不断放大。



为此，Kernel-Smith 在智能体设计上将“评测稳定性”放在首位。通过固定计算图、重复测量以及异常值剔除等机制，大幅抑制了计时噪声，确保了进化搜索动态的可靠性。同时，研究团队还为 [NVIDIA Triton](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=NVIDIA+Triton&zhida_source=entity) 和 [MetaX](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=MetaX&zhida_source=entity) Maca GPU 构建了专属的后端评估服务，提供编译、正确性和加速比的结构化执行反馈。



**核心设计二：化长为短，面向进化的后训练策略。在模型训练层面，Kernel-Smith 将训练定义为进化循环中的“局部优化器（Local Improver）”。**



具体而言，研究团队将长周期的进化轨迹转化为以“步骤”为中心的监督与强化学习信号。算法只保留那些“在保证正确性的前提下，带来了高收益性能提升”的修改步骤。通过这一设计，Kernel-Smith 不仅提升了单步修改的代码质量，更大幅提高了进化搜索中性能增益的复合增长率。



## **显著性能优势，超越顶尖闭源模型**



得益于上述两大核心设计，Kernel-Smith 在实战中展现出了显著的性能优势。在统一的进化智能体协议下，Kernel-Smith-235B-RL 在 KernelBench（Nvidia Triton 后端）上实现了整体性能的 SOTA。在严格保证功能正确性的前提下，其平均加速比不仅优于所有开源基线模型，还超越了 Gemini-3.0-pro 和 [Claude](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=Claude&zhida_source=entity)-4.6-opus 等顶尖闭源大模型。



![img](https://pic2.zhimg.com/v2-c0e3655fcc12702c1653c54224484c0d_1440w.jpg)

KernelBench-Triton 评测结果



更重要的是，评测曲线显示，Kernel-Smith-235B-RL 的平均得分增长曲线在整个搜索过程中始终处于领先地位（如下图所示），这充分证明了该模型能够最有效地利用测试时算力（Test-time Compute）实现性能跃升。



![img](https://pic3.zhimg.com/v2-c3c3187c508074bc89eb00d5acea06a6_1440w.jpg)

不同模型使用同样的 KernelSmith Agent Framework，在 KernelBench-Triton 上的迭代增长曲线图



## **走出实验室，赋能前沿创新与生产级应用**



Kernel-Smith 生成的高性能优化算子不仅成功赋能前沿架构创新，加速了 DeepSeek 新架构 Engram 并合入 DLBlas 开源算子库 ；更重要的是，它已顺利落地主流生产级推理引擎，分别为 SGLang 优化了 [FlashAttention](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity) 后端的 normal_decode_set_metadata 算子，并为 LMDeploy 优化了 DeepSeek MoE Routing 算子，真正实现了大模型自动生成算子从实验室受控评估，到前沿模型研发与生产级部署的双重跨越。



相关 Pull Request 链接：



*[https://github.com/DeepLink-org/DLBlas/pull/102](https://link.zhihu.com/?target=https%3A//github.com/DeepLink-org/DLBlas/pull/102)*



*[https://github.com/sgl-project/sglang/pull/20778](https://link.zhihu.com/?target=https%3A//github.com/sgl-project/sglang/pull/20778)*



*[https://github.com/InternLM/lmdeploy/pull/4345](https://link.zhihu.com/?target=https%3A//github.com/InternLM/lmdeploy/pull/4345)*



在 NV-H200 硬件环境下的算子隔离评测中，Kernel-Smith 展现出显著的 Test-Time Scaling 效应：随着演化迭代的深入，算子性能实现持续增长。





![img](https://pic4.zhimg.com/v2-d7da4c26b5bd152ccdfee0410f55c3df_1440w.jpg)

算子进化迭代曲线图



在主流推理引擎的实际落地中，Kernel-Smith 自动生成的算子为 SGLang 和 LMDeploy 分别带来了 4.78x 和 1.36x 的真实加速收益。在 DeepSeek Engram 复杂场景的深度探索中，模型成功跨越局部最优解，触发了从 5 倍到 12 倍以上的突破性性能跃升，最终达到 14.59x 加速。



![img](https://pic2.zhimg.com/v2-1e1cfdbfdec26f27b7de689da773dec1_1440w.jpg)

算子隔离评测结果



同时，研究团队在社区率先将自动化算子评测扩展至端到端模型吞吐。Kernel-Smith 生成的算子不仅稳定提升了 LMDeploy 的端到端吞吐（最高约3%），还可靠地降低了 SGLang 的真实服务延迟。这验证了生成代码在复杂系统中的[鲁棒性](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=鲁棒性&zhida_source=entity)，也为自动化算子在生产级引擎中的端到端集成提供了可行的实践参考。



![img](https://pic3.zhimg.com/v2-756edb2d1bf57123413d7ce918ae97c8_1440w.jpg)

LMDeploy 端到端模型吞吐评测

## **结语**



沐曦股份坚持“自主创新+开放兼容”的双轨并行路线，在底层硬件完成突破的同时，致力于围绕[MXMACA软件栈](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=MXMACA软件栈&zhida_source=entity)， 打造一个开放、自主、全栈兼容的智能计算生态，公司的目标是将建成人工智能时代的“Android”。



目前，MXMACA软件栈不仅深度兼容主流GPU生态，还支持40多种AI框架，覆盖训练、推理、科学计算全场景，可大福降低开发者的生态迁移成本。同时可支持500多款AI模型、4500多个开源项目软件兼容测试，覆盖95%的主流AI场景。此前已支持国内多款AI模型的“Day0适配”、实现即插即用。MXMACA开源社区也于去年开放，截至2026年3月13日，软件栈注册用户已超过30万人。



此次合作，既是MXMACA软件栈生态的价值体现，也为公司持续深耕AI算力基础设施、推动[高性能计算技术](https://zhida.zhihu.com/search?content_id=272667508&content_type=Article&match_order=1&q=高性能计算技术&zhida_source=entity)产业化升级奠定了坚实基础，未来双方将继续深化协同，依托Kernel-Smith与MXMACA的技术合力，为AI大模型、科学计算等领域提供更高效、更可靠的算力解决方案。
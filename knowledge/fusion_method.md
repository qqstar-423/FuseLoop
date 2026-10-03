# Triton Ascend 融合算子方案

本目录为 Stage1.5、开发、分析和决策节点提供 **10 类方法、24 个变体**。具体选项与稳定 ID 在 `knowledge/fusion_options.json`，以当前任务、芯片和 triton-ascend 能力选用，不预先规定哪种方法一定更快。

这些是设计选项，不是已验证 Triton 性能结论。旧 PyPTO 材料由用户另存备份，不作为当前框架的 API、能力或成绩。Jev 为每类独立评分，概率不必相加为 1，不等于实测收益；暂时不可表达的方法仍保留 ID，但必须说明能力缺口，不能仅凭高概率要求开发实现。

## 共同前提

- 数学语义、接口、给定 case 与精度要求不变。核心计算由自定义 `@triton.jit` 实现；host 不代算，不复用旧输入的结果。
- 每个 program 的访存遵守真实 stride、边界 mask、归约中性值和累加精度。grid/分块结合实际 Vector/Cube 核数与存活数据量选择。
- 单 kernel 不保证中间结果全在片上；多 kernel 不等于失败。只有实际编译与 profiler 才能证明物化、溢出、搬运和计算单元。
- L2 Cache 不是 DSM；容量不保证驻留，不保证不落 HBM。独立 program/stream 不能假定存在任意全局同步或跨核共享地址空间。
- GPU 专用通信、warp/SM 调度和论文原型不自动适用于昇腾 Triton。所有 API、流水选项和同步机制以安装版本与实验证据核对。
- 正式比较须同框架/后端、硬件、case、基准和计时口径；自测结果、论文数字、Jev 概率不能充当正式性能成绩。

## F1: Kernel Fission / Partial Fusion（拆分与半融合）

把复合计算拆成多个 kernel，每段选择合适的布局和分块，段内仍可融合。例如卷积 kernel 写工作区，激活 kernel 再读工作区并写输出。

适合全融合资源过重、不同阶段最优分工不同的情况。代价是工作区读写与额外启动；各 kernel 仍有真实数据依赖，不能宣称没有同步成本。Triton 中显式分配工作区，保证初始化与执行顺序，整体工作都计入评测。

变体：

- `F1.compute_epilogue_split`：计算与后处理拆分.
- `F1.partial_fusion_partitions`：分段内部融合.
- `F1.primitive_orchestration`：原语拆分后重新编排.

## F2: Vertical Fusion（垂直融合）

将相邻生产者和消费者放在同一 kernel 的计算过程中，例如矩阵计算后直接做偏置和激活，再写输出。

适合依赖、布局和分块可兼容的链路。可减少物化，但中间张量、归约或同步可能增大资源需求。对 `tl.dot` 及 CV 融合核实后端支持，不能从源码里“只有一次 launch”推断实际无 HBM 中间读写。

变体：

- `F2.compute_epilogue`：计算与后处理同核.
- `F2.loop_chain`：兼容循环链融合.
- `F2.tiled_pipeline`：按块生产消费流水.

## F3: Horizontal Fusion（水平融合）

把没有数据依赖的分支组织在一次 kernel 调用中，以减少启动或改善资源利用；不同 program 可负责不同合法区域。

适合许多小分支或互补工作。需要检查不同区域的工作量、分支条件与资源需求；两个独立分支放在同一文件或 wrapper 里不算完成水平融合，GPU 线程/warp 分区方法不能直接照搬。

变体：

- `F3.complementary_regions`：互补独立区域.
- `F3.small_branches`：多个独立小分支.

## F4: Input / Output Fusion（输入输出融合）

多个分支共享输入时复用加载，或生产者直接计算多个任务要求的输出，减少重复读写。

适合多消费者、多输出图。先确认复用范围、广播/布局和输出接口，再评估增加的活跃张量。单一卷积加激活链不因名字包含“输入输出”就自动适合这类。

变体：

- `F4.shared_input`：共享输入分支.
- `F4.producer_multi_consumer`：生产者与多个消费者.
- `F4.multi_output`：任务要求的多输出.

## F5: Hybrid Routing（按形状路由）

针对不同 shape/dtype/合法元数据选择不同的 kernel 或分块，例如部分 case 用 F1，另一些用 F2；也可以在同一融合方式内选择不同 BLOCK/grid。

这是多种具体方案的选择方式。路由条件必须来自输入元数据，完整覆盖任务，不按测试编号或答案分流；同 shape 更换数据必须仍正确。每个分支都要编译、自测和统一评测，保留通用路径处理未专门优化的合法输入。

变体：

- `F5.fusion_boundary_routing`：不同融合边界路由.
- `F5.schedule_routing`：调度与分块路由.
- `F5.specialized_with_fallback`：专用实现加通用路径.

## F6: Multi-stage Pipeline with Inter-core On-chip Sharing（跨核片上共享流水）

让多个阶段通过硬件与后端确实支持的跨核片上通信交换块数据，减少全局工作区物化。

这是强能力依赖方案。必须证明共享范围、地址可见性、同步与进度机制均可由当前 triton-ascend 表达，并覆盖正确性；仅有 L2 容量、stream 或普通 program_id 不满足前提。不能确认支持时应明确不可直接实现，可研究 F1/F2/F5，不能改名假装已有 DSM。

变体：

- `F6.cooperative_shared_tiles`：协作阶段共享数据块.
- `F6.cross_kernel_sharing`：有明确支持的跨kernel共享.

## F7: Split and Fusion Co-optimization（拆分融合协同搜索）

枚举合法拆分和融合边界，以成本估计和真实评测筛选组合；搜索出来的执行方案通常由其他方法组成。

适合多阶段算子存在多个合法分区的情况。搜索须保证依赖、精度和资源限制，控制编译/评测成本；不能跳过困难 case，也不能把估计速度写成实测。它改变选方案的方法，不新增或绕过本 workflow 的 Stage。

变体：

- `F7.reuse_guided_partition`：按数据复用搜索分区.
- `F7.cost_guided_search`：按成本搜索合法组合.

## F8: Joint Compute-Memory Fusion（计算访存联合融合）

联合考虑计算与布局变换、加载、归约或后处理的调度，使可重叠的工作合理配合，减少不必要的中间物化。

适合搬运和计算互相制约的链路。先确认当前后端是否能表达所需流水与布局；写在同一 kernel 不代表编译器自动并发，也不保证消除 event/同步。用编译结果、时间线或其他有效指标验证，与普通 F2 区分实际调度差异。

变体：

- `F8.compute_memory_pipeline`：计算与访存子图流水.
- `F8.back_to_back_compute`：相邻计算阶段联合调度.

## F9: Deep Kernel Fusion（长链深度融合）

把较长的依赖链或带共享分支的子图合并，尽量消除多次中间读写。

适合分块后活跃中间数据和精度约束仍可控制的区域。算子数量多本身不是优势；长链可能带来资源溢出、重复计算、复杂同步或错误归约。必须与合理的部分融合方案比较，不能为了“全融合”强制替换更快的方案。

变体：

- `F9.long_chain`：长依赖链.
- `F9.branched_region`：有共享分支的深度区域.

## F10: Multi-device Compute-Communication Fusion（多设备通算融合）

把任务本身要求的多卡/多节点通信与计算按块组织，研究在满足通信语义的前提下重叠执行。

单设备计算任务通常没有选择它的依据。需要实际拓扑、通信 API、可见性、进度和死锁检查；CUDA/NVLink/RDMA 原型不能自动视为昇腾后端可用。所有必需通信和准备成本纳入同口径评测，不擅自改变任务设备范围。

变体：

- `F10.single_node`：节点内通算流水.
- `F10.multi_node`：跨节点通算流水.

## 如何比较

先确认图依赖与能力前提，再按 case 分组提出可实现方案。Stage2 从 Jev 高概率且可实现的候选起步；后续 Stage3/7/8/9 结合实际代码、当前方案依据和评测证据决定保留或调整。少量 case 慢优先定位局部分块、边界和固定开销，不因均值停滞就自动更换整个融合方案。

所有尝试仍写入既有的本轮融合库、决策依据、自测、正式评测与 history 文件；保持 Stage 路由和退出规则。

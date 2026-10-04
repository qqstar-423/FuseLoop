# Triton Ascend Fusion Operator Plans

This directory provides **10 method classes with 24 variants** for the Stage1.5, development, analysis and decision nodes. Concrete options and stable IDs are in `knowledge/fusion_options.json`; choose based on the current task, chip and triton-ascend capabilities — no method is pre-declared to always be faster.

These are design options, not verified Triton performance conclusions. Old PyPTO materials were backed up separately by the user and do not serve as the current framework's APIs, capabilities, or results. Jev scores each class independently; probabilities need not sum to 1 and do not equal measured gains; methods that are currently inexpressible keep their IDs, but the capability gap must be stated — implementation cannot be demanded on high probability alone.

## Common Preconditions

- Math semantics, interfaces, given cases and precision requirements stay unchanged. Core computation is implemented by custom `@triton.jit`; the host does not compute on the kernel's behalf and does not reuse results from old inputs.
- Each program's memory accesses follow real strides, boundary masks, reduction neutral values and accumulation precision. grid/tiling are chosen against the actual Vector/Cube core counts and live data volume.
- A single kernel does not guarantee intermediate results stay on chip; multiple kernels do not mean failure. Only actual compilation and profiler evidence can prove materialization, spills, transfers and compute units.
- The L2 Cache is not DSM; capacity does not guarantee residency and does not guarantee staying off HBM. Independent programs/streams cannot assume arbitrary global synchronization or a cross-core shared address space.
- GPU-specific communication, warp/SM scheduling and paper prototypes do not automatically apply to Ascend Triton. All APIs, pipelining options and synchronization mechanisms must be verified against the installed version and experimental evidence.
- Official comparison requires the same framework/backend, hardware, cases, baseline and timing protocol; self-test results, paper numbers and Jev probabilities cannot serve as official performance results.

## F1: Kernel Fission / Partial Fusion (fission and partial fusion)

Split a compound computation into multiple kernels, choosing a suitable layout and tiling for each segment; fusion within a segment is still allowed. For example, a convolution kernel writes a workspace, and an activation kernel then reads the workspace and writes the output.

Suitable when full fusion is too resource-heavy or when different stages have different optimal divisions of labor. The cost is workspace read/write and extra launches; kernels still have real data dependencies, so claiming there is no synchronization cost is not allowed. In Triton, allocate the workspace explicitly, guarantee initialization and execution order, and count all work in the evaluation.

Variants:

- `F1.compute_epilogue_split`: split computation and epilogue.
- `F1.partial_fusion_partitions`: fusion within segments.
- `F1.primitive_orchestration`: decompose into primitives, then re-orchestrate.

## F2: Vertical Fusion (vertical fusion)

Place adjacent producers and consumers inside the same kernel's computation, e.g., after matrix computation, do bias and activation directly, then write the output.

Suitable for chains whose dependencies, layouts and tiling are compatible. It can reduce materialization, but intermediate tensors, reductions or synchronization may increase resource demand. Verify backend support for `tl.dot` and CV fusion; do not infer from "only one launch" in the source code that there are no actual HBM intermediate reads/writes.

Variants:

- `F2.compute_epilogue`: computation and epilogue in the same kernel.
- `F2.loop_chain`: fusion of compatible loop chains.
- `F2.tiled_pipeline`: block-wise producer-consumer pipelining.

## F3: Horizontal Fusion (horizontal fusion)

Organize branches without data dependencies into a single kernel invocation to reduce launches or improve resource utilization; different programs may handle different legal regions.

Suitable for many small branches or complementary work. Check the workload, branch conditions and resource needs of the different regions; putting two independent branches in the same file or wrapper does not count as horizontal fusion, and GPU thread/warp partitioning methods cannot be copied directly.

Variants:

- `F3.complementary_regions`: complementary independent regions.
- `F3.small_branches`: multiple independent small branches.

## F4: Input / Output Fusion (input/output fusion)

When multiple branches share inputs, reuse loads; or a producer directly computes several outputs required by the task, reducing duplicate reads/writes.

Suitable for multi-consumer, multi-output graphs. First confirm the reuse scope, broadcast/layout and output interfaces, then evaluate the added live tensors. A single conv-plus-activation chain is not automatically suitable for this class just because the name contains "input/output".

Variants:

- `F4.shared_input`: branches sharing inputs.
- `F4.producer_multi_consumer`: producer with multiple consumers.
- `F4.multi_output`: the multiple outputs required by the task.

## F5: Hybrid Routing (shape-based routing)

Choose different kernels or tiling for different shape/dtype/legal metadata, e.g., some cases use F1 and others use F2; you may also choose different BLOCK/grid within one fusion method.

This is a selection mechanism over multiple concrete plans. Routing conditions must come from input metadata, must fully cover the task, and must not branch by test index or answer; with the same shape but different data, correctness must still hold. Every branch must be compiled, self-tested and evaluated under the same protocol; keep a generic path to handle legal inputs that were not specially optimized.

Variants:

- `F5.fusion_boundary_routing`: routing across different fusion boundaries.
- `F5.schedule_routing`: routing of scheduling and tiling.
- `F5.specialized_with_fallback`: specialized implementations plus a generic path.

## F6: Multi-stage Pipeline with Inter-core On-chip Sharing (inter-core on-chip shared pipeline)

Have multiple stages exchange tile data via inter-core on-chip communication that the hardware and backend genuinely support, reducing global workspace materialization.

This is a strongly capability-dependent plan. You must prove that sharing scope, address visibility, synchronization and progress mechanisms can all be expressed by the current triton-ascend and that correctness is covered; L2 capacity, streams or ordinary program_id do not satisfy the preconditions. When support cannot be confirmed, state clearly that it cannot be directly implemented; F1/F2/F5 may be explored, but you cannot rename something and pretend DSM already exists.

Variants:

- `F6.cooperative_shared_tiles`: cooperating stages sharing data tiles.
- `F6.cross_kernel_sharing`: cross-kernel sharing with explicit support.

## F7: Split and Fusion Co-optimization (split-fusion co-optimization search)

Enumerate legal splits and fusion boundaries, screening combinations with cost estimates and real evaluation; the resulting execution plan is usually composed of other methods.

Suitable when a multi-stage operator has multiple legal partitions. The search must respect dependencies, precision and resource constraints and control compilation/evaluation cost; you cannot skip difficult cases, and estimated speed cannot be written up as measured. It changes how plans are selected; it does not add to or bypass this workflow's Stages.

Variants:

- `F7.reuse_guided_partition`: search partitions by data reuse.
- `F7.cost_guided_search`: search legal combinations by cost.

## F8: Joint Compute-Memory Fusion (joint compute-memory fusion)

Jointly consider the scheduling of computation and layout transforms, loads, reductions or epilogues so that overlapping-capable work cooperates reasonably, reducing unnecessary intermediate materialization.

Suitable for chains where transfers and computation constrain each other. First confirm whether the current backend can express the required pipelining and layout; being written in the same kernel does not mean the compiler automatically overlaps, nor does it guarantee eliminating events/synchronization. Verify with compilation results, the timeline or other valid metrics, and distinguish the actual scheduling difference from ordinary F2.

Variants:

- `F8.compute_memory_pipeline`: compute and memory-subgraph pipelining.
- `F8.back_to_back_compute`: jointly scheduling adjacent compute stages.

## F9: Deep Kernel Fusion (long-chain deep fusion)

Merge longer dependency chains or subgraphs with shared branches, eliminating as many intermediate reads/writes as possible.

Suitable for regions where, after tiling, live intermediate data and precision constraints remain controllable. Many operators is not an advantage in itself; long chains may bring resource spills, repeated computation, complex synchronization or wrong reductions. It must be compared against reasonable partial-fusion plans; you cannot force-replace a faster plan for the sake of "full fusion".

Variants:

- `F9.long_chain`: long dependency chains.
- `F9.branched_region`: deep regions with shared branches.

## F10: Multi-device Compute-Communication Fusion (multi-device compute-communication fusion)

Organize the multi-device/multi-node communication required by the task itself together with computation by tiles, exploring overlapped execution while satisfying communication semantics.

Single-device compute tasks usually have no basis for choosing it. Requires an actual topology, communication APIs, visibility, progress and deadlock checks; CUDA/NVLink/RDMA prototypes cannot automatically be assumed available on the Ascend backend. All required communication and preparation costs are included in a same-protocol evaluation; do not change the task's device scope on your own.

Variants:

- `F10.single_node`: intra-node compute-communication pipelining.
- `F10.multi_node`: cross-node compute-communication pipelining.

## How to Compare

First confirm graph dependencies and capability preconditions, then group by case and propose implementable plans. Stage2 starts from candidates with high Jev probability that are implementable; later Stage3/7/8/9 decide to keep or adjust based on the actual code, the current plan's rationale and evaluation evidence. When a few cases are slow, first localize per-case tiling, boundaries and fixed overhead; do not automatically switch the entire fusion plan just because the mean stagnates.

All attempts are still written into the existing fusion library, decision rationale, self-tests, official evaluation and history files for this round; keep Stage routing and exit rules.

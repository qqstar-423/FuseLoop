# Triton Ascend Architecture Programming Reference

The target is **Triton / triton-ascend** on Ascend NPU. Hardware model, core counts, memory capacities, backend and device IDs follow this prompt and `device_info.json`; the installed version determines the available APIs, and unknown capabilities must be verified.

## Basic Writing Style

- Use `import triton` and `import triton.language as tl`; place the core computation in an `@triton.jit` function and launch it via `kernel[grid](...)`.
- `tl.program_id` distinguishes compute blocks and `tl.arange` generates in-block indices; compute addresses from the real shape/stride, and load/store masks cover tail blocks and padding.
- `tl.constexpr` marks a compile-time parameter; it does not mean parameters may be fixed arbitrarily. Tile sizes, accumulation dtype and reduction neutral values must all correspond to the task.
- The host only prepares metadata, output/workspace buffers and the kernel launch; it must not complete the core computation with torch/aclnn and wrap it in an empty kernel.

## Ascend Adaptation

1. **Device**: use the logical NPU ID specified by the program. For self-tests you may read `WORKFLOW_NPU_DEVICE_ID` and call `torch.npu.set_device` before creating test tensors; existing inputs use the same device per their `device` attribute. Do not reset the `ASCEND_*VISIBLE_DEVICES` mapping, and do not use `.cuda()` or CUDA synchronization interfaces.
2. **grid**: distinguish the number of programs from the number of physical cores. Choose based on the actual Vector/Cube core counts, block counts and case sizes; if necessary, each program loops over multiple blocks. GPU experience with scheduling many programs cannot be taken directly as NPU-optimal.
3. **On-chip resources**: consider the shape, dtype, layout and buffers of all live tensors together. UB/L1/L0 capacities come from the real chip, and the compiler may insert copies or spills; the actual residency location cannot be inferred from source variables alone.
4. **Compute units**: elementwise/reduction work on Vector is normal; for matrix work, study the `tl.dot` supported by the current backend. Cube/Vector fusion and special layouts depend on backend support; whether they actually take effect requires compilation results and profiler evidence.
5. **Synchronization**: independent programs cannot assume a global barrier or an addressable shared temporary block. Pass data across kernels via workspaces and a valid execution order; verify the version and applicability before using stream concurrency, atomics and communication extensions.
6. **Compilation options**: select pipelining/autotune and other options per the installed triton-ascend documentation. Do not copy CUDA's warp, SM, Tensor Memory Accelerator or DSM configurations, and do not invent low-level compiler switches.

## Self-Test Focus

Cover all given cases, tail blocks, non-contiguous strides (when the task allows), dtypes, reduction error, and input/output aliasing constraints. With the same shape, swap inputs, weights and biases consecutively and compare against golden each time. Memory allocations may be reused; old computation results may not. The first call triggers JIT; successful installation/import is not the same as compilation and precision passing.

Single kernel, multi-kernel, partial fusion and shape-based routing can all be explored; the final judgment uses real evaluation under one protocol. A shared L2 Cache is not DSM; large capacity does not guarantee cache residency or eliminate HBM reads/writes.

## References

- [Triton Ascend Quick Start](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md): environment and NPU examples.
- [Triton Ascend Programming Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/programming_guide/index.md): grid, memory and compute organization; check against your installed version when using it.
- Project `example/`: the cann-bench Triton Ascend package example actually linked this time; use it as a reference for installation and interfaces; the example's performance does not represent the target task's performance.

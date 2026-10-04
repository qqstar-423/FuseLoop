# Triton Ascend Execution and Evaluation Checks

This project requires the core computation to be performed by a custom `@triton.jit` NPU kernel. The cann-bench report is the source of truth for scoring; source-code compliance, precision, real execution and performance validity must be checked separately — renaming kernels or checking names alone does not prove compliance.

## Handle Each Actual Error

| Situation | What to Check | Action |
|---|---|---|
| `no_npu_kernel_detected` | Whether the original report explicitly contains no valid NPU time; whether the corresponding kernel CSV, collection logs and actual invocations exist | Investigate kernels that never executed, empty invocations, backend/JIT/import errors and timing issues; do not conclude cheating merely because a report is missing |
| `cpu_fallback_detected` | Whether the original report was triggered by copy events in `api_statistic.csv`; cross-check with the implementation where the data goes within the measured region | Core computation and data must stay on the target NPU; copying to the CPU for results and copying back is forbidden |
| Case compile/run failure | The actual triton import path, the Ascend backend, CANN/torch_npu compatibility, the first JIT error | Fix according to local versions; passing installation/import does not count as first successful compilation |
| Target function missing or a stale package loaded | `cann_bench.__file__`, the interface in the task `proto.yaml` versus the package exports, the actual Python path | Use `python3 -m pip install . --force-reinstall --no-deps` with the same interpreter and verify the actually installed package; clean up build copies within this project that are confirmed broken |
| Wrong output or layout mismatch | Output shape/dtype/stride, task-specified contiguity, tail mask, reduction precision and initialization | Handle addressing/output layout correctly with Triton; do not mechanically append extra torch operators to mask implementation problems |
| Profiler collection/parsing anomaly | Whether the raw collection logs and report are complete | Investigate as an evaluation anomaly; do not summarize performance gains/losses from invalid data |

The error-code descriptions above were verified against the local cann-bench `src/kernel_eval/report/scoring.py`, `base/perf_strategy.py` and actual reports; when versions change, the current evaluation implementation prevails. Kernel count ratios or fixed name prefixes are not scoring rules added by this project.

## Development Self-Checks

1. The core math lives inside `@triton.jit`; the host only prepares metadata, output/workspace buffers and scheduling; no torch/aclnn computing on the kernel's behalf, no CPU fallback, and no calling golden to produce results.
2. Inputs and outputs use the specified logical NPU, and the actual triton backend supports Ascend; do not modify visible-device mappings to bypass the device the program specifies.
3. Installation and verification use the same Python, and the target function export matches the task; check the real import path to avoid a wrong same-named package or stale build copies.
4. The workspace is correctly initialized before being read; masks, strides, accumulation precision and output layout satisfy all given cases.
5. Across repeated calls with the same shape but different data, results are correct every time; caching results computed from old inputs/weights, pre-storing answers by test index, or evading timing is forbidden.
6. The self-test report records real commands, results and failures; official performance uses only same-protocol reports produced by cann-bench; modifying cases, golden, baselines or profiler records is forbidden.
7. Combine the source code with the profiler call chain to identify the kernels that actually executed. `AI_VECTOR_CORE` can correspond to legitimate Vector operators; when auxiliary kernels appear, verify their origin, computational responsibility and time cost — do not assert cheating from the name alone.

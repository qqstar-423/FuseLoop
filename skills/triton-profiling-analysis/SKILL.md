---
name: triton-profiling-analysis
description: Analyze the NPU profiling data produced by cann-bench for Triton Ascend operators and locate bottlenecks; applies to kernel_details, ASCEND_PROFILER_OUTPUT and slow-case analysis.
---

# Triton Ascend Profiling Analysis

Read this round's cann-bench report and Ascend profiler files, and interpret performance in the context of the current Triton implementation. Without real reports, do not fabricate timings or bottlenecks.

## Inputs

Paths are relative to this round's `<work>`; the specific collection directory is located via the report.

- `eval/<iter>/perf_result.json`: scores, slowest cases, `source_json`, `kernel_csv` and the comparison protocol; read first.
- `eval/<iter>/perf_comparison.json`: the program's ruling on cross-round comparability; a new baseline does not count as a gain or loss.
- `eval/<iter>/prof_data/<case_id>/.../ASCEND_PROFILER_OUTPUT/`: raw evidence from this collection, read in the five-file order below.
- `impl/` and the bound `develop/<iter>/` documents: confirm the real kernel, grid, tiling, mask/stride and fusion intent; historical implementations must not masquerade as the current code.
- `device_info.json`: current chip, core counts and backend capabilities; do not infer resources from other chips' parameters.

## Five-File Workflow

1. **kernel_details.csv**: examine Duration, repeat stability, core counts and compute units for all candidate kernels. Kernel names must correspond to source/JIT/call chain; do not rely on prefixes alone. Vector kernels running on `AI_VECTOR_CORE` is normal; whether matrix computation correctly uses Cube requires evidence.
2. **op_statistic.csv**: quickly locate the main time costs, then verify against the original details; do not only pick the fastest round.
3. **step_trace_time.csv**: interpret the compute/idle ratio together with case size and valid column definitions; do not directly label a low ratio as a specific bottleneck.
4. **api_statistic.csv**: check copies, host calls and the measurement scope; when the report flags a CPU fallback, investigate its origin — a single name cannot replace diagnosis.
5. **trace_view.json**: when the above evidence is insufficient, explain dependencies, waits, launches and overlap; supplementary only, it does not replace this round's `kernel_details` scoring.

When a file or column is missing, state the limitation and investigate per the actual report. Identify the evaluation's own CacheClean; the responsibility, legitimacy and overhead of other auxiliary kernels must be checked in code — do not mechanically judge cheating. Do not average times across kernels for multi-kernel solutions; official time follows the original cann-bench report.

## Output Requirements

Write the analysis into this stage's existing output files: up to 6 of the slowest cases each with a conclusion, evidence and verification steps, while also covering the distribution across all cases and slow-case trends. Read-only inputs; do not change scores, baselines, cases or raw profiler data, and do not decide a semantic exit on your own.

See [references/profiling_guide.md](references/profiling_guide.md), kept in sync with the project's `knowledge/profiling_guide.md`.

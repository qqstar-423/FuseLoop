# Triton Ascend NPU Profiling Data Interpretation Guide

Analyze cann-bench's Ascend profiler data. The current workflow uses the `kernel_details` protocol; baseline, HAP, speedup and score are read directly from the report. Paths and kernel names in this document are structural illustrations; no measured Triton results are provided.

## 1. Locating Evidence from the Report

`lib/bench_parser.py` calls cann-bench → executes the candidate on the specified NPU → `torch_npu.profiler` collects → cann-bench parses the original report → the workflow writes `eval/<iter>/perf_result.json`.

First read `perf_result.json`'s `source_json`, `cases`, `worst_6_cases` and each `kernel_csv`; locate the data through real paths, and do not guess directories from a fixed process name. Warmup/repeat and the device follow this round's command and `comparison_context`. `perf_comparison.json` states whether comparison with previous rounds is valid; with inconsistent protocols, no gain/loss is counted.

```text
eval/<iter>/prof_data/<case_id>/<this collection directory>/ASCEND_PROFILER_OUTPUT/
  kernel_details.csv   per-kernel execution details, primary evidence
  op_statistic.csv     aggregation by operator/kernel, helps locate hotspots
  step_trace_time.csv  overall compute and idle time
  api_statistic.csv    host API calls and copies
  trace_view.json      timeline, related calls and gaps
```

Some profiler versions do not produce all files or columns; state explicitly what is missing and the limits of your conclusions; do not fabricate. Keep the raw binaries and databases for further diagnosis; do not declare an operator failed merely because some auxiliary CSV is absent.

## 2. kernel_details.csv: Examining Execution Kernel by Kernel

| Column | How to Read |
|---|---|
| `Name` | Confirm kernel identity against the source code, JIT artifacts and the call chain; do not rely on fixed prefixes |
| `Duration(us)` | Look at each kernel's time per invocation in the corresponding case and its repeat stability |
| `Wait Time(us)` | Analyze waiting together with the timeline and scheduling; a single column cannot prove intra-chip pipeline bubbles |
| `Block Num` | Interpret together with the task's block count, Vector/Cube core counts and the specific backend; few cores for a small case can be reasonable |
| `Accelerator Core` | Check against the compute unit the algorithm requires; Vector operators running on `AI_VECTOR_CORE` is normal; whether matrix operators use Cube needs further verification |
| `OP State` | Record the actual value; do not judge custom implementation by static/dynamic alone |

First identify the evaluation's own `CannBenchCacheClean` according to cann-bench's rules, then analyze the candidate computation and necessary auxiliary work. When names such as transpose/pad/aclnn appear, determine whether they come from the candidate wrapper layer, data preparation or the evaluation framework; the source code is forbidden from replacing core computation with off-the-shelf operators, but do not judge by name alone.

For multi-kernel solutions, look at the complete workload; do not cherry-pick only the fastest row, and do not average multiple kernels' times and call it one fused invocation's time. The official `elapsed_us` follows this round's cann-bench `kernel_details` parsing result; it aggregates kernel execution time, does not include inter-kernel gaps, and must not be called complete end-to-end latency.

## 3. op_statistic.csv: Finding the Main Time Costs

Locate hotspots by actual time share, then return to the original repeat records. When the mean/max differ substantially, check input correspondence, JIT/warmup, run interference and collection quality; do not filter out unfavorable results to manufacture gains.

## 4. step_trace_time.csv: Compute Versus Idle

Refer to `Computing / Stage` only when column definitions are clear and the denominator is valid. Low ratios may come from small tasks, host scheduling, synchronization or other work; verify with the timeline, and do not treat a single ratio as the complete conclusion on hardware utilization.

## 5. api_statistic.csv: Host and Copies

Check the counts and durations of APIs such as `aclrtMemcpy`, then locate their scope, direction and call origin. If the actual bench reports `cpu_fallback_detected`, handle it per the original error; during diagnosis, distinguish evaluation preparation from the candidate's core computation, and do not classify all copies unconditionally as CPU computing on the kernel's behalf.

## 6. trace_view.json: Supplementary Timeline

Filter events by the current kernel's real identity, device, stream and related calls. Use the timeline to explain launches, waits and cross-kernel dependencies; adjacent events may overlap, so do not assume everything is serial or identify the same invocation by name alone.

Do not use trace events specific to other frameworks as Triton time. Historical `trace_view` results and the current `kernel_details` use different protocols and cannot be compared directly; discuss gains only after the framework/backend, hardware, cases, baselines and timing method are consistent.

## 7. How to Write the Report

For up to 6 of the slowest cases, each write: **conclusion, evidence file/line or event, next verification step**. Then review the distribution across all cases and slow-case trends, distinguishing mask/stride, tiling, fixed overhead, resource pressure and structural issues of the fusion plan. Where proof is lacking, write hypotheses and validation methods; do not give definitive conclusions.

The current plan and design files must correspond to the evaluated code. Self-tests are correctness evidence; Jev probabilities are initial directions; old cases from other frameworks only indicate historical background — none of these can substitute for current measured Triton results.

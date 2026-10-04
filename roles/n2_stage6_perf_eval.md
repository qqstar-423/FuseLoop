# Triton Ascend Operator Performance Evaluation Engineer

> **Note**: this file is the role definition document for stage6, describing the performance evaluation workflow and data formats.
> **Stage6's evaluation execution has been automated by orchestrator code** (direct subprocess calls to cann-bench) and is no longer injected into the agent.
> This file is retained as a framework design reference and is not loaded or executed by the agent.

You are the performance evaluation engineer for Triton Ascend operators. Performance evaluation is executed automatically by the orchestrator via cann-bench.

**Evaluation execution flow (done by orchestrator code, not the agent)**:
1. The orchestrator calls `python3 -m kernel_eval.cli eval` to run the performance evaluation, explicitly passing `--perf-metric-strategy kernel_details` and this run's dedicated `--reports-dir`
2. On evaluation crash, it automatically retries (at most 2 times, 10 seconds apart); each attempt uses a new directory to avoid mixing in old data
3. It deep-copies the complete `prof_data/` in the same directory as this run's report to `eval/<iter>/prof_data/`
4. It parses the in-work data to generate `perf_result.json`, so that `source_csv_dir` and the valid `kernel_csv` both point to this round's copy
5. It detects anti-cheating (score_error_code) and automatically parses kernel_csv

## Program Inputs and Artifacts (documentation, not agent prompts)

The following paths are relative to the current working directory; `<iter>` denotes directory names such as `iter0`, `iter1`, based on the round actually evaluated by the program.

- Input `task/`: linked to the `task_dir` passed by the program; read the interface, cases, and reference implementation to determine this round's evaluation subject; do not modify test definitions.
- Input `impl/`: the implementation installed and passed through this round's precision evaluation; performance data must correspond to the same code version.
- Input `device_info.json`: hardware parameters and Triton Ascend backend information; together with the task, evaluation configuration, and timing strategy recorded by the program, it determines whether scores are comparable.
- Inputs `eval/<iter>/precision_result.json`, `eval/<iter>/precision_binding.json`: the former gives the precision result, the latter binds the code, reports, and evaluation environment; after validating both, the program decides whether the performance record can enter the best library.
- Artifact `eval/<iter>/perf_result.json`: performance summary; first check whether all cases meet the target, `avg_speedup`, and HAP, then use `source_json` to trace the original report.
- Artifact `eval/<iter>/perf_reports/`: per-attempt independent raw output directories and JSON/MD/HTML report entries; use the report designated by this round's `source_json` to check baseline, candidate timings, and scoring.
- Artifact `eval/<iter>/prof_data/`: this round's profiler data; analyze kernel timing from the specific file designated by `kernel_csv` in the summary; do not mix in data from old rounds.
- Preserve the complete directory hierarchy, e.g. `prof_data/level3/fused/<operator>/<case number>/.../kernel_details.csv`. Batch-collected `_batched/` directories are kept as well; when an independent CSV is missing or multiple are found, that case's `kernel_csv` is left empty and logged; do not force a shared CSV onto a case.
- Best-selection-related `selection/current_implementation.json`, `selection/state.json`, `selection/best.json`: respectively the code and self-test evidence binding, the valid evaluation history and window state, and the current best manifest; the program combines these records to decide library entry, ranking, and exit; the specific implementation and reports are read from the `selection/records/<iter>-<fingerprint>/` snapshot referenced by the best manifest.

## Key Metric Meanings

baseline, HAP, timings, and scores are all provided by cann-bench's original evaluation flow; the workflow directly reuses the reports and does not re-measure baseline independently. `kernel_details` aggregates kernel execution time and does not include synchronization/scheduling gaps between kernels; it must not be called complete invocation time, nor be compared directly against old `trace_view` scores.

- **speedup** (per case) = `baseline_perf_us / elapsed_us`. >1 means faster than baseline, <1 means slower
- **score_i** (per-case performance score) = `(T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))`. Saturating metric (0~1)
- **performance_score** = `(Σ score_i / total_cases) × 0.5 × 100`. Full score 50
- **overall_score** = `compilation_score(out of 20) + function_score(out of 30) + performance_score(out of 50)`. Full score 100
- **avg_speedup** = the average speedup over all passing cases
- **perf_pass** = true only if all cases have speedup ≥ 1.0

## perf_result.json Fields

```jsonc
{
  "perf_pass": true/false,
  "overall_score": <float>,
  "performance_score": <float>,
  "avg_speedup": <float>,
  "score_error_code": "<anti-cheating error code, null if none>",
  "score_error": "<anti-cheating error message, null if none>",
  "source_json": "<cann-bench JSON report path>",
  "source_csv_dir": "<work>/eval/<iter>/prof_data",
  "source_md": "<cann-bench Markdown report path>",
  "worst_6_cases": [
    {
      "case_id": "level3/fused_conv_sigmoid_1",
      "case_num": "1",
      "baseline_us": 3.75,
      "elapsed_us": 35.3,
      "speedup": 0.1062,
      "kernel_csv": "<work>/eval/<iter>/prof_data/level3/fused_conv_sigmoid/1/.../kernel_details.csv"
    }
  ]
}
```

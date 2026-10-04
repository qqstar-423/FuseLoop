## Current Scenario: Zero Score or Evaluation Anomaly

First read the actual error code and report source in `eval/<iter>/perf_result.json`, then check the logs and kernel CSVs of the corresponding cases in `eval/<iter>/perf_reports/` and `eval/<iter>/prof_data/`, and check the code and rules. Missing reports, collection anomalies, or zero times do not by themselves prove cheating.

For `no_npu_kernel_detected`, distinguish: if the CSV and code truly contain only chained ready-made torch/aclnn operators, require rewriting the core computation with `@triton.jit`; if a custom kernel exists but elapsed=0, first investigate the profiler, environment, or timing. Give a fix plan per the actual error, stating which conclusions are confirmed and which evidence still needs to be gathered.

This scenario does not do ordinary slow-case tuning, does not summarize performance up/down from invalid results, and does not trigger stagnation consultation. Keep the necessary correctness and anti-cheating constraints.

## 当前场景：零分或评测异常

先读 `eval/<iter>/perf_result.json` 的实际错误码与报告来源，回查 `eval/<iter>/perf_reports/`、`eval/<iter>/prof_data/` 对应 case 的日志和 kernel CSV，再核对代码及规则。缺失报告、采集异常或时间为零不能直接认定作弊。

`no_npu_kernel_detected` 需要区分：CSV 与代码确实只有 torch/aclnn 现成算子拼接时，要求用 `@triton.jit` 重写核心计算；存在自定义 kernel 但 elapsed=0 时，先查 profiler、环境或计时。按实际错误给修复计划，说明已证实结论与尚需补充的证据。

本场景不做普通慢 case 调优，也不以无效结果总结性能涨跌或触发停滞求助。保留必要正确性和反作弊约束。

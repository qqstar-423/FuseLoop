## 性能证据口径

性能数据沿用 cann-bench 报告。Stage6 选择 `kernel_details`，baseline、HAP 和评分由工具负责，没有独立 baseline 复测；该口径汇总 kernel 执行耗时，不包含核间同步或调度间隔。历史 `trace_view` 成绩不能直接与本轮比较，也不能据此判断涨跌。

程序在 `eval/iter<iter>/perf_result.json` 保存本次测量的硬件、任务和评测协议，在 `eval/iter<iter>/perf_comparison.json` 保存前后两轮是否可比及原因。先遵守这个结论：口径不同、旧报告缺口径、case 不完整或基准不同，不用两轮均值差推断涨跌，不填写本次成功/退步经验。完整的新结果可以作为新基线；不得从旧 history 的数字自行补出“提升/退步”，也不得把当前环境口径当作旧报告的口径。

正式汇总在 `eval/<iter>/perf_result.json`；按其中 `source_json`、`kernel_csv` 追溯原始报告和 case。结合代码与 profiler 验证中间数据所在层级、搬运量和开销，不以 kernel 数量占比或零 HBM 写回来代替性能证据。若连续调用暴露旧数据缓存，优先修正确性，不能用曾经通过的 case 证明当前实现有效。

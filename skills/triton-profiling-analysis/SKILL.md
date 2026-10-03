---
name: triton-profiling-analysis
description: 分析 Triton Ascend 算子在 cann-bench 中产出的 NPU profiling 数据，定位瓶颈；适用于 kernel_details、ASCEND_PROFILER_OUTPUT 和慢 case 分析。
---

# Triton Ascend Profiling 分析

读取本轮 cann-bench 报告及昇腾 profiler 文件，结合当前 Triton 实现解释性能。没有真实报告时不编造耗时或瓶颈。

## 输入

路径相对本次 `<work>`，具体采集目录由报告定位。

- `eval/<iter>/perf_result.json`：成绩、最慢 case、`source_json`、`kernel_csv` 和比较口径，先读。
- `eval/<iter>/perf_comparison.json`：程序对跨轮可比性的裁定；新基线不当作涨跌。
- `eval/<iter>/prof_data/<case_id>/.../ASCEND_PROFILER_OUTPUT/`：本次采集原始证据，按下列五文件顺序读取。
- `impl/` 与绑定的 `develop/<iter>/` 文档：确认真实 kernel、grid、分块、mask/stride 和融合意图；历史实现不冒充当前代码。
- `device_info.json`：当前芯片、核数和后端能力，不用其他芯片参数推断资源。

## 五文件流程

1. **kernel_details.csv**：看所有候选 kernel 的 Duration、重复稳定性、核数与计算单元。核名与源码/JIT/调用链对应，不能只靠前缀。Vector kernel 使用 `AI_VECTOR_CORE` 正常；矩阵计算是否正确使用 Cube 需证据。
2. **op_statistic.csv**：快速定位主要耗时，再回原明细核实，不能只选最快一轮。
3. **step_trace_time.csv**：结合 case 大小与有效列定义解释计算/空闲比例，不直接把低比值定为某种瓶颈。
4. **api_statistic.csv**：核对拷贝、host 调用和测量范围；报告给出 CPU fallback 时排查来源，不能凭一个名字代替诊断。
5. **trace_view.json**：前述证据不足时解释依赖、等待、启动与重叠；只作补充，不替代本轮 `kernel_details` 评分。

某文件或列缺失时写明限制，按实际报告排查。识别评测自己的 CacheClean；其他辅助 kernel 的职责、合法性与开销须查代码，不机械判作弊。多 kernel 时间不取 kernel 平均，正式耗时以 cann-bench 原报告为准。

## 输出要求

按本阶段原有输出文件写分析：最多 6 个最慢 case 各有结论、证据和验证步骤，同时覆盖全部 case 分布及慢 case 趋势。只读输入，不改变评分、基准、case 或原始 profiler 数据，不自行决定语义退出。

详见 [references/profiling_guide.md](references/profiling_guide.md)，与项目 `knowledge/profiling_guide.md` 保持同步。

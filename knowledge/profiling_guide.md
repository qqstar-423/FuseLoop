# Triton Ascend NPU Profiling 数据解读指南

分析 cann-bench 的昇腾 profiler 数据。当前流程使用 `kernel_details` 口径；baseline、HAP、speedup、score 均直接读取报告。本文中的路径和 kernel 名是结构示意，没有提供任何已测 Triton 成绩。

## 1. 从报告定位证据

`lib/bench_parser.py` 调用 cann-bench → 指定 NPU 上执行候选 → `torch_npu.profiler` 采集 → cann-bench 解析原报告 → workflow 写入 `eval/<iter>/perf_result.json`。

先读 `perf_result.json` 的 `source_json`、`cases`、`worst_6_cases` 和每条 `kernel_csv`；通过真实路径找到数据，不能按固定进程名猜目录。warmup/repeat 和设备以本轮命令与 `comparison_context` 为准。`perf_comparison.json` 说明是否与前轮可比，口径不一致不算涨跌。

```text
eval/<iter>/prof_data/<case_id>/<本次采集目录>/ASCEND_PROFILER_OUTPUT/
  kernel_details.csv   每次 kernel 的执行明细，主证据
  op_statistic.csv     按算子/kernel 汇总，辅助定位热点
  step_trace_time.csv 整体计算与空闲时间
  api_statistic.csv    host API 调用与拷贝
  trace_view.json      时间线、相关调用与间隔
```

部分 profiler 版本不产出所有文件或列，明确缺失项和结论限制；不要伪造。原始二进制与数据库用于进一步诊断时保留，不能仅凭没有某辅助 CSV 就判定算子失败。

## 2. kernel_details.csv：逐 kernel 看执行

| 列 | 读法 |
|---|---|
| `Name` | 对照源码、JIT 产物与调用链确认 kernel 身份；不依赖固定前缀 |
| `Duration(us)` | 看对应 case 每次调用的各 kernel 耗时及重复稳定性 |
| `Wait Time(us)` | 结合时间线与调度分析等待，单列不能证明芯片内部流水气泡 |
| `Block Num` | 结合任务块数、Vector/Cube 核数和具体后端解释；小 case 核少可以合理 |
| `Accelerator Core` | 对照算法所需计算单元；Vector 算子在 `AI_VECTOR_CORE` 执行正常，矩阵算子是否走 Cube 需进一步核实 |
| `OP State` | 记录实际值，不能仅凭 static/dynamic 判定是否自定义实现 |

先按 cann-bench 的规则识别评测自身的 `CannBenchCacheClean`，再分析候选计算与必要辅助工作。出现 transpose/pad/aclnn 等名称时，查明是否来自候选包装层、数据准备或评测框架；源码禁止用现成算子替代核心计算，但不能仅靠名称判断。

多 kernel 方案要看完整工作量，不能只挑最快一行，不能把多个 kernel 耗时取平均当作一次融合调用耗时。正式 `elapsed_us` 以本轮 cann-bench `kernel_details` 解析结果为准；它汇总 kernel 执行时间，不包含 kernel 间隔，不能称为完整端到端延迟。

## 3. op_statistic.csv：找主要耗时

按实际耗时占比定位热点，并回看原始重复记录。均值/最值相差较大时检查输入对应关系、JIT/预热、运行干扰与采集质量；不能自行过滤不利结果来制造收益。

## 4. step_trace_time.csv：看计算与空闲

在列定义明确且分母有效时参考 `Computing / Stage`。低占比可能来自小任务、host 调度、同步或其他工作；结合时间线验证，不能把一个比值直接当作硬件利用率的完整结论。

## 5. api_statistic.csv：看 host 和拷贝

查 `aclrtMemcpy` 等 API 的次数与持续时间，再定位其发生范围、方向和调用来源。实际 bench 若给出 `cpu_fallback_detected`，按原错误处理；诊断时区分评测准备与候选核心计算，不把所有拷贝无条件归为 CPU 代算。

## 6. trace_view.json：补充时间线

按当前 kernel 的真实身份、设备、stream 和相关调用筛选事件。时间线用于解释启动、等待和跨 kernel 依赖；相邻事件可能重叠，不能假定全部串行或仅靠名字相同认定同一次调用。

不使用其他框架专属 trace 事件充当 Triton 耗时。历史 `trace_view` 成绩与当前 `kernel_details` 不同口径，不能直接比较；框架/后端、硬件、case、基准与计时方式一致后才讨论收益。

## 7. 报告怎么写

对最多 6 个最慢 case 各写：**结论、证据文件/行或事件、下一步验证**。再回看全部 case 的分布和慢 case 趋势，区分 mask/stride、分块、固定开销、资源压力与融合方案结构性问题。缺少证明时写假设和验证方法，不给确定结论。

当前方案与设计文件必须对应被评测代码。自测是正确性证据，Jev 概率是初始方向，其他框架旧案例只说明历史背景，均不能替代当前 Triton 实测。

# Triton Ascend 执行与评测检查

本项目要求核心计算由自定义 `@triton.jit` NPU kernel 完成。cann-bench 报告是评分事实来源；源码合规、精度、真实执行和性能有效性要分别检查，不能靠改核名或只看名字证明合规。

## 按实际错误处理

| 情况 | 检查内容 | 处理 |
|---|---|---|
| `no_npu_kernel_detected` | 原报告是否明确无有效 NPU 耗时；对应 kernel CSV、采集日志和实际调用是否存在 | 排查未执行 kernel、空调用、后端/JIT/import 错误及计时问题；不能仅因报告缺失就认定作弊 |
| `cpu_fallback_detected` | 原报告是否根据 `api_statistic.csv` 的拷贝事件触发；结合实现核对测量区域内的数据去向 | 核心计算和数据保留在目标 NPU；禁止拷到 CPU 求结果再拷回 |
| case 编译/运行失败 | triton 实际导入路径、Ascend 后端、CANN/torch_npu 兼容、首个 JIT 错误 | 按本机版本修复，不把安装/import 通过当作首次编译通过 |
| 目标函数找不到或加载旧包 | `cann_bench.__file__`、任务 `proto.yaml` 的接口与包导出、实际 Python 路径 | 同一解释器使用 `python3 -m pip install . --force-reinstall --no-deps`，核对安装后的实际包；清理本工程内确有问题的构建副本 |
| 输出错误或布局不符 | 输出 shape/dtype/stride、任务规定的连续性、tail mask、归约精度与初始化 | 用 Triton 正确处理寻址/输出布局；不机械追加额外 torch 算子掩盖实现问题 |
| profiler 采集/解析异常 | 原始采集日志与报告是否完整 | 按评测异常排查，不根据无效数据总结性能涨跌 |

上述错误码说明根据本地 cann-bench 的 `src/kernel_eval/report/scoring.py`、`base/perf_strategy.py` 与实际报告核对；版本变化时以当前评测实现为准。kernel 数量比例或固定名称前缀不是本项目自行增加的评分规则。

## 开发自检

1. 核心数学计算位于 `@triton.jit` 内，host 仅准备元数据、输出/工作区和调度；没有 torch/aclnn 代算、CPU fallback 或调用 golden 生成结果。
2. 输入和输出使用指定逻辑 NPU，实际 triton 后端支持 Ascend；不修改可见设备映射来绕过程序指定的设备。
3. 安装与验证使用同一 Python，目标函数导出与任务一致；检查真实导入路径，避免错误的同名包或旧构建副本。
4. 工作区在读取前正确初始化；mask、stride、累加精度和输出布局符合全部给定 case。
5. 同 shape 多次调用更换数据，结果每次正确；禁止缓存旧输入/权重的计算结果、按测试编号预存答案或规避计时。
6. 自测报告记录真实命令、结果与失败项；正式性能只使用 cann-bench 产出的同口径报告，禁止修改 case、golden、基准或 profiler 记录。
7. 结合源码与 profiler 调用链识别真正执行的 kernel。`AI_VECTOR_CORE` 可对应合法 Vector 算子，出现辅助 kernel 要核对来源、计算职责和耗时，不能单凭它的名字断言作弊。

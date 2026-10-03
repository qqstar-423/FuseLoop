# Triton Ascend 架构编程参考

目标是昇腾 NPU 上的 **Triton / triton-ascend**。硬件型号、核数、存储容量、后端及设备编号以本次 prompt 和 `device_info.json` 为准；安装版本决定可用 API，未知能力必须验证。

## 基本写法

- 使用 `import triton`、`import triton.language as tl`，核心计算放在 `@triton.jit` 函数中，通过 `kernel[grid](...)` 启动。
- `tl.program_id` 区分计算块，`tl.arange` 生成块内下标；按真实 shape/stride 计算地址，load/store 的 mask 覆盖尾块和 padding。
- `tl.constexpr` 表示编译时参数，不表示参数可以随意固定。分块大小、累加 dtype、归约中性值均应对应任务。
- host 只准备元数据、输出/工作区和 kernel 调用；不能用 torch/aclnn 完成核心计算后包一层空 kernel。

## 昇腾适配

1. **设备**：使用程序指定的逻辑 NPU 编号。自测可读取 `WORKFLOW_NPU_DEVICE_ID`，并在创建测试 tensor 前调用 `torch.npu.set_device`；已有输入按其 `device` 使用同一设备。不要重设 `ASCEND_*VISIBLE_DEVICES` 映射，不使用 `.cuda()` 或 CUDA 同步接口。
2. **grid**：区分 program 数与物理核数。按实际 Vector/Cube 核数、块数与 case 大小选择，必要时每个 program 循环处理多块；GPU 上大量 program 的调度经验不能直接当作 NPU 最优。
3. **片上资源**：同时考虑所有存活张量的 shape、dtype、布局与缓冲。UB/L1/L0 容量来自真实芯片，编译器可能插入搬运或溢出；不能仅凭源码变量推断实际驻留位置。
4. **计算单元**：elementwise/归约使用 Vector 正常；矩阵类可研究当前后端支持的 `tl.dot`。Cube/Vector 融合及特殊布局依赖后端支持，实际是否生效需要编译结果和 profiler 证据。
5. **同步**：独立 program 不能假定有全局屏障或可寻址的共享临时块。跨 kernel 通过工作区和有效执行顺序传递数据；stream 并发、原子操作与通信扩展先核对版本和适用范围。
6. **编译参数**：按已安装 triton-ascend 文档选择流水/autotune 等选项。不照搬 CUDA 的 warp、SM、Tensor Memory Accelerator 或 DSM 配置，不编造底层编译开关。

## 自测重点

覆盖给定全部 case、尾块、非连续 stride（任务允许时）、dtype、归约误差和输入/输出别名约束。相同 shape 连续更换输入、权重及偏置，每次对照 golden。可以复用内存分配，不能复用旧计算结果。首次调用触发 JIT；安装/import 成功不是编译和精度通过。

单 kernel、多 kernel、部分融合及按 shape 路由均可研究；最终依据同口径的真实评测。共享 L2 Cache 不是 DSM，容量大不保证缓存驻留或消除 HBM 读写。

## 资料

- [Triton Ascend 快速开始](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md)：环境与 NPU 示例。
- [Triton Ascend 编程指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/programming_guide/index.md)：grid、存储和计算组织；使用时对照本机版本。
- 项目 `example/`：本次实际链接的 cann-bench Triton Ascend 包示例，参考安装与接口；示例性能不代表目标任务性能。

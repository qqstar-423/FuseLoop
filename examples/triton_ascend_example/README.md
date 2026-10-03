# Triton-Ascend 最小融合算子示例

`fused_add_relu(x, y)` 用一个 Triton kernel 完成相加和 ReLU。目标是 **Linux + 昇腾 NPU**，不是 CUDA，也不是 Windows 模拟。这里提供源代码与真实自测入口，**尚未在本工作区运行 NPU 验证，不声称已有性能收益**。

目录：

```text
build.sh / setup.py               标准 cann_bench wheel 打包
cann_bench/__init__.py            导出 fused_add_relu
cann_bench/fused_add_relu.py      Python 包装与 Triton JIT kernel
task/proto.yaml                   接口与数值语义
task/cases.yaml / golden.py       精度用例与独立参考实现
self_test.py                     真实 NPU 自测，包含连续调用
```

输入 shape、dtype、device 必须相同，不做广播。支持 float16、bfloat16、float32 的有限数值；计算在 float32 进行，输出转回输入 dtype。`task/proto.yaml` 的函数名、参数顺序和返回类型与包导出一致；这是自带的教学任务，不假定它已在外部 cann-bench 注册。

kernel 的 load/store 都使用边界 mask，覆盖不满 1024 元素的尾块。包装函数只检查参数、整理布局、分配输出和启动 kernel；不调用 PyTorch 内置 add/ReLU完成核心计算。每次调用读取当前输入，重新分配输出，不保存权重、输入或中间数据缓存。非连续输入的布局复制属于本次调用成本。Triton 编译缓存可以复用已编译程序。

## 构建与自测

先在目标机器配置相互匹配的 CANN、torch、torch_npu、Triton-Ascend，以及 setuptools、wheel。Triton-Ascend 导入名仍是 `triton`；不能用默认 CUDA 版 Triton 代替。wheel 不安装这些运行时依赖。

在本目录运行：

```bash
bash build.sh
python3 -m pip install --force-reinstall --no-deps dist/cann_bench-*.whl
python3 self_test.py --device-id 0
```

`build.sh` 完成打包；首次执行 kernel 才进行 JIT，因此打包成功不等于编译运行成功。自测会检查实际 backend 为 `npu`，执行零元素、标量、整块、尾块、非连续布局、同 shape 新数据，以及同一输入原地修改后的连续调用。数值检查以独立 PyTorch 参考为准，测试中的参考计算不进入提交包。

## cann-bench 精度检查

从外部 cann-bench 仓库运行，`EXAMPLE` 填本目录的绝对路径：

```bash
EXAMPLE=/absolute/path/to/examples/triton_ascend_example
./scripts/run_evaluation.sh \
  --bench-name cann --source-dir "$EXAMPLE" \
  --task-dir "$EXAMPLE/task" --operator FusedAddRelu \
  --device-id 0 --no-perf
```

本示例没有填写性能 baseline。正式性能结论需要目标评测环境提供同口径基准后再测试。项目主流程默认仍参考外部 `cann-bench/examples/triton_ascend_cann_example`，本目录是额外的最小融合示例。

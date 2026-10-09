# Triton-Ascend Minimal Fused Operator Example

`fused_add_relu(x, y)` performs the addition and ReLU in one Triton kernel. The target is **Linux + Ascend NPU**, not CUDA and not a Windows emulation. This directory provides the source code and a real self-test entry point; **it has not yet been NPU-verified in this workspace and claims no performance gains**.

Layout:

```text
build.sh / setup.py               standard cann_bench wheel packaging
cann_bench/__init__.py            exports fused_add_relu
cann_bench/fused_add_relu.py      Python wrapper and Triton JIT kernel
task/proto.yaml                   interface and numeric semantics
task/cases.yaml / golden.py       precision cases and independent reference implementation
self_test.py                      real NPU self-test, including consecutive calls
```

Inputs must share the same shape, dtype and device; no broadcasting. Supports finite values of float16, bfloat16 and float32; the computation is done in float32 and the output is cast back to the input dtype. The function name, argument order and return type in `task/proto.yaml` match the package exports; this is a self-contained teaching task and is not assumed to be registered in an external evaluation suite.

All kernel load/store operations use boundary masks covering tail blocks of fewer than 1024 elements. The wrapper only checks arguments, arranges the layout, allocates the output and launches the kernel; it does not call PyTorch's built-in add/ReLU to complete the core computation. Each call reads the current inputs and reallocates the output; no cache of weights, inputs or intermediate data is kept. Layout copies for non-contiguous inputs count toward that call's cost. Triton's compilation cache may reuse compiled programs.

## Build and Self-Test

First set up mutually matching CANN, torch, torch_npu and Triton-Ascend, plus setuptools and wheel, on the target machine. Triton-Ascend's import name is still `triton`; the default CUDA Triton cannot substitute for it. The wheel does not install these runtime dependencies.

Run in this directory:

```bash
bash build.sh
python3 -m pip install --force-reinstall --no-deps dist/cann_bench-*.whl
python3 self_test.py --device-id 0
```

`build.sh` only does the packaging; the kernel is JIT-compiled on first execution, so packaging success does not mean compile/run success. The self-test checks that the actual backend is `npu`, and exercises zero elements, scalars, full blocks, tail blocks, non-contiguous layouts, fresh data with the same shape, and consecutive calls after in-place modification of the same input. Numerical checks use an independent PyTorch reference; the reference computation in the tests is not part of the submitted package.

## Precision Evaluation

Run from the evaluation suite's repository, with `EXAMPLE` set to the absolute path of this directory:

```bash
EXAMPLE=/absolute/path/to/examples/triton_ascend_example
./scripts/run_evaluation.sh \
  --bench-name cann --source-dir "$EXAMPLE" \
  --task-dir "$EXAMPLE/task" --operator FusedAddRelu \
  --device-id 0 --no-perf
```

This example has no performance baseline filled in. Official performance conclusions require a same-protocol baseline from the target evaluation environment before testing. The project's main workflow still defaults to the external `cann-bench/examples/triton_ascend_cann_example`; this directory is an additional minimal fusion example.

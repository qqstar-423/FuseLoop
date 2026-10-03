---
name: cann-bench-baseline-gen
description: 为 cann-bench 算子生成 baseline 性能数据（baseline_perf_us + t_hw_us）。用户提到"生成 baseline"、"跑基线"、"metadata JSON"、"baseline_perf_us"、"性能基准"时触发。
---

# cann-bench Baseline 性能数据生成

为 cann-bench 算子在当前 NPU 芯片上测量 golden 实现的 baseline 性能，输出 `metadata/<芯片名>.json`。

## 适用场景

- 新芯片（如 Ascend910B3）上没有 baseline 数据，cann-bench 性能评分全部为 0
- 需要为 `bench_lab/` 或 `tasks/` 下的算子补充 baseline

## 快速使用

### 1. 单个算子

```bash
# 在 pypto-pro-workflow 目录下，需要 CANN 环境
python3 tools/gen_baseline.py \
  --task-dir /path/to/cann-bench/bench_lab/.../level3/<operator>
```

自动检测芯片名，输出到 `task-dir` 上级的 `metadata/<芯片名>.json`。

### 2. 多个算子（追加模式）

```bash
# 第一个算子
python3 tools/gen_baseline.py \
  --task-dir /path/to/level3/fused_conv_sigmoid

# 追加第二个算子到同一个 JSON
python3 tools/gen_baseline.py \
  --task-dir /path/to/level3/another_op \
  --merge
```

### 3. 指定输出路径

```bash
python3 tools/gen_baseline.py \
  --task-dir /path/to/task \
  --output /path/to/metadata/Ascend910B3.json
```

### 4. 自定义参数

```bash
python3 tools/gen_baseline.py \
  --task-dir /path/to/task \
  --warmup 5 --trials 20 \
  --device-id 0 \
  --thw-ratio 0.1
```

## 工作原理

### 测量方法

与 `950pr.json` 一致：`torch.npu.Event` 设备端计时。

```
每个 case:
  1. warmup N 次（默认 5），不计时
  2. 正式跑 M 次（默认 20），每次用 Event 记录设备耗时
  3. 取 M 次的中位数作为 baseline_perf_us
  4. t_hw_us = max(baseline_perf_us × 0.1, 1.0μs)
```

Golden 函数从算子目录的 `golden.py` 动态加载，函数名从 `proto.yaml` 的 `schema` 字段解析。

### 输出格式

```json
{
  "_metadata": {
    "hardware": "Ascend910B3",
    "source": "gen_baseline.py",
    "baseline_source": { "measured": "torch.npu.Event ..." }
  },
  "level3": {
    "fused_conv_sigmoid": {
      "1": { "baseline_perf_us": 215.67, "t_hw_us": 21.57 },
      "2": { "baseline_perf_us": 213.98, "t_hw_us": 21.40 }
    }
  }
}
```

与 cann-bench `BaselineStore` 的 `metadata/<hardware>.json` schema 完全兼容。

### 芯片名映射

cann-bench 通过 `baseline_resolver.py` 的 `resolve_hardware()` 把芯片名映射为文件名前缀：

| 芯片报告名 | 映射结果 | 文件名 |
|---|---|---|
| Ascend910B2 | 910b2 | metadata/910b2.json |
| Ascend950PR_xxx | 950pr | metadata/950pr.json |
| Ascend910B3 | Ascend910B3 (无别名) | metadata/Ascend910B3.json |

**如果新芯片没有映射**，`resolve_hardware` 返回原始名称。脚本会提示你确认文件名是否正确。

需要添加映射时，编辑 `cann-bench/src/kernel_eval/utils/baseline_resolver.py`：

```python
PLATFORM_ALIAS = {
    ...
    "Ascend910B3": "910b3",   # 新增
}
```

然后输出文件名变为 `metadata/910b3.json`。不加映射也行，直接用 `Ascend910B3.json`。

## 端到端验证

生成 baseline 后，跑一次性能评测确认 speedup 不再为 0：

```bash
PYTHONPATH=/path/to/cann-bench/src:$PYTHONPATH \
python3 -m kernel_eval.cli eval \
  --bench-name cann \
  --task-dir /path/to/task \
  --device-id 0 \
  --warmup 2 --repeat 3 \
  --perf-metric-strategy kernel_details
```

检查报告中 `baseline_perf_us > 0` 且 `speedup > 0`。

## 前置条件

- NPU 可用（`npu-smi info`）
- `torch` + `torch_npu` 可导入
- `PyYAML` 已安装
- 算子目录包含 `golden.py`、`proto.yaml`、`cases.yaml`

## 常见问题

**Q: baseline 全是 0**
A: `metadata/<芯片名>.json` 不存在或芯片名不匹配。运行脚本生成，或检查 `baseline_resolver.py` 映射。

**Q: golden 函数调用失败**
A: 检查 golden.py 的参数签名是否与 cases.yaml 的 `input_shape`/`dtype`/`attrs` 一致。不同算子的 golden 接口可能不同。

**Q: t_hw_us 不准**
A: 默认 `baseline × 0.1` 是粗估。精确 t_hw 需要按 roofline 模型计算（参考 `hap-ascend-910b2-v2` skill）。

## 文件

| 文件 | 说明 |
|---|---|
| `tools/gen_baseline.py` | 测量脚本（入口） |
| `cann-bench/.../metadata/<chip>.json` | 输出的 baseline 数据 |
| `cann-bench/.../baseline_resolver.py` | 芯片名 → 文件名映射（可能需要修改） |

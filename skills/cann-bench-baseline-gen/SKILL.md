---
name: cann-bench-baseline-gen
description: Generate baseline performance data for a cann-bench operator (baseline_perf_us + t_hw_us). Triggered when the user mentions "generate baseline", "run the baseline", "metadata JSON", "baseline_perf_us" or "performance benchmark".
---

# cann-bench Baseline Performance Data Generation

Measure the baseline performance of the golden implementation for a cann-bench operator on the current NPU chip, and write `metadata/<chip_name>.json`.

## Applicable Scenarios

- A new chip (e.g., Ascend910B3) has no baseline data and all cann-bench performance scores are 0
- Baselines need to be added for operators under `bench_lab/` or `tasks/`

## Quick Use

### 1. Single operator

```bash
# Run in the pypto-pro-workflow directory; a CANN environment is required
python3 tools/gen_baseline.py \
  --task-dir /path/to/cann-bench/bench_lab/.../level3/<operator>
```

The chip name is auto-detected and output goes to `metadata/<chip_name>.json` next to the `task-dir`.

### 2. Multiple operators (append mode)

```bash
# First operator
python3 tools/gen_baseline.py \
  --task-dir /path/to/level3/fused_conv_sigmoid

# Append a second operator to the same JSON
python3 tools/gen_baseline.py \
  --task-dir /path/to/level3/another_op \
  --merge
```

### 3. Specify the output path

```bash
python3 tools/gen_baseline.py \
  --task-dir /path/to/task \
  --output /path/to/metadata/Ascend910B3.json
```

### 4. Custom parameters

```bash
python3 tools/gen_baseline.py \
  --task-dir /path/to/task \
  --warmup 5 --trials 20 \
  --device-id 0 \
  --thw-ratio 0.1
```

## How It Works

### Measurement method

Consistent with `950pr.json`: `torch.npu.Event` device-side timing.

```
Per case:
  1. warmup N times (default 5), untimed
  2. run M times officially (default 20), recording device time with an Event each time
  3. take the median of the M runs as baseline_perf_us
  4. t_hw_us = max(baseline_perf_us × 0.1, 1.0μs)
```

The golden function is loaded dynamically from the operator directory's `golden.py`, and the function name is parsed from the `schema` field of `proto.yaml`.

### Output format

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

Fully compatible with the cann-bench `BaselineStore` `metadata/<hardware>.json` schema.

### Chip name mapping

cann-bench maps chip names to filename prefixes via `resolve_hardware()` in `baseline_resolver.py`:

| Chip report name | Mapping result | Filename |
|---|---|---|
| Ascend910B2 | 910b2 | metadata/910b2.json |
| Ascend950PR_xxx | 950pr | metadata/950pr.json |
| Ascend910B3 | Ascend910B3 (no alias) | metadata/Ascend910B3.json |

**If a new chip has no mapping**, `resolve_hardware` returns the original name. The script will prompt you to confirm whether the filename is correct.

To add a mapping, edit `cann-bench/src/kernel_eval/utils/baseline_resolver.py`:

```python
PLATFORM_ALIAS = {
    ...
    "Ascend910B3": "910b3",   # new entry
}
```

The output filename then becomes `metadata/910b3.json`. It also works without adding a mapping, using `Ascend910B3.json` directly.

## End-to-End Verification

After generating the baseline, run a performance evaluation once to confirm speedup is no longer 0:

```bash
PYTHONPATH=/path/to/cann-bench/src:$PYTHONPATH \
python3 -m kernel_eval.cli eval \
  --bench-name cann \
  --task-dir /path/to/task \
  --device-id 0 \
  --warmup 2 --repeat 3 \
  --perf-metric-strategy kernel_details
```

Check that the report has `baseline_perf_us > 0` and `speedup > 0`.

## Prerequisites

- NPU available (`npu-smi info`)
- `torch` + `torch_npu` importable
- `PyYAML` installed
- The operator directory contains `golden.py`, `proto.yaml`, `cases.yaml`

## FAQ

**Q: baseline is all 0**
A: `metadata/<chip_name>.json` does not exist or the chip name does not match. Run the script to generate it, or check the `baseline_resolver.py` mapping.

**Q: golden function call fails**
A: Check whether golden.py's parameter signature matches cases.yaml's `input_shape`/`dtype`/`attrs`. Golden interfaces may differ across operators.

**Q: t_hw_us is inaccurate**
A: The default `baseline × 0.1` is a rough estimate. Precise t_hw requires a roofline-model calculation (see the `hap-ascend-910b2-v2` skill).

## Files

| File | Description |
|---|---|
| `tools/gen_baseline.py` | Measurement script (entry point) |
| `cann-bench/.../metadata/<chip>.json` | Output baseline data |
| `cann-bench/.../baseline_resolver.py` | Chip name → filename mapping (may need modification) |

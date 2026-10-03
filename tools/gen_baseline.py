#!/usr/bin/env python3
"""Generate baseline_perf_us + t_hw_us for cann-bench operators on current NPU.

Usage:
    # Single operator (bench_lab or tasks)
    python3 tools/gen_baseline.py \\
        --task-dir /path/to/cann-bench/bench_lab/.../level3/fused_conv_sigmoid

    # Specify output (default: auto-detect metadata dir + chip name)
    python3 tools/gen_baseline.py \\
        --task-dir /path/to/task \\
        --output /path/to/metadata/Ascend910B3.json

    # Override warmup/trials
    python3 tools/gen_baseline.py --task-dir /path/to/task --warmup 5 --trials 20

    # Merge into existing JSON (add more operators)
    python3 tools/gen_baseline.py --task-dir /path/to/another_op --merge

Methodology: torch.npu.Event device elapsed time (warmup + trials, median).
t_hw_us = max(baseline_perf_us * 0.1, 1.0).
Output JSON format matches cann-bench metadata/<hardware>.json schema.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def detect_chip() -> str:
    """Return NPU chip name as reported by torch_npu."""
    import torch_npu
    return torch_npu.npu.get_device_properties(0).name


def find_metadata_dir(task_dir: Path) -> Optional[Path]:
    """Walk up from task_dir to find the nearest metadata/ directory."""
    current = task_dir.parent
    for _ in range(10):
        candidate = current / "metadata"
        if candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def parse_level_op(task_dir: Path) -> Tuple[str, str]:
    """Extract (level_key, op_name) from task path like .../level3/fused_conv_sigmoid."""
    op_name = task_dir.name
    level_dir = task_dir.parent.name
    m = re.match(r"level(\d+)", level_dir)
    if m:
        return f"level{m.group(1)}", op_name
    return level_dir, op_name


def load_golden(task_dir: Path) -> Any:
    """Dynamically import golden.py and return the golden function."""
    golden_path = task_dir / "golden.py"
    if not golden_path.exists():
        raise FileNotFoundError(f"golden.py not found: {golden_path}")

    proto_path = task_dir / "proto.yaml"
    func_name = None
    if proto_path.exists():
        import yaml
        with open(proto_path) as f:
            proto = yaml.safe_load(f)
        schema = proto.get("schema", "")
        m = re.match(r"^(\w+)\s*\(", schema.strip())
        if m:
            func_name = m.group(1)

    spec = importlib.util.spec_from_file_location("_golden", str(golden_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if func_name and hasattr(module, func_name):
        return getattr(module, func_name)

    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_"):
            return obj
    raise RuntimeError(f"No callable found in {golden_path}")


def load_cases(task_dir: Path) -> List[Dict]:
    """Load cases from cases.yaml."""
    import yaml
    with open(task_dir / "cases.yaml") as f:
        data = yaml.safe_load(f)
    return data["cases"]


DTYPE_MAP = {
    "float16": torch.float16, "float32": torch.float32,
    "bfloat16": torch.bfloat16, "int8": torch.int8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
}


def build_tensor(shape, dtype_str, value_range, device):
    dtype = DTYPE_MAP.get(dtype_str, torch.float32)
    lo, hi = value_range if value_range else [-1, 1]
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device).uniform_(lo, hi)


def build_inputs(case: Dict, device: str) -> Tuple[List, Dict]:
    """Build input tensors and attrs from a case dict."""
    shapes = case["input_shape"]
    dtypes = case["dtype"]
    ranges = case.get("value_range", [])
    attrs = case.get("attrs", {})

    tensors = []
    for i, (shape, dt) in enumerate(zip(shapes, dtypes)):
        if shape is None or dt is None:
            tensors.append(None)
            continue
        vr = ranges[i] if i < len(ranges) and ranges[i] else None
        if isinstance(shape[0], list):
            tensor_list = [build_tensor(s, d, vr[j] if vr and j < len(vr) else None, device)
                           for j, (s, d) in enumerate(zip(shape, dt))]
            tensors.append(tensor_list)
        else:
            tensors.append(build_tensor(shape, dt, vr, device))

    return tensors, attrs


def measure_case(golden_fn, case: Dict, device_id: int,
                 warmup: int, trials: int) -> float:
    """Measure one case, return median elapsed_us."""
    device = f"npu:{device_id}"
    tensors, attrs = build_inputs(case, device)

    def run():
        return golden_fn(*tensors, **attrs)

    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    times = []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)

    return round(statistics.median(times), 2)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline JSON for cann-bench operators")
    parser.add_argument("--task-dir", required=True, help="Path to operator task directory")
    parser.add_argument("--output", help="Output JSON path (default: auto-detect)")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--merge", action="store_true",
                        help="Merge into existing JSON instead of overwriting")
    parser.add_argument("--thw-ratio", type=float, default=0.1,
                        help="t_hw_us = max(baseline * ratio, 1.0)")
    args = parser.parse_args()

    import torch_npu
    torch.npu.set_device(args.device_id)

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        print(f"ERROR: task dir not found: {task_dir}", file=sys.stderr)
        sys.exit(1)

    chip = detect_chip()
    print(f"Chip: {chip}")
    print(f"Task: {task_dir}")

    level_key, op_name = parse_level_op(task_dir)
    print(f"Operator: {level_key}/{op_name}")

    if args.output:
        out_path = Path(args.output)
    else:
        metadata_dir = find_metadata_dir(task_dir)
        if not metadata_dir:
            print("ERROR: cannot find metadata/ dir; use --output", file=sys.stderr)
            sys.exit(1)
        out_path = metadata_dir / f"{chip}.json"

    golden_fn = load_golden(task_dir)
    cases = load_cases(task_dir)
    print(f"Cases: {len(cases)}, warmup={args.warmup}, trials={args.trials}\n")

    results = {}
    for case in cases:
        cid = case["case_id"]
        try:
            baseline_us = measure_case(golden_fn, case, args.device_id,
                                       args.warmup, args.trials)
            t_hw = round(max(baseline_us * args.thw_ratio, 1.0), 2)
            results[str(cid)] = {"baseline_perf_us": baseline_us, "t_hw_us": t_hw}
            print(f"  case {cid:>2}: baseline={baseline_us:>8.2f} us, t_hw={t_hw:>6.2f} us")
        except Exception as exc:
            print(f"  case {cid:>2}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)

    if not results:
        print("ERROR: no cases measured", file=sys.stderr)
        sys.exit(1)

    if args.merge and out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {
            "_metadata": {
                "description": "cann-bench baseline performance data",
                "hardware": chip,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "gen_baseline.py (torch.npu.Event device elapsed time)",
                "baseline_source": {
                    "measured": f"torch.npu.Event device elapsed time "
                                f"(warmup={args.warmup}, trials={args.trials}, median)",
                },
                "t_hw_us_note": f"t_hw_us = max(baseline_perf_us * {args.thw_ratio}, 1.0us)",
            },
        }

    data.setdefault(level_key, {})[op_name] = results

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    print(f"Measured: {len(results)}/{len(cases)} cases")
    print(f"Operator path in JSON: {level_key}.{op_name}")

    check_resolver(chip)


def check_resolver(chip: str):
    """Check if cann-bench baseline_resolver recognizes this chip."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from kernel_eval.utils.baseline_resolver import resolve_hardware
        resolved = resolve_hardware(chip)
        if resolved == chip:
            print(f"\nNote: baseline_resolver returns '{chip}' as-is (no alias).")
            print(f"  cann-bench will look for metadata/{chip}.json — make sure it exists.")
        else:
            print(f"\nbaseline_resolver maps '{chip}' -> '{resolved}'")
            print(f"  cann-bench will look for metadata/{resolved}.json")
    except ImportError:
        print(f"\nCould not check baseline_resolver. Ensure metadata/{chip}.json is in place.")


if __name__ == "__main__":
    main()

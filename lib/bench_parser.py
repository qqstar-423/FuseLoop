"""
Evaluation result parsing — reads the JSON reports produced by cann-bench and extracts
the fields needed for routing decisions. Supports automatic parsing of both precision
and performance evaluations.
"""

import json
import math
import os
import glob
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .profiler_archive import case_kernel_csv


def parse_precision_result(path: str) -> Dict[str, Any]:
    """Read precision_result.json and return the precision verdict. path is the full file path."""
    if not os.path.exists(path):
        return {"precision_overall": False, "_error": f"file does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_perf_result(path: str) -> Dict[str, Any]:
    """Read perf_result.json and return the performance verdict. path is the full file path."""
    if not os.path.exists(path):
        return {"perf_pass": False, "_error": f"file does not exist: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_build_success(path: str) -> bool:
    """Read build.log and check whether it contains STATUS: SUCCESS. path is the full file path."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return bool(lines) and lines[-1] == "STATUS: SUCCESS"


def analyze_kernel_csv_for_anticheat(perf_result: Dict[str, Any], max_cases: int = 3) -> str:
    """Summarize recorded kernel names, without inferring implementation ownership.

    The evaluator's original error remains the routing fact. This text is only
    evidence for Stage9 to cross-check against source and actual JIT launches.
    """
    import csv
    from collections import Counter
    cases = perf_result.get("worst_6_cases", [])
    if not cases:
        return "(no worst_6_cases data)"
    lines = []
    for index, case in enumerate(cases[:max_cases]):
        path = case.get("kernel_csv", "")
        case_id = case.get("case_id", f"case_{index}")
        if not path or not os.path.exists(path):
            lines.append(f"- {case_id}: kernel_csv does not exist")
            continue
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as stream:
                names = Counter((row.get("Name") or "").strip() for row in csv.DictReader(stream))
            names.pop("", None)
            lines.append(f"- {case_id}: {sum(names.values())} kernel events in total; CSV={path}")
            if names:
                shown = ", ".join(f"{name} ×{count}" for name, count in names.most_common(5))
                lines.append(f"  Sampled names and counts (including tool/auxiliary events; this alone does not establish origin): {shown}")
            else:
                lines.append("  No nonempty kernel names were read; check the collection log against the raw report.")
        except Exception as exc:
            lines.append(f"- {case_id}: csv parsing failed ({exc})")
    lines.append("Names and event counts cannot prove a custom kernel, cheating or successful fusion; verify against the original report's error code, "
                 "the implementation code, @triton.jit and actual kernel[grid](...) calls.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# cann-bench direct execution + report parsing
# ═══════════════════════════════════════════════════════════════

def _find_good_ctypes_dir() -> str:
    """
    Find a lib-dynload directory containing a usable _ctypes.so and put it first on PYTHONPATH.
    Automatically adapts to the current Python version and CPU architecture.
    Only a _ctypes module matching the current interpreter's extension suffix is selected.
    """
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/usr/lib/python{py_ver}/lib-dynload",
        "/usr/lib/python3/dist-packages",
    ]
    for d in glob.glob("/usr/lib/python3.*/lib-dynload"):
        if d not in candidates:
            candidates.append(d)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    for d in candidates:
        ctypes_so = os.path.join(d, f"_ctypes{suffix}")
        if os.path.exists(ctypes_so):
            return d
    return ""


def _build_eval_env(config_path=None) -> Dict[str, str]:
    """Use the same complete CANN environment as startup probes and agents."""
    from .cann_env import build_cann_env
    env = build_cann_env(config_path=config_path)
    good_ctypes = _find_good_ctypes_dir()
    pypath_parts = ([good_ctypes] if good_ctypes else []) + [
        part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    env["PYTHONPATH"] = ":".join(pypath_parts)
    env.setdefault("WORKFLOW_NPU_DEVICE_ID", os.environ.get("WORKFLOW_NPU_DEVICE_ID", "0"))
    env.pop("LD_PRELOAD", None)
    return env


def _find_latest_report(reports_dir: str, op_name: str, suffix: str = ".json", after_ts: float = 0) -> Optional[str]:
    """Find the latest evaluation report in the cann-bench reports directory. With after_ts>0 only files newer than that timestamp are accepted."""
    pattern = os.path.join(reports_dir, f"{op_name}_eval_*{suffix}")
    files = sorted(glob.glob(pattern), reverse=True)
    if after_ts > 0:
        files = [f for f in files if os.path.getmtime(f) > after_ts]
    return files[0] if files else None


def run_perf_eval(task_dir: str, cannbench_src: str, device_id: int = 0,
                  log_callback=None, reports_dir=None, config_path=None) -> Tuple[bool, str, str]:
    """
    Execute the cann-bench performance evaluation directly (not through an agent).
    Uses kernel_details to tally per-kernel execution time of a single call; both timing and
    scoring come from the tool. Returns (success, stdout, report_json_path).
    """
    cmd = [
        "python3", "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--task-dir", task_dir,
        "--device-id", str(device_id),
        "--warmup", "2", "--repeat", "3",
        "--perf-metric-strategy", "kernel_details",
    ]
    source = Path(cannbench_src).resolve()
    reports_dir = str(Path(reports_dir).resolve()) if reports_dir else str(
        (source.parent if source.name == "src" else source) / "reports")
    cmd.extend(["--reports-dir", reports_dir])
    env = _build_eval_env(config_path)
    env["WORKFLOW_NPU_DEVICE_ID"] = str(device_id)
    env["PYTHONPATH"] = f"{cannbench_src}:{cannbench_src}/cann_bench_utils:{env['PYTHONPATH']}"

    if log_callback:
        log_callback(f"Running performance evaluation: {' '.join(cmd[:6])}...")

    import time as _time
    start_ts = _time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cannbench_src, timeout=3600)

    if log_callback:
        log_callback(f"Performance evaluation finished, returncode={proc.returncode}")
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-500:] if proc.stderr else "(empty)"
            stdout_tail = proc.stdout[-500:] if proc.stdout else "(empty)"
            log_callback(f"stderr: {stderr_tail}")
            log_callback(f"stdout: {stdout_tail}")

    # Find a report newly produced by the evaluation (only files newer than start_ts are accepted)
    op_name = Path(task_dir).name.lower()
    report_path = _find_latest_report(reports_dir, op_name, ".json", after_ts=start_ts)

    if proc.returncode != 0 and not report_path:
        if log_callback:
            log_callback(f"Evaluation failed (returncode={proc.returncode}) and produced no new report")

    return proc.returncode == 0, proc.stdout + proc.stderr, report_path or ""


def run_precision_eval(task_dir: str, cannbench_src: str, device_id: int = 0,
                       log_callback=None, reports_dir=None, config_path=None) -> Tuple[bool, str, str]:
    """
    Execute the cann-bench precision evaluation directly (not through an agent).
    Returns (success, stdout, report_json_path).
    """
    cmd = [
        "python3", "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--task-dir", task_dir,
        "--device-id", str(device_id),
        "--no-perf",
    ]
    if reports_dir:
        cmd.extend(["--reports-dir", str(reports_dir)])
    env = _build_eval_env(config_path)
    env["WORKFLOW_NPU_DEVICE_ID"] = str(device_id)
    env["PYTHONPATH"] = f"{cannbench_src}:{cannbench_src}/cann_bench_utils:{env['PYTHONPATH']}"

    if log_callback:
        log_callback(f"Running precision evaluation: {' '.join(cmd[:6])}...")

    import time as _time
    start_ts = _time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cannbench_src, timeout=3600)

    if log_callback:
        log_callback(f"Precision evaluation finished, returncode={proc.returncode}")

    op_name = os.path.basename(task_dir.rstrip("/"))
    reports_dir = reports_dir or os.path.join(cannbench_src, "reports")
    report_path = _find_latest_report(reports_dir, op_name, ".json", after_ts=start_ts)

    return proc.returncode == 0, proc.stdout + proc.stderr, report_path or ""


def parse_cannbench_report_to_perf_result(report_path: str, source_csv_dir: str = "") -> Dict[str, Any]:
    """
    Parse the cann-bench JSON report into the contents of perf_result.json.
    """
    if not report_path or not os.path.exists(report_path):
        return {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                "_error": f"report does not exist: {report_path}"}

    with open(report_path, "r", encoding="utf-8") as stream:
        d = json.load(stream)
    op = d.get("operators", [{}])[0] if d.get("operators") else {}
    cases = op.get("cases", [])

    # Extract metrics
    overall_score = op.get("score", 0.0) or 0.0
    perf_score = op.get("performance_score", 0.0) or 0.0
    avg_speedup = op.get("avg_speedup", 0.0) or 0.0
    score_error_code = op.get("score_error_code")
    score_error = op.get("score_error")

    # Judge only the report\'s complete case set; never re-time or recompute speedup/HAP.
    def positive_finite(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)

    perf_pass = (bool(cases) and len(cases) == op.get("total_cases")
                 and not score_error_code and not score_error
                 and all(c.get("status") == "success"
                         and positive_finite(c.get("elapsed_us"))
                         and positive_finite(c.get("speedup"))
                         and c["speedup"] >= 1.0 for c in cases))

    # worst_6_cases
    sorted_cases = sorted(cases, key=lambda c: c.get("speedup") if positive_finite(c.get("speedup")) else 0)
    full_cases = []
    for c in sorted_cases:
        case_id = c.get("case_id", "?")
        case_num = case_id.split("_")[-1] if "_" in case_id else "?"
        kernel_csv = case_kernel_csv(source_csv_dir, case_id)
        full_cases.append({
            **c,  # keep tool metrics such as perf_score (HAP), t_hw_us and op_times as-is.
            "case_id": case_id,
            "case_num": case_num,
            "baseline_us": c.get("baseline_perf_us", 0),
            "elapsed_us": c.get("elapsed_us", 0),
            "speedup": c.get("speedup", 0),
            "kernel_csv": kernel_csv,
        })

    # source_md
    source_md = report_path.replace(".json", ".md") if report_path.endswith(".json") else ""

    return {
        "perf_pass": perf_pass,
        "overall_score": overall_score,
        "compilation_score": op.get("compilation_score"),
        "compile_runtime_score": op.get("compile_runtime_score"),
        "function_score": op.get("function_score"),
        "performance_score": perf_score,
        "avg_speedup": avg_speedup,
        "total_cases": op.get("total_cases", 0),
        "passed_cases": op.get("passed_cases", 0),
        "failed_cases": op.get("failed_cases", 0),
        "cases": full_cases,
        "score_error_code": score_error_code,
        "score_error": score_error,
        "source_json": report_path,
        "source_csv_dir": source_csv_dir,
        "source_md": source_md,
        "worst_6_cases": full_cases[:6],
    }


def parse_cannbench_report_to_precision_result(report_path: str) -> Dict[str, Any]:
    """
    Parse the cann-bench JSON report into the contents of precision_result.json.
    """
    if not report_path or not os.path.exists(report_path):
        return {"precision_overall": False, "total_cases": 0, "passed_cases": 0,
                "failed_cases": 0, "_error": f"report does not exist: {report_path}"}

    d = json.load(open(report_path, "r", encoding="utf-8"))
    op = d.get("operators", [{}])[0] if d.get("operators") else {}

    total = op.get("total_cases", 0) or 0
    passed = op.get("passed_cases", 0) or 0
    failed = op.get("failed_cases", 0) or 0

    return {
        "precision_overall": passed == total and total > 0,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "source_report": report_path,
    }

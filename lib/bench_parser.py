"""
评测结果解析 — 读 cann-bench 产出的 JSON 报告，提取路由判断所需字段。
支持精度评测和性能评测的自动解析。
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
    """读 precision_result.json，返回精度判定结果。path 为完整文件路径。"""
    if not os.path.exists(path):
        return {"precision_overall": False, "_error": f"文件不存在: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_perf_result(path: str) -> Dict[str, Any]:
    """读 perf_result.json，返回性能判定结果。path 为完整文件路径。"""
    if not os.path.exists(path):
        return {"perf_pass": False, "_error": f"文件不存在: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_build_success(path: str) -> bool:
    """读 build.log，检查是否含 STATUS: SUCCESS。path 为完整文件路径。"""
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
        return "(无 worst_6_cases 数据)"
    lines = []
    for index, case in enumerate(cases[:max_cases]):
        path = case.get("kernel_csv", "")
        case_id = case.get("case_id", f"case_{index}")
        if not path or not os.path.exists(path):
            lines.append(f"- {case_id}: kernel_csv 不存在")
            continue
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as stream:
                names = Counter((row.get("Name") or "").strip() for row in csv.DictReader(stream))
            names.pop("", None)
            lines.append(f"- {case_id}: 共 {sum(names.values())} 条 kernel 事件；CSV={path}")
            if names:
                shown = ", ".join(f"{name} ×{count}" for name, count in names.most_common(5))
                lines.append(f"  采样名称及次数（含工具/辅助事件，不能据此认定来源）: {shown}")
            else:
                lines.append("  未读到非空 kernel 名称；核对采集日志与原始报告。")
        except Exception as exc:
            lines.append(f"- {case_id}: csv 解析失败 ({exc})")
    lines.append("名称和事件数量不能证明自定义 kernel、作弊或融合成功；请结合原报告错误码、"
                 "实现代码、@triton.jit 和 kernel[grid](...) 实际调用核对。")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# cann-bench 直接执行 + 报告解析
# ═══════════════════════════════════════════════════════════════

def _find_good_ctypes_dir() -> str:
    """
    找到一个含可用 _ctypes.so 的 lib-dynload 目录，放到 PYTHONPATH 最前面。
    自动适配当前 Python 版本和 CPU 架构。
    仅选取与当前解释器扩展后缀匹配的 _ctypes 模块。
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
    """在 cann-bench reports 目录找最新的评测报告。after_ts>0 时只接受比该时间戳新的文件。"""
    pattern = os.path.join(reports_dir, f"{op_name}_eval_*{suffix}")
    files = sorted(glob.glob(pattern), reverse=True)
    if after_ts > 0:
        files = [f for f in files if os.path.getmtime(f) > after_ts]
    return files[0] if files else None


def run_perf_eval(task_dir: str, cannbench_src: str, device_id: int = 0,
                  log_callback=None, reports_dir=None, config_path=None) -> Tuple[bool, str, str]:
    """
    直接执行 cann-bench 性能评测（不通过 agent）。
    使用 kernel_details 统计单次调用的各 kernel 执行时间；耗时与评分均由工具产出。
    返回 (success, stdout, report_json_path)
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
        log_callback(f"执行性能评测: {' '.join(cmd[:6])}...")

    import time as _time
    start_ts = _time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cannbench_src, timeout=3600)

    if log_callback:
        log_callback(f"性能评测完成，returncode={proc.returncode}")
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-500:] if proc.stderr else "(空)"
            stdout_tail = proc.stdout[-500:] if proc.stdout else "(空)"
            log_callback(f"stderr: {stderr_tail}")
            log_callback(f"stdout: {stdout_tail}")

    # 找评测后新产出的报告（只接受比 start_ts 新的）
    op_name = Path(task_dir).name.lower()
    report_path = _find_latest_report(reports_dir, op_name, ".json", after_ts=start_ts)

    if proc.returncode != 0 and not report_path:
        if log_callback:
            log_callback(f"评测失败(returncode={proc.returncode})且无新报告产出")

    return proc.returncode == 0, proc.stdout + proc.stderr, report_path or ""


def run_precision_eval(task_dir: str, cannbench_src: str, device_id: int = 0,
                       log_callback=None, reports_dir=None, config_path=None) -> Tuple[bool, str, str]:
    """
    直接执行 cann-bench 精度评测（不通过 agent）。
    返回 (success, stdout, report_json_path)
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
        log_callback(f"执行精度评测: {' '.join(cmd[:6])}...")

    import time as _time
    start_ts = _time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cannbench_src, timeout=3600)

    if log_callback:
        log_callback(f"精度评测完成，returncode={proc.returncode}")

    op_name = os.path.basename(task_dir.rstrip("/"))
    reports_dir = reports_dir or os.path.join(cannbench_src, "reports")
    report_path = _find_latest_report(reports_dir, op_name, ".json", after_ts=start_ts)

    return proc.returncode == 0, proc.stdout + proc.stderr, report_path or ""


def parse_cannbench_report_to_perf_result(report_path: str, source_csv_dir: str = "") -> Dict[str, Any]:
    """
    解析 cann-bench JSON 报告，生成 perf_result.json 的内容。
    """
    if not report_path or not os.path.exists(report_path):
        return {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                "_error": f"报告不存在: {report_path}"}

    with open(report_path, "r", encoding="utf-8") as stream:
        d = json.load(stream)
    op = d.get("operators", [{}])[0] if d.get("operators") else {}
    cases = op.get("cases", [])

    # 提取指标
    overall_score = op.get("score", 0.0) or 0.0
    perf_score = op.get("performance_score", 0.0) or 0.0
    avg_speedup = op.get("avg_speedup", 0.0) or 0.0
    score_error_code = op.get("score_error_code")
    score_error = op.get("score_error")

    # 仅判断报告中的完整 case 集合，不重新计时或计算 speedup/HAP。
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
            **c,  # 原样保留 perf_score(HAP)、t_hw_us、op_times 等工具指标。
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
    解析 cann-bench JSON 报告，生成 precision_result.json 的内容。
    """
    if not report_path or not os.path.exists(report_path):
        return {"precision_overall": False, "total_cases": 0, "passed_cases": 0,
                "failed_cases": 0, "_error": f"报告不存在: {report_path}"}

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

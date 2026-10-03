#!/usr/bin/env python3
"""
Triton Ascend Operator Workflow — 确定性控制平面
========================================
唯一启动入口。Python if/else + while 控制路由，stage1.5 调用 Jev 评分。
阶段切换完全由本程序控制，agent 只执行不判断流程。

用法:
    python orchestrator.py --task-dir /path/to/cannbench/task/<op>
    python orchestrator.py --task-dir /path/to/task --max-iter 20
"""

import argparse
import atexit
import copy
import hashlib
import logging
import math
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.state import State
from lib.handoff import read_file_safe, atomic_write_json, atomic_write_text
from lib.tech_lead import (Stage9DecisionValidationError, merge_tech_lead_update,
                           validate_stage9_request_conditions)
from lib.stage9_plan import (build_decision_schema, build_decision_template,
                            validate_current_plan)
from lib.human_review import HumanReview
from lib.stage9_scenes import classify_scene, build_scene_role, scene_input_keys, is_performance_scene
from lib.stage9_human import (evidence_sources, render_question, human_prompt,
                              development_human_prompt, validate_execution_receipt)
from lib.prompt_files import file_hint
from lib.performance_comparison import compare_performance
from lib.framework_target import (FRAMEWORK, BACKEND, PROGRAMMING_MODEL,
                                  ensure_workflow_target, ensure_example_link, ensure_task_link, assert_target_implementation,
                                  detect_triton_runtime, read_cann_toolchain, validate_device_id)
from lib.knowledge_metadata import build_knowledge_environment, log_knowledge_write
from lib.history_manager import refresh_round_comparison
from lib.fusion_selection import (run_fusion_selection, fusion_library_path,
                                  format_fusion_library_for_prompt,
                                  fusion_requirements_byte_budget)
from lib.fusion_evidence import (development_prompt, begin_development, finalize_development, load_evidence,
                                 format_evidence_for_prompt, implementation_hash, restore_imported_evidence)
from lib.init_impl import (prepare_init_impl, load_init_impl_manifest, validate_init_impl_inputs)
from lib.semantic_exit import (record_evaluation, load_selection_status,
                               format_selection_for_prompt)
from lib.semantic_notices import log_semantic_trigger
from lib.regression_restore import (prepare_regression_action, apply_regression_action,
                                    load_regression_action, regression_action_prompt)
from lib.bench_parser import (parse_precision_result, parse_perf_result, check_build_success,
                              analyze_kernel_csv_for_anticheat, run_perf_eval, run_precision_eval,
                              parse_cannbench_report_to_perf_result, parse_cannbench_report_to_precision_result)
from lib.agent_runner import (run_agent, isolated_agent_configuration,
                              bind_workflow_agent_configuration)
from lib.cann_env import reset_cann_cache
from lib.profiler_archive import archive_profiler_data
from lib.profile_report import (profile_report_path, stamp_stage7_report,
                                validate_profile_report_directory, write_stage9_report)
from lib.logger import setup_logger, setup_state_logger, setup_node_logger, setup_history_logger
from lib.history_manager import load_history, save_history, append_round, append_ledger_only, format_for_prompt, get_latest_fix_plan, append_proven_pattern, load_proven_patterns, format_proven_patterns_for_prompt, append_regression_pattern, load_regression_patterns, format_regression_patterns_for_prompt, append_pitfall, load_pitfalls, format_pitfalls_for_prompt

import yaml
import json as _json_module


def detect_npu_device(device_id=0, config_path=None) -> dict:
    """Read Ascend hardware parameters without importing a kernel DSL runtime."""
    import subprocess as _sp
    import time as _time_mod
    from lib.cann_env import build_cann_env
    validate_device_id(device_id)
    info = {"chip_model": "unknown", "soc_version": "unknown", "npu_arch": "unknown",
            "ai_core_num": 0, "ub_size_kb": 0, "l1_size_kb": 0, "l0a_size_kb": 0,
            "l0b_size_kb": 0, "l0c_size_kb": 0, "l2_size_kb": 0,
            "cube_core_num": 0, "vector_core_num": 0, "device_id": device_id,
            "programming_model": PROGRAMMING_MODEL, "detect_method": "none"}
    env = build_cann_env({"WORKFLOW_NPU_DEVICE_ID": str(device_id)}, config_path=config_path)
    script = (
        "import torch, torch_npu\n"
        f"torch.npu.set_device({device_id})\n"
        f"props = torch_npu.npu.get_device_properties({device_id})\n"
        "print('WORKFLOW_SOC=' + props.name)\n"
        "print('CUBE_CORES=' + str(props.cube_core_num))\n"
        "print('VECTOR_CORES=' + str(props.vector_core_num))\n"
        "print('L2_BYTES=' + str(props.L2_cache_size))\n"
    )
    soc_name = None
    for attempt in range(1, 4):
        try:
            result = _sp.run(["python3", "-c", script], capture_output=True,
                             text=True, timeout=30, env=env)
            if result.returncode == 0:
                fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                soc_name = fields.get("WORKFLOW_SOC", "").strip() or None
                if soc_name:
                    info["soc_version"] = soc_name
                    for key, field, divisor in (("cube_core_num", "CUBE_CORES", 1),
                                                ("vector_core_num", "VECTOR_CORES", 1),
                                                ("l2_size_kb", "L2_BYTES", 1024)):
                        try:
                            info[key] = int(fields[field]) // divisor
                        except (KeyError, ValueError):
                            pass
                    break
            print(f"[芯片检测] 第{attempt}次失败: {result.stderr[-300:]}", file=sys.stderr)
        except Exception as exc:
            print(f"[芯片检测] 第{attempt}次异常: {exc}", file=sys.stderr)
        if attempt < 3:
            _time_mod.sleep(5)
    if not soc_name:
        for key in ("SOC_VERSION", "ASCEND_SOC_VERSION"):
            if os.environ.get(key):
                soc_name = os.environ[key]
                info["soc_version"] = soc_name
                break
    if not soc_name:
        return info
    script = (
        "from tbe.common.platform import get_soc_spec, set_current_compile_soc_info\n"
        f"set_current_compile_soc_info({soc_name!r})\n"
        "for key in ('UB_SIZE', 'L1_SIZE', 'L0A_SIZE', 'L0B_SIZE', 'L0C_SIZE', 'CORE_NUM'):\n"
        " print(key + '=' + str(get_soc_spec(key)))\n"
    )
    try:
        result = _sp.run(["python3", "-c", script], capture_output=True,
                         text=True, timeout=30, env=env)
        if result.returncode == 0:
            fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            for key, field in (("ub_size_kb", "UB_SIZE"), ("l1_size_kb", "L1_SIZE"),
                               ("l0a_size_kb", "L0A_SIZE"), ("l0b_size_kb", "L0B_SIZE"),
                               ("l0c_size_kb", "L0C_SIZE")):
                info[key] = int(fields[field]) // 1024
            info["ai_core_num"] = int(fields["CORE_NUM"])
            info["detect_method"] = "tbe_device_query"
        else:
            print(f"[芯片检测] tbe 查询失败: {result.stderr[-300:]}", file=sys.stderr)
    except Exception as exc:
        print(f"[芯片检测] tbe 查询异常: {exc}", file=sys.stderr)
    # Device-reported SoC is authoritative; names only label the architecture.
    # Never derive core counts or buffer capacities from a chip name.
    soc = soc_name.lower()
    for tokens, architecture in ((("950", "3510"), "dav-3510"),
                                 (("910b", "910c"), "dav-c220"),
                                 (("310p",), "dav-lite"), (("310",), "dav-mini"),
                                 (("910a", "910pro"), "dav-c100")):
        if any(token in soc for token in tokens):
            info["npu_arch"] = architecture
            break
    info["chip_model"] = soc_name
    return info


def format_device_info_for_prompt(device_info: dict) -> str:
    """Show the measured chip and installed Triton Ascend runtime to all stages."""
    versions = device_info.get("runtime_versions", {})
    return (
        "🔧 当前昇腾 NPU 与 Triton Ascend 环境（所有 stage 共享）\n"
        f"  芯片型号: {device_info['chip_model']}\n"
        f"  SoC 版本: {device_info['soc_version']}\n"
        f"  NPU 架构: {device_info['npu_arch']}\n"
        f"  框架: {FRAMEWORK}；后端: {BACKEND}；运行时版本: {_json_module.dumps(versions, ensure_ascii=False)}\n"
        f"  编程模型: {device_info.get('programming_model', PROGRAMMING_MODEL)}\n"
        f"  后端 target: {device_info.get('driver_backend', 'unknown')}/{device_info.get('target_arch', 'unknown')}\n"
        f"  AI Core 数: {device_info['ai_core_num']}（Cube {device_info.get('cube_core_num', '?')} + Vector {device_info.get('vector_core_num', '?')}）\n"
        f"  UB: {device_info['ub_size_kb']} KB；L1: {device_info['l1_size_kb']} KB\n"
        f"  L0A/L0B/L0C: {device_info.get('l0a_size_kb', '?')}/{device_info.get('l0b_size_kb', '?')}/{device_info.get('l0c_size_kb', '?')} KB\n"
        f"  L2 Cache: {device_info.get('l2_size_kb', '?')} KB\n"
        f"  Device ID: {device_info['device_id']}（现有可见设备映射中的逻辑编号）\n"
        f"  检测方式: {device_info['detect_method']}\n"
        "  使用 import triton、import triton.language as tl、@triton.jit 与显式 grid。\n"
        "  BLOCK 分块、mask、stride 和矩阵计算参数须结合这些资源与本机后端支持；片上缓冲区由编译器管理。\n"
        "  不照搬 CUDA warp、shared-memory、PTX 或特定 GPU 的异步指令假设；以已安装 triton-ascend 的 API 为准。\n"
        + file_hint(Path(__file__).parent, Path(__file__).parent / "knowledge/arch_programming_guide.md",
                    "Triton Ascend 架构编程指南", "核对本机 API、分块、边界 mask 和内存访问约束",
                    base_label="项目根目录")
        + "  example/ 示例只用于接口与基本写法参考；分块参数仍需针对当前算子和芯片验证。\n"
    )


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for agent in config.get("agents", {}).values():
            agent["workflow_config_path"] = os.path.abspath(config_path)
            agent["device_id"] = config.get("hardware", {}).get("device_id", 0)
        return config
    return {}


def append_work_record(work_dir: str, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(work_dir, "WORK_RECORD.md")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## [{ts}] {message}\n")


def log_io(log, stage: str, inputs: list, outputs: list):
    """打印阶段的输入/输出文件路径"""
    log.info(f"[{stage}] 输入: {', '.join(inputs)}")
    log.info(f"[{stage}] 预期输出: {', '.join(outputs)}")


def fusion_review_hint(status: dict, iteration: int) -> str:
    """Describe only this iteration's valid underperforming stagnation window."""
    if not (status.get("eligible") and status.get("review_fusion")
            and status.get("latest_iteration") == iteration):
        return ""
    window = status["window"]
    return (
        f"【场景2停滞：重点审查融合方案】仍有 case 的 speedup < 1；"
        f"已连续完成 y={window['required_improvements']} 次有效性能迭代（不含基线），"
        f"窗口 iter{window['start_iteration']}→iter{window['end_iteration']}，"
        f"历史最佳 avg_speedup {window['start_best_avg_speedup']:.6g}→"
        f"{window['end_best_avg_speedup']:.6g}，累计有效提升 "
        f"{window['cumulative_improvement']:.2%} < {window['threshold']:.0%}。\n"
        "请重点比较 JSON 融合候选库及概率、当前融合选择依据、瓶颈分析、搜索结果和未达标 case 趋势，"
        "动态决定保留、局部优化或更换方案，并在原 P0/P1/P2 建议中说明依据。"
        "若少数慢 case 持续改善，可继续局部优化；不得仅因平均提升不足 5% 强制换方案。"
    )


def transition(state, new_stage: str, state_log, reason: str = ""):
    """状态切换 + 写日志（含切换原因）。"""
    old = state.current_stage
    state.current_stage = f"iter{state.iteration}_{new_stage}"
    state.flush()
    reason_text = f" | {reason}" if reason else ""
    state_log.info(f"{old} → {state.current_stage}{reason_text}")


def stage_number(value):
    """Match the complete stage number; stage10 must never match stage1."""
    import re
    match = re.fullmatch(r"(?:iter\d+_)?stage(1\.5|\d+)", value)
    if not match:
        return None
    return 1.5 if match.group(1) == "1.5" else int(match.group(1))


def comparison_context(task_dir, device_info, config):
    """Compare measurements only within the same task, hardware and evaluator protocol."""
    # CANN can change while a long-running workflow remains alive. Re-read only
    # its version metadata at each precision/performance boundary; do not rerun
    # a Torch/NPU probe here, and never mutate an earlier context's evidence.
    hardware = copy.deepcopy(device_info)
    toolchain = hardware.get("toolchain")
    if isinstance(toolchain, dict) and toolchain.get("verified") is True:
        roots = toolchain.get("cann_roots")
        if not isinstance(roots, dict) or not roots:
            raise RuntimeError("CANN 工具链原已验证，但缺少可复核的安装目录；禁止继续比较性能。")
        hardware["toolchain"] = read_cann_toolchain(roots)

    def source_hash(root, suffixes):
        root = Path(root)
        digest = hashlib.sha256()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part in {"__pycache__", ".git", "reports", "build"} for part in relative.parts):
                    continue
                if path.is_file() and path.suffix in suffixes:
                    digest.update(relative.as_posix().encode("utf-8") + b"\0")
                    digest.update(path.read_bytes())
                    digest.update(b"\0")
        return digest.hexdigest()

    bench_root = Path(config.get("paths", {}).get("cannbench_repo", "../cann-bench"))
    if not bench_root.is_absolute():
        bench_root = Path(__file__).resolve().parent / bench_root
    bench_source = bench_root / "src" if (bench_root / "src").is_dir() else bench_root
    # CANN-Bench loads fixed baseline/t_hw assets from the nearest metadata/
    # ancestor, outside the operator task. Changes to these assets change the
    # measurement basis even when task files and evaluator code stay identical.
    metadata_path = None
    current = Path(task_dir).resolve()
    while True:
        if (current / "metadata").is_dir():
            metadata_path = current / "metadata"
            break
        if current == bench_root.resolve() or current.parent == current:
            break
        current = current.parent
    return {
        "framework": FRAMEWORK, "backend": BACKEND,
        "runtime_versions": copy.deepcopy(hardware.get("runtime_versions", {})),
        "hardware": hardware,
        "evaluation_device_id": config.get("hardware", {}).get("device_id", 0),
        "hardware_config": copy.deepcopy(config.get("hardware", {})),
        "task_sha256": source_hash(task_dir, {".md", ".yaml", ".yml", ".py", ".json"}),
        "metric": "kernel_details", "baseline_protocol": "cann-bench-native/kernel-details-v1",
        "warmup": 2, "repeat": 3,
        "evaluator_sha256": source_hash(bench_source / "kernel_eval", {".py"}),
        "baseline_metadata_sha256": source_hash(metadata_path, {".json"}) if metadata_path else None,
    }


def _implementation_hash_or_empty(work_dir):
    try:
        return implementation_hash(Path(work_dir) / "impl")
    except (OSError, ValueError):
        return ""


def _file_revision(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
            hashlib.sha256(path.read_bytes()).hexdigest())


def _record_performance_selection(work_dir, iteration, perf, context, config, before_hash):
    """Bind the precision result and performance run to the tested source version."""
    evaluation_dir = Path(work_dir) / "eval" / f"iter{iteration}"
    precision = parse_precision_result(str(evaluation_dir / "precision_result.json"))
    evidence = load_evidence(work_dir)
    try:
        binding = _json_module.loads((evaluation_dir / "precision_binding.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        binding = {}
    precision_path = evaluation_dir / "precision_result.json"
    precision_sha = hashlib.sha256(precision_path.read_bytes()).hexdigest() if precision_path.is_file() else ""
    after_hash = _implementation_hash_or_empty(work_dir)
    if (binding.get("eligible") is not True or not before_hash or before_hash != after_hash
            or binding.get("impl_sha256") != before_hash
            or binding.get("precision_sha256") != precision_sha
            or binding.get("comparison_context") != context
            or perf.get("comparison_context") != context
            or perf.get("comparison_context_stable") is not True):
        evidence = {**evidence, "eligible": False,
                    "reason": "精度、自测与性能的代码版本或评测口径未一致绑定；本轮不纳入最佳记录和停滞窗口。"}
    return record_evaluation(work_dir, iteration, perf=perf, precision=precision,
                             evidence=evidence, comparison_context=context, config=config)


# cannbot 全局约束（拼在每次 cannbot prompt 末尾）
CANNBOT_CONSTRAINT = (
    "\n⚠️ 约束：所有临时文件、调试脚本必须放在 /tmp 下，禁止在工作目录或项目根目录创建临时文件。\n"
    + file_hint(Path(__file__).parent, Path(__file__).parent / "knowledge/anti_cheat_reference.md",
                "反作弊参考", "开发完成后按实际错误处理表与自检清单检查执行路径和评测行为", base_label="项目根目录")
)


def allocate_work_directory(base_dir, op_name, timestamp):
    """Reserve a new run directory atomically, even for simultaneous launches."""
    if not op_name or op_name in {".", ".."} or any(char in op_name for char in "/\\"):
        raise ValueError("算子名称必须是单个目录名，不能包含路径分隔符")
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{op_name}_{timestamp}"
    candidate = base / stem
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return str(candidate)
        except FileExistsError:
            candidate = base / f"{stem}_{uuid.uuid4().hex[:8]}"


@isolated_agent_configuration
def main():
    parser = argparse.ArgumentParser(description="Triton Ascend Operator Workflow 控制平面")
    parser.add_argument("--task-dir", required=True, help="cannbench task 目录路径")
    parser.add_argument("--op-name", default=None, help="算子名称")
    parser.add_argument("--max-iter", type=int, default=None, help="最大迭代次数")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--work-dir", default=None, help="工作目录")
    parser.add_argument("--init-impl", default=None, help="应急导入已有实现及对应开发材料，新建 work，跳过 Stage1/1.5/2，直接从 Stage4 评测")
    hint_source = parser.add_mutually_exclusive_group()
    hint_source.add_argument("--optimize-hint", default=None,
                             help="新任务注入 Stage1/2；应急导入时保存为后续优化参考；长文本请用 --optimize-hint-file")
    hint_source.add_argument("--optimize-hint-file", default=None,
                             help="完整读取 UTF-8 方向说明文件，与 --optimize-hint 互斥；应急导入仍先评测，不提前修改代码")
    args = parser.parse_args()
    if args.optimize_hint_file:
        try:
            args.optimize_hint = Path(args.optimize_hint_file).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            parser.error(f"--optimize-hint-file 无法读取 UTF-8 文件（{type(exc).__name__}）: {args.optimize_hint_file}")

    if args.init_impl and args.work_dir:
        parser.error("--init-impl 和 --work-dir 不能同时使用：前者新建应急任务，后者恢复现有任务")
    if args.init_impl:
        if not Path(args.init_impl).is_dir():
            parser.error(f"--init-impl 目录不存在: {args.init_impl}")
        assert_target_implementation(args.init_impl)

    reset_cann_cache()
    config = load_config(args.config)
    bind_workflow_agent_configuration(config, args.config)
    max_iterations = args.max_iter if args.max_iter is not None else config.get("workflow", {}).get("max_iterations", 20)
    if max_iterations < 0:
        parser.error("--max-iter must be nonnegative")
    window = config.get("workflow", {}).get("recent_iterations_window", 5)

    task_dir = os.path.abspath(args.task_dir)
    op_name = args.op_name or os.path.basename(task_dir.rstrip("/"))

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = allocate_work_directory(Path(__file__).resolve().parent / "work", op_name, ts)

    ensure_workflow_target(work_dir)
    for subdir in ["impl", "build", "eval", "profile", "search", "develop", "log", "knowledge"]:
        Path(os.path.join(work_dir, subdir)).mkdir(parents=True, exist_ok=True)

    ensure_task_link(work_dir, task_dir)

    cannbench_repo = Path(config.get("paths", {}).get("cannbench_repo", "../cann-bench"))
    if not cannbench_repo.is_absolute():
        cannbench_repo = Path(__file__).resolve().parent / cannbench_repo
    ensure_example_link(work_dir, cannbench_repo)

    optimize_hint = args.optimize_hint or ""

    log = setup_logger(work_dir)
    state_log = setup_state_logger(work_dir)
    history_log = setup_history_logger(work_dir)
    node_logs = {
        "N1": setup_node_logger(work_dir, "N1"),
        "N2": setup_node_logger(work_dir, "N2"),
        "N3": setup_node_logger(work_dir, "N3"),
        "N4": setup_node_logger(work_dir, "N4"),
        "N5": setup_node_logger(work_dir, "N5"),
    }
    log.info("=" * 60)
    log.info("Triton Ascend Operator Workflow 启动")
    log.info(f"算子: {op_name} | Task: {task_dir}")
    log.info(f"Work: {work_dir} | 最大迭代: {max_iterations}")
    if args.init_impl:
        log.info("===== 应急导入已有实现 ===== 来源=%s；新 work=%s", args.init_impl, work_dir)
    elif optimize_hint:
        log.info(f"💡 融合方向提示: {optimize_hint[:200]}...")
    log.info("=" * 60)

    imported = (prepare_init_impl(args.init_impl, work_dir, task_dir,
                                 optimize_hint=optimize_hint, log=log)
                if args.init_impl else load_init_impl_manifest(work_dir))
    if imported:
        log.info("[应急导入] 来源与复制清单：%s", Path(work_dir) / "init_impl_manifest.json")
        if not optimize_hint:
            optimize_hint = imported.get("optimize_hint", "")

    state = State.load_or_create(work_dir, op_name=op_name, max_iterations=max_iterations)
    if stage_number(state.current_stage) == 9 and not state.stage9_context:
        # Older checkpoints already archived the failure reason in each request.
        archives = sorted((Path(work_dir) / "knowledge/stage9" / f"iter{state.iteration}").glob("*/request.json"),
                          key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for archive in archives:
            request = _json_module.loads(archive.read_text(encoding="utf-8"))
            if request.get("iteration") == state.iteration and request.get("fail_reason"):
                state.stage9_context = {"iteration": state.iteration, "fail_reason": request["fail_reason"],
                                        "scene": request.get("scene"), "perf_diff": request.get("perf_diff", {}),
                                        "phase": "reviewing", "message_ids": []}
                break
    human_config = config.get("workflow", {}).get("human_review", {})
    for option in ("proactive_enabled", "consultation_enabled"):
        if option in human_config and type(human_config[option]) is not bool:
            raise ValueError(f"workflow.human_review.{option} must be a boolean")
    # An old custom config without this section retains its unattended behavior.
    state.human_review_config = {
        "proactive_enabled": human_config.get("proactive_enabled", False),
        "consultation_enabled": human_config.get("consultation_enabled", False),
    }
    human = HumanReview(work_dir)
    human.configure(active_enabled=state.human_review_config["proactive_enabled"],
                    consultation_enabled=state.human_review_config["consultation_enabled"])
    state.flush()
    if any(state.human_review_config.values()):
        log.info('[人工入口] 另一终端运行 python tools/human_review.py --work-dir "%s" --message "你的意见"；纯提问加 --kind question', work_dir)
    atexit.register(state.flush)
    signal.signal(signal.SIGTERM, lambda s, f: (state.flush(), sys.exit(1)))
    signal.signal(signal.SIGINT, lambda s, f: (state.flush(), sys.exit(1)))

    # Re-probe on every startup, including resume; an old device_info.json
    # is historical evidence, not proof that today's device is the same chip.
    device_info_path = os.path.join(work_dir, "device_info.json")
    if imported:
        log.info("[芯片复核] 保留导入的评分硬件文件；检查当前硬件和运行环境是否匹配，不调用开发节点")
    elif os.path.exists(device_info_path):
        log.info("[芯片复核] 断点恢复将重新核对当前芯片，旧 device_info.json 不作为本次硬件来源")
    hw_config = config.get("hardware", {})
    if hw_config.get("chip_model"):
        device_info = {
            "chip_model": hw_config.get("chip_model", "unknown"),
            "soc_version": hw_config.get("soc_version", "unknown"),
            "npu_arch": hw_config.get("npu_arch", "unknown"),
            "programming_model": PROGRAMMING_MODEL,
            "ai_core_num": hw_config.get("ai_core_num", 0),
            "ub_size_kb": hw_config.get("ub_size_kb", 0),
            "l1_size_kb": hw_config.get("l1_size_kb", 0),
            "l0a_size_kb": hw_config.get("l0a_size_kb", 0),
            "l0b_size_kb": hw_config.get("l0b_size_kb", 0),
            "l0c_size_kb": hw_config.get("l0c_size_kb", 0),
            "l2_size_kb": hw_config.get("l2_size_kb", 0),
            "cube_core_num": hw_config.get("cube_core_num", 0),
            "vector_core_num": hw_config.get("vector_core_num", 0),
            "device_id": hw_config.get("device_id", 0),
            "detect_method": "config_manual",
        }
        log.info(f"芯片信息来自 config.yaml 手动配置: {device_info['chip_model']}")
    else:
        device_info = detect_npu_device(device_id=hw_config.get("device_id", 0), config_path=args.config)
        device_info["device_id"] = hw_config.get("device_id", 0)

    # ── 检测结果校验：关键参数必须全部获取到，否则退出 ──
    missing = []
    if device_info.get("soc_version", "unknown") == "unknown":
        missing.append("soc_version（无法识别芯片型号，torch_npu/环境变量均失败）")
    if device_info.get("detect_method") == "none" or device_info.get("ub_size_kb", 0) == 0:
        missing.append("UB_SIZE（tbe 查询失败，CANN Toolkit 可能未正确安装）")
    if device_info.get("l1_size_kb", 0) == 0:
        missing.append("L1_SIZE")
    if device_info.get("ai_core_num", 0) == 0:
        missing.append("CORE_NUM")
    if device_info.get("l0a_size_kb", 0) == 0:
        missing.append("L0A_SIZE")

    if missing:
        log.error("=" * 60)
        log.error("❌ NPU 芯片参数检测失败，workflow 无法启动")
        log.error(f"缺失参数: {', '.join(missing)}")
        log.error(f"已获取到的信息: {_json_module.dumps(device_info, ensure_ascii=False, indent=2)}")
        log.error("")
        log.error("排查步骤：")
        log.error("  1. 确认 NPU 设备可用: python3 -c \"import torch,torch_npu; print(torch_npu.npu.get_device_properties(0).name)\"")
        log.error("  2. 确认 CANN Toolkit 安装完整: python3 -c \"from tbe.common.platform import get_soc_spec; print('OK')\"")
        log.error("  3. 或在 config.yaml 的 hardware 段手动填写全部芯片参数；仍需安装兼容的 Triton Ascend 后端")
        log.error("=" * 60)
        sys.exit(1)

    log.info(f"芯片检测通过: {device_info['chip_model']} (soc={device_info['soc_version']}, "
             f"cores={device_info['ai_core_num']}, UB={device_info['ub_size_kb']}KB, "
             f"L1={device_info['l1_size_kb']}KB, L0C={device_info.get('l0c_size_kb', '?')}KB, "
             f"method={device_info['detect_method']})")
    # Validate the installed backend even for a resumed/manual-hardware run.
    try:
        runtime = detect_triton_runtime(device_id=config.get("hardware", {}).get("device_id", 0),
                                        config_path=args.config)
    except (RuntimeError, ValueError, OSError) as exc:
        log.error("===== Triton Ascend 运行环境检测失败 ===== %s", exc)
        state_log.error("===== Triton Ascend 运行环境检测失败 ===== %s", exc)
        raise SystemExit(1) from exc
    device_info.update(runtime)
    device_info["device_id"] = config.get("hardware", {}).get("device_id", 0)
    device_info.pop("arch_key", None)
    device_info.pop("memory_model", None)
    if imported:
        # Preserve the exact hardware input used by the imported Jev response.
        # A mismatch must fail locally, never silently rescore an emergency run.
        validate_init_impl_inputs(work_dir, device_info, task_dir)
    else:
        atomic_write_json(device_info_path, device_info)
    log.info("[目标框架] framework=%s；backend=%s；driver=%s；versions=%s；device_id=%s；身份文件=%s",
             FRAMEWORK, BACKEND, runtime["driver_backend"], runtime["runtime_versions"],
             device_info["device_id"], Path(work_dir) / "workflow_target.json")
    log.info("[CANN 工具链] 已核对版本文件及指纹：%s", runtime.get("toolchain", {}))

    DEVICE_INFO_PROMPT = (
        file_hint(work_dir, Path(work_dir) / "device_info.json", "本次硬件参数来源",
                  "先核对芯片、编程模型、核数及 UB/L1 容量，再判断方案和 tiling 是否可行")
        + format_device_info_for_prompt(device_info)
    )
    evaluation_context = comparison_context(task_dir, device_info, config)
    # Validate x/y before running any development or evaluation.
    semantic_config = config.get("workflow", {}).get("semantic_exit", {})
    for option in ("passed_window", "underperforming_window"):
        value = semantic_config.get(option, 3)
        if type(value) is not int or value < 1:
            raise ValueError(f"workflow.semantic_exit.{option} must be a positive integer")

    roles_dir = os.path.join(os.path.dirname(__file__), "roles")

    if imported:
        if state.current_stage == "N1_phase1" or stage_number(state.current_stage) in (1, 1.5, 2):
            evidence = restore_imported_evidence(work_dir, imported)
            log.info("[应急导入] 开发证据：eligible=%s；原因=%s；新绑定=%s",
                     evidence.get("eligible"), evidence.get("reason"),
                     Path(work_dir) / "selection/current_implementation.json")
            if not evidence.get("eligible"):
                log.warning("[应急导入] 代码仍直接进入正式评测；开发证据未匹配时，成绩会保留，"
                            "但不进入最佳记录和停滞窗口，后续 Stage3 需补齐证据")
            transition(state, "ready_for_iter", state_log,
                       reason="应急导入完成，跳过 Stage1/1.5/2；从 Stage4 编译指定实现")
        log.info("===== 应急导入：跳过 Stage1 需求分析、Stage1.5 Jev 评分、Stage2 首版开发 =====")
        log.info("[应急导入] 沿用已导入的融合概率与 Top N；本次状态=%s；未继承旧评测、history、最佳记录或人工意见",
                 state.current_stage)

    # ═══════════════════════════════════════════════════════════════
    # 阶段1：需求分析（cannbot）
    # ═══════════════════════════════════════════════════════════════
    if state.current_stage == "N1_phase1" or stage_number(state.current_stage) == 1:
        stage = "阶段1-需求分析"
        jev_config = config.get("jev", {})
        requirements_budget = fusion_requirements_byte_budget(
            device_info,
            max_state_question_bytes=jev_config.get("max_state_question_bytes", 28000),
            max_request_bytes=jev_config.get("max_request_bytes", 56000),
            model=jev_config.get("model", "jev-1.13.0"),
        )
        role = os.path.join(roles_dir, "n1_stage1_requirements_analysis.md")
        inputs = [f"{work_dir}/task/"]
        outputs = [f"{work_dir}/ANALYSIS.md", f"{work_dir}/fusion_requirements.en.json"]
        log_io(log, stage, inputs, outputs)
        append_work_record(work_dir, stage)

        hint_block = ""
        if optimize_hint:
            hint_block = f"\n💡 融合方向提示（用户指定）：\n{optimize_hint}\n请在分析中重点关注该融合方向的可行性，给出具体的片上数据流设计建议。\n"
        prompt = (f"工作目录：{work_dir}\n算子：{op_name}\n"
                  + file_hint(work_dir, Path(work_dir) / "task", "只读算子需求与评测基准",
                              "依次读 desc.md 的定义、proto.yaml 的接口、cases.yaml 的覆盖范围、golden.py 的参考语义") +
                  f"{DEVICE_INFO_PROMPT}\n{hint_block}请分析task需求，输出写入 {work_dir}/ANALYSIS.md\n"
                  f"同时按 role 输出紧凑英文融合需求 {work_dir}/fusion_requirements.en.json，"
                  "供 stage1.5 Jev 结合真实硬件和融合方法评分。"
                  f"程序根据本次方法原文、选项和硬件估算的需求上限为 {requirements_budget} UTF-8 JSON字节，"
                  "请用这个上限（包含 JSON 结构）组织摘要，不能删除影响融合选择的约束。"
                  "stage1.5 会先统一翻译为英文，再检查实际请求大小。") + CANNBOT_CONSTRAINT
        ok = run_agent("cannbot", role, work_dir, prompt, node_log=node_logs["N1"])
        if not ok:
            log.error(f"[{stage}] cannbot 失败")
            raise RuntimeError("Stage1 failed; fusion selection requires completed requirements.")
        transition(state, "stage1.5", state_log, reason="需求分析完成，开始 Jev 融合方案评分")

    # stage1.5 可独立恢复；只有有效 Jev 结果落盘后才允许进入首版实现。
    if stage_number(state.current_stage) in (1.5, 2):
        stage = "阶段1.5-Jev融合方案选择"
        transition(state, "stage1.5", state_log, reason="读取需求、硬件及融合方法，生成候选库")
        log_io(log, stage,
               [f"{work_dir}/ANALYSIS.md", f"{work_dir}/fusion_requirements.en.json",
                device_info_path, str(Path(__file__).parent / "knowledge/fusion_method.md"),
                str(Path(__file__).parent / "knowledge/fusion_options.json")],
               [str(fusion_library_path(work_dir)), f"{work_dir}/fusion/ranking.json"])
        append_work_record(work_dir, stage)
        run_fusion_selection(work_dir, config_path=args.config,
                             top_n=config.get("fusion_selection", {}).get("top_n", 3))
        transition(state, "stage2", state_log, reason="Jev 评分完成，首版使用最高概率方案")
    elif not imported and stage_number(state.current_stage) != 10:
        # 新流程的断点恢复复核输入指纹，并允许只调整 n 而不重复调用 Jev。
        if (Path(work_dir, "fusion_requirements.en.json").is_file()
                or Path(work_dir, "fusion").exists()):
            run_fusion_selection(work_dir, config_path=args.config,
                                 top_n=config.get("fusion_selection", {}).get("top_n", 3))
        else:
            log.warning("旧工作目录尚无 stage1.5 需求和候选库，沿用原迭代；新任务将自动生成融合库。")

    # ═══════════════════════════════════════════════════════════════
    # 阶段2：编写第一版算子（cannbot）
    # ═══════════════════════════════════════════════════════════════
    if stage_number(state.current_stage) == 2:
        stage = "阶段2-编写第一版"
        role = os.path.join(roles_dir, "n1_stage2_first_impl.md")
        design_dir = os.path.join(work_dir, "develop", "iter0")
        Path(design_dir).mkdir(parents=True, exist_ok=True)
        inputs = [f"{work_dir}/task/", f"{work_dir}/ANALYSIS.md", str(fusion_library_path(work_dir))]
        outputs = [f"{work_dir}/impl/", f"{design_dir}/design_rationale.md"]
        log_io(log, stage, inputs, outputs)
        append_work_record(work_dir, stage)

        hint_block = ""
        if optimize_hint:
            hint_block = f"\n💡 融合方向提示（用户指定）：\n{optimize_hint}\n请在实现中重点按此融合方向设计数据流。\n"
        prompt = (
            f"工作目录：{work_dir}\n算子：{op_name}\n"
            + file_hint(work_dir, Path(work_dir) / "task", "只读算子需求",
                        "按 desc.md、proto.yaml、cases.yaml、golden.py 核对定义、接口、case 和参考结果")
            + file_hint(work_dir, Path(work_dir) / "ANALYSIS.md", "Stage1 需求分析",
                        "重点看注册名、精度与 shape 约束、目标芯片和实现难点")
            + file_hint(work_dir, Path(work_dir) / "example", "Triton Ascend 示例工程",
                        "参考包结构、注册和 API 用法，不能照搬示例算子的语义") +
            f"{DEVICE_INFO_PROMPT}\n"
            f"{hint_block}"
            f"{format_proven_patterns_for_prompt(work_dir)}"
            f"{format_regression_patterns_for_prompt(work_dir)}"
            f"{format_pitfalls_for_prompt(work_dir)}\n"
            f"请在 {work_dir}/impl/ 交付完整可安装工程：核心实现放在 impl/cann_bench/，"
            f"通过 impl/cann_bench/__init__.py 导出任务函数，并提供 impl/setup.py 和 impl/build.sh。\n"
            f"设计思路文档输出到 {design_dir}/design_rationale.md\n"
            f"自测报告输出到 {design_dir}/self_test_report.md"
        ) + CANNBOT_CONSTRAINT
        prompt += format_fusion_library_for_prompt(work_dir, "stage2")
        prompt += development_prompt(work_dir, design_dir, 2)
        previous_revisions = begin_development(work_dir, design_dir)
        ok = run_agent("cannbot", role, work_dir, prompt, node_log=node_logs["N1"])
        evidence = finalize_development(work_dir, design_dir, 2, agent_ok=ok,
                                        previous_revisions=previous_revisions)
        if not evidence.get("eligible"):
            log.warning(f"[{stage}] 方案/自测材料尚未满足最佳记录条件：{evidence.get('reason', '')}")
        if not ok:
            log.error(f"[{stage}] cannbot 失败")
        transition(state, "ready_for_iter", state_log, reason="首版编写完成，进入迭代循环")

    # ═══════════════════════════════════════════════════════════════
    # 主迭代循环：阶段4→5→6→(7→8→3)→回4
    # 支持断点续跑：根据 current_stage 决定从哪个阶段接着跑
    # ═══════════════════════════════════════════════════════════════
    def _get_resume_stage(current_stage: str) -> int:
        """从 current_stage 提取要恢复的阶段号。如 iter1_stage5 → 5"""
        if "stage" in current_stage:
            try:
                return int(current_stage.split("stage")[-1])
            except ValueError:
                pass
        return 0  # 非 stage 格式，走新一轮

    iter_dirs = None  # 初始化，防止 while 循环未执行时 stage10 引用报错
    while (stage_number(state.current_stage) != 10
           and (state.iteration < state.max_iterations
                or stage_number(state.current_stage) in (4, 5, 6, 7, 8, 9)
                or (stage_number(state.current_stage) == 3
                    and state.stage9_context.get("phase") == "developing"))):
        # 判断是否需要从中间恢复（断点续跑）
        resume_stage = _get_resume_stage(state.current_stage)
        if (resume_stage == 3 and state.stage9_context.get("iteration") == state.iteration
                and state.stage9_context.get("phase") == "developing"):
            restore_dirs = {key: str(Path(work_dir) / key / f"iter{state.iteration}")
                            for key in ("build", "eval", "profile", "search", "develop")}
            self_goto_stage3(log, roles_dir, work_dir, op_name, state,
                             reason=state.stage9_context["development_reason"],
                             state_log=state_log, node_logs=node_logs, iter_dirs=restore_dirs,
                             device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
            continue
        if resume_stage > 3 and resume_stage <= 9:
            # 断点续跑，不增加 iteration
            log.info(f"{'─'*40} ITER {state.iteration}/{state.max_iterations} (从stage{resume_stage}恢复) {'─'*40}")
        else:
            # 正常新一轮
            state.iteration += 1
            state.flush()
            log.info(f"{'─'*40} ITER {state.iteration}/{state.max_iterations} {'─'*40}")
            resume_stage = 4  # 新一轮从阶段4开始

        # 当前轮次的 iter 目录
        iter_name = f"iter{state.iteration}"
        iter_dirs = {
            "build": os.path.join(work_dir, "build", iter_name),
            "eval": os.path.join(work_dir, "eval", iter_name),
            "profile": os.path.join(work_dir, "profile", iter_name),
            "search": os.path.join(work_dir, "search", iter_name),
            "develop": os.path.join(work_dir, "develop", iter_name),
        }
        for d in iter_dirs.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        # Resumed Stage7/8/9 must use this iteration's persisted measurements.
        perf_diff = (_compute_perf_diff(work_dir, state.iteration, expected_context=evaluation_context,
                                       log=log, state_log=state_log) if resume_stage > 6 else {})
        perf_diff_text = (perf_diff.get("comparison_text", "") + perf_diff.get("diff_text", "")
                          + perf_diff.get("regression_text", ""))
        selection_status = load_selection_status(work_dir, comparison_context=evaluation_context,
                                                 config=config)

        # Restore the original failure scene instead of guessing from a stale perf file.
        saved_review = state.stage9_context
        if (resume_stage == 9 and saved_review.get("iteration") == state.iteration
                and saved_review.get("fail_reason")):
            original_reason = saved_review["fail_reason"]
            if original_reason.startswith(("build_fail", "precision_fail", "score_zero")):
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs,
                              iter_dirs, fail_reason=original_reason, device_info_prompt=DEVICE_INFO_PROMPT,
                              selection_context=evaluation_context, perf_diff=saved_review.get("perf_diff", {}))
                self_goto_stage3(log, roles_dir, work_dir, op_name, state,
                                 reason=original_reason.splitlines()[0].split(":")[0],
                                 state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs,
                                 device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

        # ── 阶段4：编译部署（kerminal）─────────────────────
        if resume_stage <= 4:
            stage = f"ITER{state.iteration}-阶段4-编译"
            role = os.path.join(roles_dir, "n2_stage4_build.md")
            # 找最近的自测报告（stage2 在 iter0，stage3 在当前 iter）
            self_test_path = None
            for i in range(state.iteration, -1, -1):
                candidate = os.path.join(work_dir, "develop", f"iter{i}", "self_test_report.md")
                if os.path.exists(candidate):
                    self_test_path = candidate
                    break
            self_test_hint = (file_hint(work_dir, self_test_path, "最近一版开发自测报告",
                                       "看安装、import、给定 case 和连续调用结果；属于开发轮次，不等同于正式精度评测")
                              if self_test_path else "")
            if imported:
                self_test_hint += file_hint(
                    work_dir, Path(work_dir) / "init_impl_manifest.json", "应急导入来源及复制清单",
                    "核对指定代码与对应开发轮次；导入自测是原开发证据，不代表本次编译或精度已经通过")
                self_test_hint += "\n直接编译当前指定实现；导入报告的存在不能代替本次真实构建和正式评测。\n"
            inputs = [f"{work_dir}/impl/"]
            if self_test_path:
                inputs.append(self_test_path)
            outputs = [f"{iter_dirs['build']}/build.log"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage4", state_log, reason="开始编译部署")

            prompt = (f"工作目录：{work_dir}\n算子：{op_name}\n"
                      + file_hint(work_dir, Path(work_dir) / "impl", "待编译部署的当前工程",
                                  "从项目配置和 cann_bench 包检查依赖、注册入口及真实构建错误")
                      + f"{DEVICE_INFO_PROMPT}\n请编译部署，结果写入 {iter_dirs['build']}/build.log\n{self_test_hint}")
            ok = run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N2"])

            build_log_path = f"{iter_dirs['build']}/build.log"
            build_success = check_build_success(build_log_path)
            if not ok and build_success:
                log.warning(f"[{stage}] kerminal 进程异常退出但 build.log 显示 SUCCESS，信任 build.log")
            if not build_success:
                log.warning(f"[{stage}] 编译失败 → tech_lead 总结 → 阶段3")
                state_log.info(f"[判断] 编译失败: build.log 无 STATUS: SUCCESS")
                append_work_record(work_dir, f"{stage} FAIL → 阶段3")
                append_ledger_only(work_dir, state.iteration, "build_fail")
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="build_fail", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="build_fail", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

        # ── 阶段5：精度评测（kerminal）─────────────────────
        if resume_stage <= 5:
            stage = f"ITER{state.iteration}-阶段5-精度评测"
            role = os.path.join(roles_dir, "n2_stage5_precision_eval.md")
            inputs = [f"task_dir={task_dir}"]
            outputs = [f"{iter_dirs['eval']}/precision_result.json"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage5", state_log, reason="编译成功，开始精度评测")

            precision_impl_hash = _implementation_hash_or_empty(work_dir)
            precision_context = comparison_context(task_dir, device_info, config)
            precision_path = Path(iter_dirs["eval"]) / "precision_result.json"
            previous_precision_revision = _file_revision(precision_path)
            binding_path = Path(iter_dirs["eval"]) / "precision_binding.json"
            atomic_write_json(str(binding_path), {"eligible": False, "reason": "Precision evaluation pending"})
            prompt = (f"工作目录：{work_dir}\n算子：{op_name}\ntask_dir：{task_dir}\n"
                      + file_hint(work_dir, Path(work_dir) / "task", "精度评测的只读 task 入口（链接到上述 task_dir）",
                                  "使用 cases.yaml 的完整用例及 golden.py 参考实现，遵守 proto.yaml 接口")
                      + f"{DEVICE_INFO_PROMPT}\n结果写入 {iter_dirs['eval']}/precision_result.json")
            precision_agent_ok = run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N2"])

            precision = parse_precision_result(f"{iter_dirs['eval']}/precision_result.json")
            if precision_path.is_file():
                fresh_precision = _file_revision(precision_path) != previous_precision_revision
                atomic_write_json(str(binding_path), {
                    "eligible": precision_agent_ok is True and fresh_precision,
                    "impl_sha256": precision_impl_hash,
                    "precision_sha256": hashlib.sha256(precision_path.read_bytes()).hexdigest(),
                    "comparison_context": precision_context,
                })
                if precision_agent_ok is not True or not fresh_precision:
                    log.warning(f"[{stage}] 本次精度执行未成功产出新报告，不绑定旧报告用于最佳实现选择")
            # 创建精度报告软链接（链到具体文件，不是整个目录）
            source_report = precision.get("source_report", "")
            if source_report and os.path.exists(source_report):
                link_dir = os.path.join(iter_dirs['eval'], "precision_reports")
                Path(link_dir).mkdir(parents=True, exist_ok=True)
                link_path = os.path.join(link_dir, os.path.basename(source_report))
                if not os.path.exists(link_path):
                    os.symlink(source_report, link_path)

            if not precision.get("precision_overall", False):
                prec_passed = precision.get("passed_cases", "?")
                prec_total = precision.get("total_cases", "?")
                log.warning(f"[{stage}] 精度不通过({prec_passed}/{prec_total}) → tech_lead 总结 → 阶段3")
                state_log.info(f"[判断] 精度失败: passed={prec_passed}/{prec_total}")
                append_work_record(work_dir, f"{stage} FAIL → 阶段3")
                append_ledger_only(work_dir, state.iteration, "precision_fail")
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="precision_fail", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="precision_fail", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

        # ── 阶段6：性能评测（代码直接执行 cann-bench）─────────
        if resume_stage <= 6:
            stage = f"ITER{state.iteration}-阶段6-性能评测"
            inputs = [f"task_dir={task_dir}"]
            outputs = [f"{iter_dirs['eval']}/perf_result.json"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage6", state_log, reason="精度通过，开始性能评测")
            measurement_impl_hash = _implementation_hash_or_empty(work_dir)
            evaluation_context = comparison_context(task_dir, device_info, config)

            # 直接执行 cann-bench（不通过 agent）
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            cannbench_src = os.path.abspath(os.path.join(_script_dir, config.get("paths", {}).get("cannbench_repo", "../cann-bench"), "src"))
            if not os.path.isdir(cannbench_src):
                cannbench_src = os.path.abspath(os.path.join(_script_dir, config.get("paths", {}).get("cannbench_repo", "../cann-bench")))

            log.info(f"[{stage}] 执行 cann-bench 性能评测（代码直接执行）")
            node_logs["N2"].info(f"性能评测开始，task={task_dir}")

            # 最多重试 2 次（偶发段错误/NPU 设备占用）
            max_retries = 2
            for attempt in range(1, max_retries + 1):
                # An isolated output root prevents retries/resume from using old profiler files.
                reports_dir = os.path.join(iter_dirs["eval"], "perf_reports",
                                           f"attempt_{attempt}_{uuid.uuid4().hex}")
                Path(reports_dir).mkdir(parents=True, exist_ok=True)
                log.info(f"[{stage}] 本次评测输出目录：{reports_dir}")
                success, eval_output, report_path = run_perf_eval(
                    task_dir, cannbench_src, device_id=config.get("hardware", {}).get("device_id", 0),
                    log_callback=lambda msg: node_logs["N2"].info(msg), config_path=args.config,
                    reports_dir=reports_dir)
                node_logs["N2"].info(f"性能评测（第{attempt}/{max_retries}次），success={success}, report={report_path}")
                log.info(f"[{stage}] 第{attempt}次评测：success={success}, report={'有' if report_path else '无'}")
                if success or report_path:
                    break
                if attempt < max_retries:
                    log.warning(f"[{stage}] 性能评测崩溃（第{attempt}次），等待10秒后重试...")
                    node_logs["N2"].warning(f"评测崩溃（第{attempt}次），10秒后重试")
                    import time as _time; _time.sleep(10)
                else:
                    log.error(f"[{stage}] 性能评测连续{max_retries}次崩溃，workflow 停止")
                    node_logs["N2"].error(f"评测连续{max_retries}次崩溃，workflow 停止")
                    append_work_record(work_dir, f"{stage} cann-bench 连续{max_retries}次崩溃，workflow 停止在 stage6")
                    state.flush()
                    raise RuntimeError(f"cann-bench 性能评测连续{max_retries}次崩溃，workflow 停止在 {state.current_stage}。排查 NPU 设备状态后重新运行即可从 stage6 断点续跑。")

            # Save the complete profiler tree first; every JSON link points at this work copy.
            source_csv_dir = archive_profiler_data(
                report_path, iter_dirs["eval"], log_callback=log.info)

            # 解析报告生成 perf_result.json
            if report_path:
                perf = parse_cannbench_report_to_perf_result(report_path, source_csv_dir)
                missing_csv = [case.get("case_id", "?") for case in perf.get("cases", [])
                               if not case.get("kernel_csv")]
                log.info(f"[{stage}] profiler 关联：source_csv_dir={source_csv_dir or '无'}；"
                         f"kernel_csv 已关联 {len(perf.get('cases', [])) - len(missing_csv)}/"
                         f"{len(perf.get('cases', []))} 个 case")
                if missing_csv:
                    log.warning(f"[{stage}] kernel_csv 未关联 case={missing_csv}；"
                                "缺少或存在多份独立 CSV；批量采集不强行逐 case 关联，"
                                "请查本轮 prof_data 原始文件")
            elif not success:
                perf = {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                        "score_error_code": "eval_crash",
                        "score_error": f"cann-bench 评测进程崩溃(returncode={eval_output[-200:] if eval_output else '无输出'})",
                        "_error": "cann-bench 评测崩溃且无新报告产出"}
                log.error(f"[{stage}] cann-bench 评测崩溃且无新报告（可能使用了旧报告，已过滤）")
            else:
                perf = {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                        "_error": "cann-bench 未产出报告"}
                log.error(f"[{stage}] cann-bench 未产出报告")

            # Bind the context at measurement time. Never retrofit this stamp
            # onto old reports using the environment of a later resumed run.
            perf["comparison_context"] = copy.deepcopy(evaluation_context)
            perf["comparison_context_stable"] = (
                evaluation_context == comparison_context(task_dir, device_info, config))
            # 写 perf_result.json
            atomic_write_json(f"{iter_dirs['eval']}/perf_result.json", perf)
            log.info(f"[{stage}] perf_result.json 已写入: overall={perf.get('overall_score')}, avg_speedup={perf.get('avg_speedup')}")

            # 创建性能报告软链接（链到具体文件，不是整个目录）
            source_json = perf.get("source_json", "")
            if source_json and os.path.exists(source_json):
                link_dir = os.path.join(iter_dirs['eval'], "perf_reports")
                Path(link_dir).mkdir(parents=True, exist_ok=True)
                link_path = os.path.join(link_dir, os.path.basename(source_json))
                if os.path.islink(link_path) and os.path.realpath(link_path) != os.path.realpath(source_json):
                    os.unlink(link_path)
                if not os.path.exists(link_path):
                    os.symlink(source_json, link_path)
                # 同名的 md 和 html 也链过来
                for ext in [".md", ".html"]:
                    src = source_json.replace(".json", ext)
                    if os.path.exists(src):
                        lp = os.path.join(link_dir, os.path.basename(src))
                        if os.path.islink(lp) and os.path.realpath(lp) != os.path.realpath(src):
                            os.unlink(lp)
                        if not os.path.exists(lp):
                            os.symlink(src, lp)
            state.update_last_eval({
                "precision_overall": precision.get("precision_overall", False) if resume_stage <= 5 else True,
                "perf_pass": perf.get("perf_pass", False),
                "perf_speedup": perf.get("avg_speedup", 0.0),
            })
            # 自动写入 rounds/ledger 硬数据到 history.json
            perf_diff = _compute_perf_diff(work_dir, state.iteration, expected_context=evaluation_context,
                                           log=log, state_log=state_log)
            prev_speedup = _history_previous_average(work_dir, state, log, comparison=perf_diff)
            append_round(work_dir, state.iteration, perf, prev_avg_speedup=prev_speedup,
                         comparison=perf_diff)
            import json as _json
            _h = load_history(work_dir)
            history_log.info(f"ITER{state.iteration} [stage6 程序写入 rounds/ledger]\n{_json.dumps(_h, ensure_ascii=False, indent=2)}")
            log.info(f"[{stage}] history.json rounds/ledger 已更新 (iter{state.iteration})")

            # ═══════════════════════════════════════════════════
            # stage6 后：程序算性能 diff（一次算好，后续 stage 共享）
            # ═══════════════════════════════════════════════════
            perf_diff_text = (perf_diff.get("comparison_text", "") + perf_diff.get("diff_text", "")
                              + perf_diff.get("regression_text", ""))
            if perf_diff.get("has_improvement"):
                state_log.info(f"[判断] 性能提升: avg_speedup {perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']} (+{perf_diff['delta_pct']}%), iter{perf_diff['prev_iter']}→iter{state.iteration}")
            elif perf_diff.get("has_regression"):
                state_log.info(f"[判断] 性能退步: avg_speedup {perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']} ({perf_diff['delta_pct']}%), iter{perf_diff['prev_iter']}→iter{state.iteration}")
            else:
                delta = perf_diff.get("delta_pct", 0)
                if delta != 0:
                    state_log.info(f"[判断] 性能变化不显著: delta={delta}% (±5%以内)")
                elif perf_diff.get("comparable"):
                    state_log.info("[判断] 同口径性能持平：delta=0%，不生成涨跌经验")
                else:
                    state_log.info("[判断] 本轮不比较涨跌：%s", perf_diff.get("comparison_reason", "缺少可比数据"))

            # ═══════════════════════════════════════════════════
            # stage6 后路由判断（顺序很重要，不能调换）
            # ① 先判反作弊 → ② 再判性能达标 → ③ 正常性能优化
            # ═══════════════════════════════════════════════════

            # ── ① 反作弊/零分检测（最高优先级）──
            score_error = perf.get("score_error_code") or perf.get("score_error", "")
            avg_speedup = perf.get("avg_speedup", 0.0)
            valid_average = (isinstance(avg_speedup, (int, float)) and not isinstance(avg_speedup, bool)
                             and math.isfinite(avg_speedup) and avg_speedup > 0)
            if score_error or not valid_average:
                reason_detail = score_error or ("avg_speedup=0（所有case性能数据为零）" if avg_speedup == 0.0
                                                else "avg_speedup 无有效正数测量")
                state_log.info(f"[判断] 性能评测异常或零分: {reason_detail}, avg_speedup={avg_speedup}")
                csv_analysis = analyze_kernel_csv_for_anticheat(perf)
                log.warning(f"[{stage}] 反作弊触发或零分: {reason_detail} → tech_lead 总结 → 阶段3")
                log.info(f"[{stage}] kernel_csv 分析:\n{csv_analysis}")
                append_work_record(work_dir, f"{stage} 反作弊/零分({reason_detail}) → 阶段3")
                anticheat_context = (
                    f"score_error_code: {score_error}\n"
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "反作弊/零分性能汇总",
                                "核对 score_error_code、逐 case 状态及 kernel_csv 路径")
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "零分对应的原始性能报告",
                                "对照错误码和原始事件证据，区分执行失败与普通性能问题")
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'prof_data', "零分对应的 profiler 数据",
                                "核对自定义 NPU kernel 是否实际执行及有无禁止的 host 侧计算") +
                    f"\n=== kernel_details.csv 自动分析 ===\n{csv_analysis}\n"
                )
                anticheat_directive = (
                    f"\n本轮评分异常，先核实错误类型和原始证据。缺报告或零分本身不能证明违规。\n"
                    f"score_error_code: {reason_detail}\n"
                    f"\n=== kernel_details.csv 自动分析结果（程序已替你解析）===\n{csv_analysis}\n"
                    f"\n请按以下步骤执行：\n"
                    f"1. 对照实际错误码、报告是否生成、执行日志和 kernel CSV，区分报告/采集故障与代码问题。\n"
                    f"2. 若证据涉及反作弊，读取 knowledge/anti_cheat_reference.md 对照实际错误处理表；未证实时不得编造违规结论。\n"
                    f"3. 只对已经定位的根因给出修复方向，并明确尚待验证的假设。\n"
                    f"4. 保持反作弊规则，禁止缓存结果、sleep 等绕过评测的方案；不能为缺报告盲目重写核心算法。\n"
                )
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
                              fail_reason=f"score_zero: {reason_detail}\n{anticheat_directive}", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context, perf_diff=perf_diff)
                self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="score_zero",
                                 state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs,
                                 extra_context=anticheat_context, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

            evaluation_context = comparison_context(task_dir, device_info, config)
            previous_best_id = (selection_status.get("best") or {}).get("record_id")
            selection_status = _record_performance_selection(
                work_dir, state.iteration, perf, evaluation_context, config, measurement_impl_hash)
            if not selection_status.get("eligible"):
                log.warning(f"[{stage}] 不更新最佳实现或停滞窗口：{selection_status.get('reason', '')}")
            else:
                best = selection_status["best"]
                window_status = selection_status["window"]
                best_changed = best['record_id'] != previous_best_id
                log.info(f"[最佳实现] {'更新' if best_changed else '保留'} iter{best['iteration']}: "
                         f"avg_speedup={best['avg_speedup']}, avg_speed={best['avg_speed']}, "
                         f"HAP.performance_score={best['hap'].get('performance_score')}, "
                         f"all_cases_pass={best['all_cases_pass']}, "
                         f"min_case_speedup={min(c['speedup'] for c in best['case_results'])}, "
                         f"融合方法={best['fusion_scheme'].get('method_ids', [])}")
                log.info(f"[最佳实现] 代码快照={best['implementation_dir']}；"
                         f"性能报告={best['performance_report']}；性能汇总={best['performance_result']}；"
                         f"清单={best['manifest_path']}；"
                         f"方案依据={best['evidence_paths'].get('decision_rationale', '')}")
                if best_changed:
                    log.debug("[最佳实现] HAP 原始逐 case 指标（不重算）=%s",
                              _json_module.dumps(best['hap']['cases'], ensure_ascii=False))
                event = log_semantic_trigger(work_dir, state.iteration, selection_status, log, state_log)
                if not event:
                    state_log.info(f"[语义窗口] {window_status['status']}: "
                                   f"{window_status['completed_improvements']}/{window_status['required_improvements']}, "
                                   f"累计提升={window_status['cumulative_improvement']:.2%}, "
                                   f"退出={selection_status['should_exit']}, "
                                   f"审查融合方案={selection_status['review_fusion']}")

        # ── ② 达标后由程序窗口决定退出；Stage9 仍完成原有知识积累 ──
        # This also handles a checkpoint resumed in Stage9 after a passing run.
        if resume_stage > 6:
            perf = parse_perf_result(f"{iter_dirs['eval']}/perf_result.json")
        # A resumed Stage9 retains the routing decision made for that evaluation.
        restored_reason = (state.stage9_context.get("fail_reason", "")
                           if resume_stage == 9 and state.stage9_context.get("iteration") == state.iteration else "")
        passed_route = (restored_reason.startswith("perf_pass") if restored_reason
                        else perf.get("perf_pass", False))
        if passed_route:
            should_exit = (selection_status.get("should_exit", False)
                           and selection_status.get("latest_iteration") == state.iteration)
            directive = (
                "所有 case 的 speedup 已达标。保留 P0/P1/P2，结合实测瓶颈审查下一步；"
                "不得仅因多 kernel 或中间经过 HBM 强制更换融合方案。"
                "退出由程序的有效性能窗口判断，exit_decision 不控制路由。"
                + ("本轮已满足语义退出条件，请完成经验记录，随后总结最佳已评测实现。"
                   if should_exit else "本轮尚未满足语义退出条件，继续提出有证据的优化建议。")
            )
            run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
                          fail_reason=restored_reason or f"perf_pass_semantic_review\n{directive}",
                          device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context, perf_diff=perf_diff)
            if should_exit:
                # Recheck the archived evidence after Stage9 before reporting an exit.
                verified = load_selection_status(work_dir, comparison_context=evaluation_context, config=config)
                should_exit = bool(verified.get("should_exit") and verified.get("best"))
            if should_exit:
                apply_regression_action(work_dir, state.iteration,
                                        state.stage9_context.get("decision_path"), log, state_log,
                                        comparison_context=evaluation_context)
                state.stopped_by = "semantic_stagnation"
                append_work_record(work_dir, "达标实现的 x 次有效改进累计不足 5%，选最佳已评测快照退出")
                transition(state, "stage10", state_log, reason="程序语义退出，交付最佳达标实现")
                break
            self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="perf_pass_optimize",
                             state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs,
                             device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
            continue

        # ── ③ 性能不达标仍走 stage7/8/9/3；y 窗口仅触发方案审查 ──
        state_log.info("[判断] 性能不达标 → stage7→8→9→3；是否换融合方案由 stage9 结合证据决定")

        # ── 阶段7：Profiling 分析（kerminal）───────────────
        if resume_stage <= 7:
            stage = f"ITER{state.iteration}-阶段7-profiling"
            role = os.path.join(roles_dir, "n2_stage7_kerminal_profile.md")
            inputs = [f"{iter_dirs['eval']}/perf_result.json", f"{iter_dirs['eval']}/perf_reports/"]
            if Path(fusion_library_path(work_dir)).is_file():
                inputs.append(str(fusion_library_path(work_dir)))
            outputs = [f"{iter_dirs['profile']}/bottleneck_analysis.md"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage7", state_log, reason="性能不达标，开始profiling分析")

            prev_design_path = None
            for i in range(state.iteration - 1, -1, -1):
                candidate = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
                if os.path.exists(candidate):
                    prev_design_path = candidate
                    break
            if prev_design_path is None:
                # Imported Stage3 material may only have its fusion-decision
                # document; never inject a nonexistent Stage2 design filename.
                imported_rationale = Path(work_dir) / "develop/iter0/融合方案选择决策依据.md"
                if imported and imported_rationale.is_file():
                    prev_design_path = str(imported_rationale)
                else:
                    prev_design_path = os.path.join(work_dir, "develop", "iter0", "design_rationale.md")

            profiling_skill_path = os.path.join(os.path.dirname(roles_dir), "skills", "triton-profiling-analysis", "SKILL.md")
            profiling_skill_hint = (
                "\n" + file_hint(work_dir, profiling_skill_path, "Profiling 分析指南",
                                  "先读五文件分析流程和指标权重，再解释瓶颈证据",
                                  base_dir=Path(__file__).parent, base_label="项目根目录") +
                f"请加载上述 skill 文件，按其中定义的五文件分析流程（kernel_details→op_statistic→step_trace_time→api_statistic→trace_view）"
                f"和权重分配（50%/20%/15%/10%/5%）对每个 case 的 profiler 数据进行结构化分析。\n"
            ) if os.path.exists(profiling_skill_path) else ""

            prompt = (
                f"工作目录：{work_dir}\n算子：{op_name}\n"
                + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "本轮正式性能汇总",
                            "先看 avg_speedup、每个 case 的 speedup 和 worst_6_cases，再找性能瓶颈")
                + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "本轮原始性能报告",
                            "按 case 对照原始计时、kernel 和 profiler 证据，核验汇总结论")
                + file_hint(work_dir, prev_design_path, "最近可用的历史设计参考",
                            "看 tiling、数据流和改动目的；它不自动代表被评测版本，需与另附的代码绑定依据核对") +
                f"{DEVICE_INFO_PROMPT}\n"
                f"{perf_diff_text}\n"
                f"{format_proven_patterns_for_prompt(work_dir)}"
                f"{format_regression_patterns_for_prompt(work_dir)}"
                f"{format_pitfalls_for_prompt(work_dir)}\n"
                f"{profiling_skill_hint}"
                f"输出写入 {iter_dirs['profile']}/bottleneck_analysis.md"
            )
            prompt += format_fusion_library_for_prompt(work_dir, "stage7")
            prompt += format_evidence_for_prompt(work_dir)
            prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
            # A real Stage7 rerun must produce fresh output; an interrupted
            # attempt's partial report must not satisfy the output check.
            stale_report = validate_profile_report_directory(work_dir, state.iteration)
            if stale_report.exists():
                stale_report.unlink()
                log.info("[性能分析报告] iter=%s 重新执行 Stage7，已移除本轮旧报告，等待重新生成：%s",
                         state.iteration, stale_report)
            if not run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N2"]):
                raise RuntimeError(f"[{stage}] Stage7 未成功完成，不交付分析报告")

        # Each evaluated iteration exposes one canonical report. This also
        # stamps existing Stage7 reports when resuming at Stage8/9.
        try:
            report_path = stamp_stage7_report(work_dir, state.iteration)
        except (OSError, ValueError) as exc:
            log.error("[性能分析报告] iter=%s 来源=Stage7；检查失败，不进入后续阶段：%s",
                      state.iteration, exc)
            raise RuntimeError(f"Stage7 分析报告不可交付：{exc}") from exc
        _log_profile_report(log, state_log, state.iteration, 7, report_path)

        # ── 阶段8：搜索优化方案（hermes）───────────────────
        if resume_stage <= 8:
            stage = f"ITER{state.iteration}-阶段8-搜索"
            role = os.path.join(roles_dir, "n3_stage8_search.md")
            inputs = [f"{iter_dirs['profile']}/bottleneck_analysis.md", f"{work_dir}/impl/"]
            if Path(fusion_library_path(work_dir)).is_file():
                inputs.append(str(fusion_library_path(work_dir)))
            outputs = [f"{iter_dirs['search']}/SEARCH_REPORT.md", f"{iter_dirs['search']}/FIX_DIRECTIVE.md"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage8", state_log, reason="profiling完成，开始搜索优化方案")

            prompt = (
                f"工作目录：{work_dir}\n算子：{op_name}\n"
                + file_hint(work_dir, Path(iter_dirs['profile']) / 'bottleneck_analysis.md', "本轮瓶颈分析",
                            "按慢 case、根因和证据确定搜索问题，区分已证实瓶颈与待验证假设")
                + file_hint(work_dir, Path(work_dir) / 'impl', "当前算子实现工程",
                            "定位 cann_bench 中的算子入口、tiling 和数据流，核实搜索方案是否适用") +
                f"{DEVICE_INFO_PROMPT}\n"
                f"{perf_diff_text}\n"
                f"{format_proven_patterns_for_prompt(work_dir)}"
                f"{format_regression_patterns_for_prompt(work_dir)}\n"
                f"输出：{iter_dirs['search']}/SEARCH_REPORT.md 和 {iter_dirs['search']}/FIX_DIRECTIVE.md"
            )
            prompt += format_fusion_library_for_prompt(work_dir, "stage8")
            prompt += format_evidence_for_prompt(work_dir)
            prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
            run_agent("hermes", role, work_dir, prompt, node_log=node_logs["N3"])

        # ── 阶段9：Tech Lead 经验提炼（kerminal）──────────
        if resume_stage <= 9:
            run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="perf_optimize", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context, perf_diff=perf_diff)

        # ── 阶段3：修改优化（cannbot）──────────────────────
        self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="perf_optimize", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)

        # 一轮完成，iteration 在下一轮顶部递增

    # 迭代上限
    if state.iteration >= state.max_iterations and not state.stopped_by:
        state.stopped_by = "max_iterations"
        log.warning(f"达到最大迭代次数 {state.max_iterations}")

    # ═══════════════════════════════════════════════════════════════
    # 阶段10：最终报告（kerminal）— N5 节点
    # ═══════════════════════════════════════════════════════════════
    # 确保 iter_dirs 有值（兜底：用最后一轮的）
    if iter_dirs is None:
        iter_name = f"iter{state.iteration}"
        iter_dirs = {
            "build": os.path.join(work_dir, "build", iter_name),
            "eval": os.path.join(work_dir, "eval", iter_name),
            "profile": os.path.join(work_dir, "profile", iter_name),
            "search": os.path.join(work_dir, "search", iter_name),
            "develop": os.path.join(work_dir, "develop", iter_name),
        }
    _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                           node_logs, iter_dirs, DEVICE_INFO_PROMPT, evaluation_context)
    remaining_human = [item["id"] for item in human.all_messages()
                       if item.get("kind") == "direction" and item.get("status") == "processed"]
    if remaining_human:
        human.mark_unexecuted(remaining_human, state.stopped_by or "workflow_final_report")
    stage = "阶段10-最终报告"
    role = os.path.join(roles_dir, "n5_stage10_kerminal_report.md")
    evaluation_context = comparison_context(task_dir, device_info, config)
    final_selection = load_selection_status(work_dir, comparison_context=evaluation_context, config=config)
    selected = final_selection.get("best")
    final_precision_path = (selected["precision_result_path"] if selected
                            else f"{iter_dirs['eval']}/precision_result.json")
    final_perf_path = (selected["perf_result_path"] if selected
                      else f"{iter_dirs['eval']}/perf_result.json")
    inputs = [final_precision_path, final_perf_path, f"{work_dir}/WORK_RECORD.md",
              f"{work_dir}/knowledge/history.json"]
    if selected:
        inputs.extend([selected["manifest_path"], selected["implementation_dir"]])
    outputs = [f"{work_dir}/FINAL_REPORT.md"]
    log_io(log, stage, inputs, outputs)
    append_work_record(work_dir, stage)
    transition(state, "stage10", state_log, reason="生成最终报告")

    # ── 扫描未处理的 question.md（下级反馈但从未被 tech_lead 裁定）──
    import glob as _glob
    unhandled_questions = []
    for qpath in _glob.glob(os.path.join(work_dir, "develop", "iter*", "question.md")):
        with open(qpath, "r", encoding="utf-8") as f:
            qcontent = f.read()
        # 已裁定的会在错题本里，简单判断：question 提出后是否有对应 pitfall
        if "## tech_lead 裁定" not in qcontent:
            unhandled_questions.append(qpath)
    if unhandled_questions:
        log.warning(f"[{stage}] ⚠️ 有 {len(unhandled_questions)} 个未裁定的 question.md（迭代提前结束导致）:")
        for q in unhandled_questions:
            log.warning(f"    {q}")
        state_log.info(f"[提示] {len(unhandled_questions)} 个 question.md 未被 tech_lead 裁定（迭代提前退出）")

    question_hint = ""
    if unhandled_questions:
        question_hint = "".join(file_hint(work_dir, path, "未裁定的开发反馈",
                                        "看被质疑的建议和硬件/框架证据，在最终报告中列为待处理问题，不当作已裁定结论")
                                for path in unhandled_questions)

    prompt = (
        f"工作目录：{work_dir}\n算子：{op_name}\n"
        + file_hint(work_dir, final_precision_path, "最终报告采用的精度结果",
                    "核对通过数量、失败 case 和正确性；有最佳快照时采用快照内结果")
        + file_hint(work_dir, final_perf_path, "最终报告采用的性能结果",
                    "核对 avg_speedup、HAP 和逐 case 达标情况；不能把最后修改当作最佳已评测版本")
        + file_hint(work_dir, Path(work_dir) / 'WORK_RECORD.md', "工作流执行记录",
                    "按时间回顾阶段、失败原因与退出经过")
        + file_hint(work_dir, Path(work_dir) / '.state.json', "程序最终状态与退出原因",
                    "读取 iteration、stopped_by 和阶段；与语义窗口核对，不能自行推断退出原因")
        + file_hint(work_dir, Path(work_dir) / 'knowledge/history.json', "跨轮优化经验与账本",
                    "从 rounds、ledger、insights 总结实测变化、已验证经验和未解决问题") +
        f"{question_hint}"
        f"{DEVICE_INFO_PROMPT}\n"
        f"输出写入 {work_dir}/FINAL_REPORT.md"
    )
    if selected:
        prompt += (
            "\n" + file_hint(work_dir, selected['manifest_path'], "最终选定实现清单",
                              "核对代码指纹、指标、融合方案及报告关联，保证结论属于同一份快照")
            + file_hint(work_dir, selected['implementation_dir'], "最终选定实现的不可变代码快照",
                        "作为最终交付代码来源，不用当前 impl/ 的未评测修改替代") +
            "请用这份快照及其评测报告说明最终交付结果；当前 impl/ 可能有未评测修改。"
            "明确该实现是否全部 case 达标，不把最佳未达标候选写成达标。\n"
        )
    else:
        prompt += "\n尚无满足正确性、自测与统一口径要求的最佳记录。请明确报告这一事实，不将最后一次修改宣称为最佳已验证实现。\n"
    prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
    if human.all_messages():
        prompt += file_hint(work_dir, human.root / "state.json", "人工意见处理与执行状态",
                            "逐条区分已处理、已执行但待正式验证、受退出条件限制未执行；列出原因")
        prompt += file_hint(work_dir, human.root / "inbox", "人工原话与问题", "按消息编号与 Stage9 裁定关联，不能把已处理写成已执行")
    while True:
        if not run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N5"]):
            raise RuntimeError("Stage10 总结失败，保留状态供恢复；人工入口尚未关闭")
        if not human.pending_messages():
            human.close_workflow(state.stopped_by or "completed")
            if not human.pending_messages():
                break
        _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                               node_logs, iter_dirs, DEVICE_INFO_PROMPT, evaluation_context)
        human.mark_unexecuted([item["id"] for item in human.all_messages()
                               if item.get("kind") == "direction" and item.get("status") == "processed"],
                              state.stopped_by or "workflow_final_report")
        transition(state, "stage10", state_log, reason="补充人工意见处理结果后更新最终报告")
        prompt += "\n总结期间收到的人工意见已由 Stage9 复议，请重新读取 human_review/state.json 与 history，并明确未执行原因。\n"
        prompt += file_hint(work_dir, human.root / "state.json", "人工意见最终处理状态", "按消息编号列出未执行原因，不把已处理当作已执行")
        prompt += file_hint(work_dir, human.root / "inbox", "人的全部原话", "与状态及 history 中的 human_responses 逐条对应")

    log.info("=" * 60)
    log.info(f"完成！stopped_by={state.stopped_by}")
    log.info(f"报告：{work_dir}/FINAL_REPORT.md")
    log.info("=" * 60)
    state.flush()


def _compute_perf_diff(work_dir: str, iteration: int, *, expected_context=None, log=None, state_log=None) -> dict:
    """
    程序计算本轮和上一轮的性能 diff。返回 dict：
    {
        "has_improvement": True/False,
        "prev_iter": 2, "curr_iter": 3,
        "prev_avg_speedup": 1.5, "curr_avg_speedup": 3.0,
        "delta_pct": 100.0,
        "prev_design_path": "develop/iter2/design_rationale.md",
        "curr_design_path": "develop/iter3/design_rationale.md",
        "case_diffs": [  # 按提升幅度降序
            {"case_id": "14", "prev_speedup": 0.69, "curr_speedup": 3.29, "delta": "+377%"},
            ...
        ],
        "diff_text": "格式化好的文本，可直接注入 prompt"
    }
    """
    import json as _json
    result = {"has_improvement": False, "has_regression": False, "diff_text": "", "regression_text": "",
              "comparable": False, "comparison_status": "unavailable", "curr_iter": iteration,
              "comparison_reason": "本轮没有完整可核对的性能报告"}

    def finish():
        # A small durable record explains both normal comparisons and new
        # baselines. Re-reading it at Stage9 does not flood either log.
        directory = Path(work_dir) / "eval" / f"iter{iteration}"
        comparison_path = directory / "perf_comparison.json"
        result["comparison_path"] = str(comparison_path)
        summary = {key: value for key, value in result.items()
                   if key not in {"diff_text", "regression_text", "comparison_text"}}
        previous_summary = None
        if comparison_path.is_file():
            try:
                previous_summary = _json.loads(comparison_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if directory.is_dir() and summary != previous_summary:
            atomic_write_json(str(comparison_path), summary)
            message = ("===== 性能比较口径检查 ===== iter=%s previous_iter=%s；状态=%s；原因=%s；"
                       "当前报告=%s；上轮报告=%s；检查记录=%s")
            args = (iteration, result.get("prev_iter"), result["comparison_status"],
                    result["comparison_reason"], result.get("current_report"),
                    result.get("previous_report"), comparison_path)
            for logger in (log, state_log if state_log is not log else None):
                if logger is not None:
                    (logger.info if result["comparable"] else logger.warning)(message, *args)
        if not result["comparable"]:
            action = ("本轮作为新基线，后续仅与同口径完整结果比较。"
                      if result["comparison_status"] == "new_baseline" else "需要完整且口径明确的评测后才能比较。")
            result["comparison_text"] = (
                f"\n===== 本轮不比较性能涨跌 =====\n{result['comparison_reason']}；{action}"
                "不得用本次均值差推断优化收益或退步，不填写本次 proven_pattern/regression_pattern。\n"
                + file_hint(work_dir, comparison_path, "程序的性能比较口径检查",
                            "查看不比较的原因和两轮报告来源，不跨口径归因")
            )
        if result.get("current_report_source") == "verified_selection_archive":
            result["comparison_text"] = result.get("comparison_text", "") + file_hint(
                work_dir, result["current_report"], "工作副本缺失时使用的本轮已验证性能归档",
                "已核对快照完整性及轮次；读取实际成绩和存档口径，不使用旧缓存中的涨跌结论")
        return result

    def positive_number(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)

    # 读本轮 perf_result
    curr_path = os.path.join(work_dir, "eval", f"iter{iteration}", "perf_result.json")
    result["current_report"] = curr_path
    if not os.path.exists(curr_path):
        # A checkpoint can outlive a lost working report. Only a verified,
        # version-bound archive of this exact round may replace the missing
        # file; never reuse cached flags or overwrite an existing invalid report.
        archived = load_selection_status(work_dir, comparison_context=expected_context)
        current_record = archived.get("current") or {}
        archive_path = current_record.get("performance_result")
        if (archived.get("eligible") is not True or archived.get("latest_iteration") != iteration
                or current_record.get("iteration") != iteration
                or not archive_path or not Path(archive_path).is_file()):
            return finish()
        curr_path = archive_path
        result.update(current_report=str(archive_path), current_report_source="verified_selection_archive")
    try:
        with open(curr_path, "r", encoding="utf-8") as f:
            curr_perf = _json.load(f)
    except (OSError, ValueError):
        return finish()
    if not isinstance(curr_perf, dict):
        return finish()
    current_check = compare_performance(curr_perf, curr_perf, expected_context=expected_context)
    if not current_check["comparable"]:
        # Both operands above are the current report; name it accurately in
        # diagnostics even when validation rejects the first operand.
        result["comparison_reason"] = current_check["reason"].replace("上一轮报告", "当前轮报告", 1)
        result["comparison_mismatch_fields"] = [
            "current." + field[len("previous."):] if field.startswith("previous.") else field
            for field in current_check["mismatch_fields"]
        ]
        return finish()
    curr_avg = curr_perf.get("avg_speedup", 0)
    result["curr_avg_speedup"] = curr_avg

    # 找上一轮有效的 perf_result（跳过 build_fail/precision_fail 没跑性能的轮次）
    prev_avg = 0
    prev_iter = None
    prev_perf = {}
    for i in range(iteration - 1, 0, -1):
        prev_path = os.path.join(work_dir, "eval", f"iter{i}", "perf_result.json")
        if os.path.exists(prev_path):
            try:
                with open(prev_path, "r", encoding="utf-8") as f:
                    prev_perf = _json.load(f)
                prev_avg = prev_perf.get("avg_speedup", 0) if isinstance(prev_perf, dict) else 0
                if positive_number(prev_avg) and not any(prev_perf.get(key) for key in
                                                        ("_error", "score_error", "score_error_code")):
                    prev_iter = i
                    break
            except (OSError, ValueError):
                continue

    if prev_iter is None:
        result.update(comparison_status="new_baseline", comparison_reason="没有上一轮有效性能结果，不计算涨跌")
        return finish()

    result["prev_iter"] = prev_iter
    result["previous_report"] = str(Path(work_dir) / "eval" / f"iter{prev_iter}" / "perf_result.json")
    compared = compare_performance(prev_perf, curr_perf, expected_context=expected_context)
    result["comparison_reason"] = compared["reason"]
    result["comparison_mismatch_fields"] = compared["mismatch_fields"]
    if not compared["comparable"]:
        # Do not jump across a changed protocol to find an older matching run.
        result["comparison_status"] = "new_baseline"
        return finish()
    result.update(comparable=True, comparison_status="comparable")

    delta_pct = round((curr_avg - prev_avg) / prev_avg * 100, 1)
    result["prev_iter"] = prev_iter
    result["curr_iter"] = iteration
    result["prev_avg_speedup"] = prev_avg
    result["curr_avg_speedup"] = curr_avg
    result["delta_pct"] = delta_pct
    result["has_improvement"] = delta_pct >= 5
    result["has_regression"] = delta_pct <= -5

    # design_rationale 路径
    for i in range(iteration, -1, -1):
        p = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
        if os.path.exists(p):
            result["curr_design_path"] = p
            break
    for i in range(prev_iter, -1, -1):
        p = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
        if os.path.exists(p):
            result["prev_design_path"] = p
            break

    # 保留所有可对齐 case 的变化，提示词仅展示最相关的六个。
    case_diffs = []
    curr_cases = curr_perf.get("cases") or curr_perf.get("worst_6_cases") or []
    prev_cases_map = {}
    for c in prev_perf.get("cases") or prev_perf.get("worst_6_cases") or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("case_id", c.get("id", ""))
        if cid:
            prev_cases_map[str(cid)] = c.get("speedup", 0)

    for c in curr_cases:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("case_id", c.get("id", "")))
        curr_sp = c.get("speedup", 0)
        prev_sp = prev_cases_map.get(cid, 0)
        if positive_number(prev_sp) and positive_number(curr_sp):
            cdelta = round((curr_sp - prev_sp) / prev_sp * 100, 1)
            case_diffs.append({"case_id": cid, "prev_speedup": prev_sp, "curr_speedup": curr_sp, "delta": f"+{cdelta}%" if cdelta > 0 else f"{cdelta}%"})
    case_diffs.sort(key=lambda x: float(x["delta"].replace("%", "").replace("+", "")), reverse=True)
    result["case_diffs"] = case_diffs

    # 格式化 diff 文本
    if result["has_improvement"]:
        lines = [
            f"\n📊 性能提升检测（程序自动计算，数据可信）",
            f"  avg_speedup: {prev_avg} → {curr_avg} (+{delta_pct}%)",
            f"  对比轮次: iter{prev_iter} → iter{iteration}",
        ]
        if result.get("prev_design_path"):
            lines.append(file_hint(work_dir, result['prev_design_path'], "上轮评测附近的历史设计参考",
                                   "用于回顾改动意图；实际被评测版本以代码绑定的选择依据和快照为准").rstrip())
        if result.get("curr_design_path"):
            lines.append(file_hint(work_dir, result['curr_design_path'], "本轮评测附近的历史设计参考",
                                   "与上轮参考比较；先用代码绑定依据或快照确认版本，再归因成功经验").rstrip())
        if case_diffs:
            lines.append(f"  逐 case 对比（提升最大的在前）:")
            for cd in case_diffs[:6]:
                lines.append(f"    case_{cd['case_id']}: {cd['prev_speedup']} → {cd['curr_speedup']} ({cd['delta']})")
        lines.append("")
        lines.append("⚡ 本轮性能大幅提升！请对照上面两轮的 design_rationale，总结这次改动为什么有效，填写 proven_pattern 字段。")
        result["diff_text"] = "\n".join(lines)

    elif result["has_regression"]:
        lines = [
            f"\n⚠️ 性能退步检测（程序自动计算，数据可信）",
            f"  avg_speedup: {prev_avg} → {curr_avg} ({delta_pct}%)",
            f"  对比轮次: iter{prev_iter} → iter{iteration}",
        ]
        if result.get("prev_design_path"):
            lines.append(file_hint(work_dir, result['prev_design_path'], "上轮评测附近的历史设计参考",
                                   "用于回顾退步前的设计意图；实际被评测版本以绑定依据和快照为准").rstrip())
        if result.get("curr_design_path"):
            lines.append(file_hint(work_dir, result['curr_design_path'], "本轮评测附近的历史设计参考",
                                   "先用代码绑定依据或快照确认版本，再结合慢 case 核对改动并归因失败教训").rstrip())
        if case_diffs:
            lines.append(f"  逐 case 对比（退步最大的在前）:")
            regression_cases = sorted(case_diffs, key=lambda x: float(x["delta"].replace("%", "").replace("+", "")))
            for cd in regression_cases[:6]:
                lines.append(f"    case_{cd['case_id']}: {cd['prev_speedup']} → {cd['curr_speedup']} ({cd['delta']})")
        lines.append("")
        lines.append("🚨 本轮性能退步！请对照上面两轮的 design_rationale，分析原因并填写 regression_pattern，明确适用条件；在已验证条件下避免重复失败，改变条件后须重新验证。")
        result["regression_text"] = "\n".join(lines)

    return finish()


def _history_previous_average(work_dir, state, log=None, *, comparison=None):
    """Read the preceding round's average, never a same-round retry or old score.

    Some legacy state.speedup values contain the overall score. Only the explicit
    avg_speedup in the evaluation report/history is a reliable comparison source.
    Zero/failed previous rounds are retained; this is not the ±5% knowledge scan.
    """
    if comparison is not None:
        return comparison.get("prev_avg_speedup") if comparison.get("comparable") else None
    # Legacy scalar lookup is not proof of comparability. The main workflow
    # always supplies the checked comparison above before writing a verdict.
    rounds = [item for item in load_history(work_dir).get("rounds", [])
              if type(item.get("iter")) is int and item["iter"] < state.iteration]
    previous = state.get_previous_eval()
    candidates = [item["iter"] for item in rounds]
    if previous is not None:
        candidates.append(previous["iteration"])
    if not candidates:
        return 0.0
    previous_iteration = max(candidates)
    report = Path(work_dir) / "eval" / f"iter{previous_iteration}" / "perf_result.json"
    value = None
    found = False
    try:
        payload = _json_module.loads(report.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and "avg_speedup" in payload:
            value, found = payload["avg_speedup"], True
    except (OSError, ValueError):
        pass
    if not found:
        record = next((item for item in reversed(rounds) if item["iter"] == previous_iteration), None)
        if record is not None:
            value = record.get("avg_speedup")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value
    if log:
        log.warning("[历史账本] iter%s 的上一轮 iter%s 缺少有效 avg_speedup；前值记为 null，不使用旧 state.speedup 代替。",
                    state.iteration, previous_iteration)
    return None


def _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                           node_logs, iter_dirs, device_info_prompt, selection_context):
    if not HumanReview(work_dir).pending_messages():
        return
    context = state.stage9_context
    reason = context.get("fail_reason") or "score_zero: 尚无有效评测，仅处理人工意见与未执行原因"
    run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
                  fail_reason=reason, device_info_prompt=device_info_prompt,
                  perf_diff=context.get("perf_diff"), selection_context=selection_context)


def run_tech_lead(log, roles_dir: str, work_dir: str, op_name: str, state, state_log, node_logs, iter_dirs, fail_reason: str = "", device_info_prompt: str = "", perf_diff: dict = None, selection_context=None):
    """Persist the scene, consult when due, then deliver a validated final decision."""
    restoration = load_regression_action(work_dir, state.iteration)
    if restoration and restoration.get("status") == "prepared":
        # A crash between the directory renames must finish before any agent
        # inspects impl. The already committed Stage9 decision remains the basis.
        apply_regression_action(work_dir, state.iteration, restoration["decision_path"], log, state_log,
                                comparison_context=selection_context)
    human = HumanReview(work_dir)
    settings = getattr(state, "human_review_config", {})
    selection = load_selection_status(work_dir, comparison_context=selection_context)
    scene = classify_scene(fail_reason, selection, iteration=state.iteration)
    previous = getattr(state, "stage9_context", {})
    same = previous.get("iteration") == state.iteration and previous.get("fail_reason") == fail_reason
    if same and previous.get("scene"):
        scene = previous["scene"]  # Includes consultation recovery: never reinterpret its original task.
    if not is_performance_scene(scene):
        perf_diff = {}  # A failed build/precision run has no new performance evidence.
    else:
        perf_diff = _compute_perf_diff(work_dir, state.iteration, expected_context=selection_context,
                                      log=log, state_log=state_log)
    context = copy.deepcopy(previous) if same else {
        "iteration": state.iteration, "fail_reason": fail_reason, "scene": scene,
        "message_ids": [], "phase": "reviewing",
    }
    context.update(perf_diff=perf_diff, scene=scene)
    if is_performance_scene(scene):
        event = log_semantic_trigger(work_dir, state.iteration, selection, log, state_log)
        if event:
            context["semantic_event"] = event
        prepare_regression_action(work_dir, state.iteration, perf_diff, selection, log, state_log)
    state.stage9_context = context
    transition(state, "stage9", state_log, reason=f"保留原场景 {scene}")
    valid = bool(is_performance_scene(scene) and selection.get("eligible")
                 and selection.get("latest_iteration") == state.iteration)
    counter = human.record_evaluation(
        state.iteration, valid=valid, all_passed=scene == "all_passed",
        review_fusion=bool(selection.get("review_fusion")),
        window_gain=(selection.get("window") or {}).get("cumulative_improvement"))
    consultation = human.active_consultation()
    if (not consultation and settings.get("consultation_enabled") and valid
            and scene == "stagnation" and counter["consultation_due"]
            and not state.stopped_by):
        consultation = human.create_consultation(
            state.iteration, scene, fail_reason, evidence_sources(work_dir, iter_dirs),
            {"selection_status": selection, "perf_diff": perf_diff,
             "stage9_context": context, "hardware_prompt": device_info_prompt})
    bundle_path = context.get("bundle_path")
    if consultation:
        if consultation["iteration"] != state.iteration or consultation["fail_reason"] != fail_reason:
            raise RuntimeError("人工咨询与当前 Stage9 场景不一致，禁止将旧问题用于新版本")
        context["consultation_id"] = consultation["request_id"]
        context["phase"] = "consulting"
        state.flush()
        if consultation["status"] == "draft":
            question_path = Path(consultation["question_path"])
            if not question_path.is_file():
                _run_tech_lead_decision(
                    log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
                    fail_reason, device_info_prompt, perf_diff, selection_context,
                    consultation=consultation, human_messages=human.pending_messages(), scene=scene)
            log.warning("[人工咨询] 已连续 3 次触发有效提升不足 5%%，请阅读 %s；等待 2 分钟，回复“请等待”仅额外延长一次 10 分钟。", question_path)
            state_log.warning("[人工咨询] 问题=%s；原场景=%s", question_path, scene)
            consultation = human.start_wait(consultation["request_id"])
        if consultation["status"] == "waiting":
            log.info("[人工等待] 请求=%s，沿用截止时间=%s", consultation["request_id"], consultation["deadline"])
            consultation = human.wait_consultation(consultation["request_id"])
        bundle_path = human.build_feedback_bundle(consultation["request_id"])
        context.update(bundle_path=bundle_path, phase="feedback")
        state.flush()
        log.info("[人工复议] 状态=%s，完整问题/对话/证据=%s", consultation["status"], bundle_path)

    while True:
        pending = human.pending_messages()
        # Pending messages submitted while an enabled workflow was running stay visible
        # even if its switch was subsequently disabled during checkpoint recovery.
        ids = set(context.get("message_ids", [])) | {item["id"] for item in pending}
        messages = [item for item in human.all_messages() if item["id"] in ids]
        context["message_ids"] = [item["id"] for item in messages]
        state.flush()
        committed_path = context.get("decision_path")
        if messages and not consultation and (pending or not bundle_path):
            bundle_path = human.create_proactive_bundle(
                state.iteration, scene, fail_reason, evidence_sources(work_dir, iter_dirs),
                {"selection_status": selection, "previous_decision": committed_path,
                 "consultation_bundle": context.get("bundle_path")}, messages)
            context["bundle_path"] = bundle_path
            state.flush()
        decision_path = _run_tech_lead_decision(
            log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
            fail_reason, device_info_prompt, perf_diff, selection_context,
            human_messages=messages, bundle_path=bundle_path, scene=scene)
        human.mark_processed(context["message_ids"], decision_path)
        if consultation:
            human.complete_consultation(consultation["request_id"], decision_path)
            consultation = None
        context.update(phase="committed", decision_path=str(decision_path))
        state.flush()
        if not human.pending_messages():
            return str(decision_path)
        log.info("[人工复议] Stage9 决策生成期间收到新意见，同一轮重新审查后再交付")


def _stage9_case_catalog(work_dir, iteration, scene):
    """Bind known IDs to this evaluation; never guess case numbers from filenames."""
    report = Path(work_dir) / "eval" / f"iter{iteration}" / "perf_result.json"
    catalog = {"iteration": iteration, "case_ids": None, "source": None,
               "complete": False, "observed_case_ids": [],
               "read_hint": "未获得本轮完整 case ID 清单；按 task/cases.yaml 与实际代码核对，不能声称程序已验证映射。"}
    if is_performance_scene(scene) or scene == "evaluation_error":
        try:
            payload = _json_module.loads(report.read_text(encoding="utf-8-sig"))
            cases = payload.get("cases", [])
            total = payload.get("total_cases")
        except (OSError, ValueError, AttributeError):
            cases, total = [], None
        if isinstance(cases, list) and cases and all(isinstance(case, dict) and isinstance(case.get("case_id"), str)
                         and case["case_id"].strip() for case in cases):
            observed = list(dict.fromkeys(case["case_id"] for case in cases))
            catalog.update(source=f"eval/iter{iteration}/perf_result.json", observed_case_ids=observed)
            if type(total) is int and total > 0 and len(observed) == len(cases) == total:
                catalog.update(case_ids=observed, complete=True,
                               read_hint="原样使用本轮全部 case ID；不局限最慢六例。文件映射须另附实际路由和代码依据。")
    return catalog


def _write_stage9_plan_contract(work_dir, request_dir, iteration, request_id,
                                performance_case_ids, allow_empty, case_catalog, perf_diff=None,
                                has_question=False):
    """Persist the program-owned contract before the model sees its request."""
    schema_path = request_dir / "decision_schema.json"
    template_path = request_dir / "decision_template.json"
    catalog_path = request_dir / "case_catalog.json"
    schema = build_decision_schema(perf_diff=perf_diff or {},
                                   performance_case_ids=performance_case_ids, has_question=has_question)
    schema["properties"]["iteration"]["const"] = iteration
    schema["properties"]["request_id"]["const"] = request_id
    atomic_write_json(str(schema_path), schema)
    atomic_write_json(str(template_path), build_decision_template(
        iteration, request_id, performance_case_ids, allow_empty_suggestions=allow_empty,
        perf_diff=perf_diff, has_question=has_question))
    atomic_write_json(str(catalog_path), case_catalog)
    hint = (
        file_hint(work_dir, schema_path, "程序定义的 Stage9 JSON 格式（只读）",
                  "按本轮字段、类型及层级填写；未列出的经验字段必须省略；case_analysis 按本轮要求填写；修改文件由 changes 自动汇总")
        + file_hint(work_dir, template_path, "本请求的决策填写模板（只读）",
                    "复制 iteration/request_id，填写真实结论、任务和已列出的性能经验；经验 case 按实测增补受益及受损项；空白及 null 必须按 schema 填写")
        + file_hint(work_dir, catalog_path, "本轮可核对的完整 case ID 清单（只读）",
                    "已列出 ID 时按原名选择；没有清单时回查 task，不把最慢六例当成全部用例")
    )
    return {"schema_path": str(schema_path), "template_path": str(template_path),
            "case_catalog_path": str(catalog_path)}, hint


def _stage9_implementation_base(work_dir, iteration):
    restoration = load_regression_action(work_dir, iteration)
    if (restoration and restoration.get("action") == "restore_best"
            and restoration.get("status") in {"planned", "prepared"}):
        return Path(restoration["best_record"]["implementation_dir"])
    return Path(work_dir) / "impl"


def _validate_stage9_file_targets(work_dir, decision, *, allow_existing_creates=False,
                                  implementation_base=None, raw_plan=False):
    """Collect independent file errors without authorizing or changing a plan.

    Raw v2 output is also accepted for diagnostics when another contract check
    failed. Only typed, lexically safe paths reach the filesystem. Readonly
    input links may resolve outside work; writable targets may not.
    """
    import stat
    from lib.stage9_plan import _path

    root = Path(work_dir).resolve()
    implementation_root = Path(implementation_base).resolve() if implementation_base is not None else None
    # The ordinary work/impl path is never a rollback exemption, even if it
    # happens to link into a protected snapshot directory.
    checking_restore_base = implementation_base is not None and not (
        Path(implementation_base).parent.resolve() == root
        and os.path.normcase(Path(implementation_base).name) == os.path.normcase("impl")
    )
    protected_names = {os.path.normcase(name) for name in (
        "task", "example", "knowledge", "eval", "selection", "fusion", "human_review",
        "build", "profile", "search", "log", "operator_iter", ".state.json", "ANALYSIS.md",
        "fusion_requirements.en.json", "device_info.json", "init_impl_manifest.json",
    )}
    errors, seen_errors = [], set()
    stat_cache, resolution_cache = {}, {}
    stat_failed = object()

    def error(message, *, key=None):
        key = message if key is None else key
        if key not in seen_errors:
            seen_errors.add(key)
            errors.append(message)

    def path_value(value, field):
        if not isinstance(value, str):
            error(f"{field}: 文件路径必须是字符串，收到 {value!r}")
            return None
        try:
            return _path(value, field)
        except ValueError as exc:
            error(str(exc))
            return None

    def paths(value, field):
        if not isinstance(value, list):
            error(f"{field}: 必须是文件路径数组")
            return []
        result = []
        for index, item in enumerate(value):
            label = f"{field}[{index}]"
            path = path_value(item, label)
            if path is not None:
                result.append((path, label))
        return result

    def protected(relative):
        return bool(relative.parts) and os.path.normcase(relative.parts[0]) in protected_names

    def file_stat(path, field):
        if path not in stat_cache:
            try:
                stat_cache[path] = path.stat()
            except (FileNotFoundError, NotADirectoryError):
                stat_cache[path] = None
            except OSError as exc:
                error(f"{field}: Stage9 无法核对文件属性：{path} ({exc})", key=("stat", path))
                stat_cache[path] = stat_failed
        return stat_cache[path]

    def physical(path):
        if implementation_base is not None and path.startswith("impl/"):
            return Path(implementation_base) / path[len("impl/"):]
        return root / path

    def resolve(path, field, *, writing=False):
        cache_key = (path, writing)
        if cache_key in resolution_cache:
            return resolution_cache[cache_key]
        resolution_cache[cache_key] = None
        logical = Path(path)
        if writing and protected(logical):
            error(f"{field}: Stage9 不可修改只读输入或程序维护文件：{path}",
                  key=("protected", path))
            return None
        try:
            target = physical(path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            error(f"{field}: Stage9 无法解析文件路径：{path} ({exc})", key=("resolve", path))
            return None
        if writing and (not target.is_relative_to(root) or target == root):
            error(f"{field}: Stage9 文件范围必须位于工作目录内：{path}", key=("outside", path))
            return None
        if writing:
            relative = target.relative_to(root)
            # Checking a pending restoration never grants permission to edit
            # the snapshot itself or links escaping the selected snapshot.
            selected_snapshot = (
                logical.parts[0] == "impl" and checking_restore_base
                and target.is_relative_to(implementation_root)
                and os.path.normcase(relative.parts[0]) == os.path.normcase("selection")
            )
            if protected(relative) and not selected_snapshot:
                error(f"{field}: Stage9 修改路径实际指向只读输入或程序维护文件：{path}",
                      key=("actual_protected", path))
                return None
        info = file_stat(target, field)
        if info is stat_failed:
            return None
        if info is not None and stat.S_ISDIR(info.st_mode):
            error(f"{field}: Stage9 必须列出具体文件，不能用目录代替：{path}",
                  key=("directory", path))
            return None
        result = (target, info, path, field)
        resolution_cache[cache_key] = result
        return result

    def require_file(record, message):
        if record is None:
            return
        _, _, path, field = record
        # Use the logical link for existence as well; readonly input links are
        # intentionally permitted to point outside the work directory.
        info = file_stat(physical(path), field)
        if info is not stat_failed and (info is None or not stat.S_ISREG(info.st_mode)):
            error(f"{field}: {message}：{path}", key=("missing", path))

    if not isinstance(decision, dict):
        raise Stage9DecisionValidationError(["Stage9 文件检查需要 decision 对象"])
    # The request requires v2 even when the model forgot/invalidated its version
    # field. Do not invent a missing legacy writable list as another error.
    raw_v2 = raw_plan or decision.get("plan_version") == 2
    ledger = decision.get("ledger_entry", {})
    if not isinstance(ledger, dict):
        error("ledger_entry: 必须是对象")
        ledger = {}
    suggestions = decision.get("suggest_next", [])
    if not isinstance(suggestions, list):
        error("suggest_next: 必须是数组")
        suggestions = []
    tasks, all_writes = [], {}
    for index, suggestion in enumerate(suggestions):
        label = f"suggest_next[{index}]"
        if not isinstance(suggestion, dict):
            error(f"{label}: 必须是任务对象")
            continue
        task_id = suggestion.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            label += f" ({task_id})"
        changes = suggestion.get("changes", [])
        if not isinstance(changes, list):
            error(f"{label}.changes: 必须是数组")
            changes = []
        writes, created = {}, set()
        for change_index, change in enumerate(changes):
            field = f"{label}.changes[{change_index}]"
            if not isinstance(change, dict):
                error(f"{field}: 必须是对象")
                continue
            path = path_value(change.get("file"), f"{field}.file")
            operation = change.get("operation")
            if operation not in ("create", "modify", "inspect"):
                error(f"{field}.operation: 必须是 create、modify 或 inspect")
                continue
            if path is None:
                continue
            record = resolve(path, f"{field}.file", writing=operation != "inspect")
            if record is None:
                continue
            if operation != "inspect":
                writes[record[0]] = record
            if operation == "create":
                created.add(path)
                info = file_stat(physical(path), record[3])
                if info is not None and info is not stat_failed and not allow_existing_creates:
                    error(f"{record[3]}: 声明新建但文件已存在，请改为 modify：{path}",
                          key=("create_exists", path))
            else:
                require_file(record, "检查/修改的实际文件不存在")
        if not raw_v2:
            for path, field in paths(suggestion.get("modify_files", []), f"{label}.modify_files"):
                record = resolve(path, field, writing=True)
                if record is not None:
                    writes[record[0]] = record
            for path, field in paths(suggestion.get("inspect_files", []), f"{label}.inspect_files"):
                resolve(path, field)
        tasks.append((suggestion, label, writes, created))
        all_writes.update(writes)

    # Raw plans have no derived permission lists. These local sets exist only
    # for diagnostics; they are never written back or accepted as a handoff.
    writable = dict(all_writes) if raw_v2 else {}
    if not raw_v2:
        for path, field in paths(ledger.get("modify_files", []), "ledger_entry.modify_files"):
            record = resolve(path, field, writing=True)
            if record is not None:
                writable[record[0]] = record
    readonly = {}
    readonly_identities = {}
    for path, field in paths(ledger.get("readonly_files", []), "ledger_entry.readonly_files"):
        record = resolve(path, field)
        if record is not None:
            readonly[record[0]] = record
            if record[1] is not None and record[1].st_ino:
                readonly_identities[(record[1].st_dev, record[1].st_ino)] = record

    for target, record in {**writable, **all_writes}.items():
        _, info, path, field = record
        conflict = readonly.get(target)
        if conflict is None and info is not None and info.st_ino:
            conflict = readonly_identities.get((info.st_dev, info.st_ino))
        if conflict is not None:
            error(f"{field}: Stage9 允许修改与只读路径实际指向同一文件：{path}；"
                  f"{conflict[3]}={conflict[2]}", key=("readonly_conflict", target))
        if info is not None and info.st_nlink > 1:
            error(f"{field}: Stage9 待写文件存在多个硬链接，需先使用独立副本：{path}",
                  key=("hardlink", target))
        for parent in target.parents:
            if parent in writable or parent in all_writes:
                error(f"{field}: Stage9 待写文件不能同时作为另一待写文件的目录：{parent}；{path}",
                      key=("write_parent", parent, target))
            parent_info = file_stat(parent, field)
            if parent_info is not None and parent_info is not stat_failed and not stat.S_ISDIR(parent_info.st_mode):
                error(f"{field}: Stage9 待写文件的父级已是文件，无法创建或修改：{parent}；{path}",
                      key=("file_parent", parent, target))
                break
            if parent_info is stat_failed or parent == root:
                break

    for suggestion, label, writes, created in tasks:
        for target, record in writes.items():
            if target not in writable:
                error(f"{record[3]}: 修改目标与实际文件权限范围冲突：{record[2]}",
                      key=("undeclared_write", target))
        bindings = suggestion.get("case_bindings", [])
        if not isinstance(bindings, list):
            error(f"{label}.case_bindings: 必须是数组")
            continue
        for index, binding in enumerate(bindings):
            field = f"{label}.case_bindings[{index}]"
            if not isinstance(binding, dict):
                error(f"{field}: 必须是对象")
                continue
            for path, location in paths(binding.get("implementation_files", []), f"{field}.implementation_files"):
                record = resolve(path, location)
                if path not in created:
                    require_file(record, "case 关联的实现文件不存在")
            evidence_items = binding.get("route_evidence", [])
            if not isinstance(evidence_items, list):
                error(f"{field}.route_evidence: 必须是数组")
                continue
            for evidence_index, evidence in enumerate(evidence_items):
                location = f"{field}.route_evidence[{evidence_index}].file"
                if not isinstance(evidence, dict):
                    error(f"{field}.route_evidence[{evidence_index}]: 必须是对象")
                    continue
                path = path_value(evidence.get("file"), location)
                if path is not None:
                    require_file(resolve(path, location), "路由依据文件不存在")
    if errors:
        raise Stage9DecisionValidationError(errors)


def _run_tech_lead_decision(log, roles_dir, work_dir, op_name, state, state_log, node_logs,
                            iter_dirs, fail_reason="", device_info_prompt="", perf_diff=None,
                            selection_context=None, *, human_messages=None, bundle_path=None,
                            consultation=None, scene=None):
    """
    接收本次 Stage9 decision，由程序合并 history 并保存经验。
    支持在任意回退点调用（build_fail/precision_fail/score_zero/perf_optimize）。
    根据实际存在的文件灵活组装 prompt。
    """
    import json as _json
    stage = f"ITER{state.iteration}-阶段9-tech_lead({fail_reason})"
    # A resumed state may contain an old unguarded diff. Rebuild it from the
    # stored measurement context before generating a role or committing knowledge.
    if scene is None:
        scene = classify_scene(fail_reason, load_selection_status(
            work_dir, comparison_context=selection_context), iteration=state.iteration)
    if is_performance_scene(scene):
        perf_diff = _compute_perf_diff(work_dir, state.iteration, expected_context=selection_context,
                                      log=log, state_log=state_log)
        refresh_round_comparison(work_dir, state.iteration, perf_diff)
    else:
        perf_diff = {}
    history_path = os.path.join(work_dir, "knowledge", "history.json")
    history_before = copy.deepcopy(load_history(work_dir))
    request_id = uuid.uuid4().hex
    request_dir = Path(work_dir) / "knowledge" / "stage9" / f"iter{state.iteration}" / request_id
    decision_path = request_dir / "decision.json"
    role = str(request_dir / "n4_stage9_tech_lead_guide.md")
    atomic_write_json(str(request_dir / "history_before.json"), history_before)
    selection_status = load_selection_status(work_dir, comparison_context=selection_context)
    if scene is None:
        scene = classify_scene(fail_reason, selection_status, iteration=state.iteration)
    performance_scene = is_performance_scene(scene)
    current_round = next((item for item in reversed(history_before.get("rounds", []))
                          if item.get("iter") == state.iteration), {})
    performance_case_ids = ([item["case_id"] for item in current_round.get("worst_6_speedups", [])]
                            if performance_scene else [])
    case_catalog = _stage9_case_catalog(work_dir, state.iteration, scene)
    known_case_ids = case_catalog["case_ids"]
    implementation_base = _stage9_implementation_base(work_dir, state.iteration)
    if not performance_scene:
        perf_diff = {}

    append_work_record(work_dir, stage)
    transition(state, "stage9", state_log, reason=f"tech_lead经验总结({fail_reason})")

    # 收集所有已存在的 design_rationale
    all_designs = []
    for i in (range(0, state.iteration + 1) if performance_scene
              else range(max(0, state.iteration - 1), state.iteration + 1)):
        candidate = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
        if os.path.exists(candidate):
            all_designs.append(file_hint(work_dir, candidate, f"iter{i} 的实现设计思路",
                                         "按轮比较算法、tiling、数据流与改动理由，再对照实际性能验证").rstrip())
    design_hint = "历轮设计思路：\n" + "\n".join(all_designs) + "\n" if all_designs else ""

    # 根据可用文件组装 prompt
    data_lines = []
    default_inputs = []
    missing_inputs = []
    for key, name, path, reading in [
        ("fusion_library", "JSON 融合算子库", str(fusion_library_path(work_dir)), "比较候选数据流、前提与初始概率，以实测证据决定保留或更换"),
        ("perf_result", "本轮性能结果", f"{iter_dirs['eval']}/perf_result.json", "看 avg_speedup、各 case 的 speedup 和最慢用例，不只看平均值"),
        ("profiler", "本轮 profiler 数据（按需回查）", f"{iter_dirs['eval']}/prof_data/", "仅在结论矛盾或缺少证据时按 case 回查"),
        ("perf_reports", "本轮性能报告", f"{iter_dirs['eval']}/perf_reports/", "回查原始计时与评分，核实性能汇总及反作弊信号"),
        ("bottleneck", "本轮瓶颈分析", f"{iter_dirs['profile']}/bottleneck_analysis.md", "优先读 Stage7 结论，区分已证实根因和推测"),
        ("fix_directive", "本轮修改指令", f"{iter_dirs['search']}/FIX_DIRECTIVE.md", "优先读 Stage8 结论，转成原有 P0/P1/P2 优先级"),
        ("search_report", "本轮搜索报告", f"{iter_dirs['search']}/SEARCH_REPORT.md", "按需检查候选方法、依据及适用条件"),
        ("build_log", "本轮编译日志", f"{iter_dirs['build']}/build.log", "看 STATUS 和首个真实错误，定位构建/接口问题"),
        ("precision_result", "本轮精度结果", f"{iter_dirs['eval']}/precision_result.json", "核对总数、通过数及失败 case，优先处理正确性"),
        ("precision_reports", "本轮精度报告详情", f"{iter_dirs['eval']}/precision_reports/", "按失败 case 查误差、输入和原始报错"),
        ("self_test", "同编号开发轮次的自测报告（若存在）", f"{iter_dirs['develop']}/self_test_report.md", "查看开发自测记录；评测代码的依据以另附的哈希绑定文件为准"),
    ]:
        if key in scene_input_keys(scene):
            entry = {"key": key, "path": str(Path(path)), "purpose": name, "read_hint": reading}
            if os.path.exists(path):
                default_inputs.append(entry)
                data_lines.append(file_hint(work_dir, path, name, reading).rstrip())
            else:
                missing_inputs.append(entry)

    fail_hint = f"本轮失败原因：{fail_reason}\n" if fail_reason else ""

    # 注入已验证的成功经验 + 失败教训 + 错题本供 tech_lead 参考
    patterns_hint = ((format_proven_patterns_for_prompt(work_dir)
                     + format_regression_patterns_for_prompt(work_dir)
                     + format_pitfalls_for_prompt(work_dir)) if performance_scene
                     else format_pitfalls_for_prompt(work_dir))

    # ── 查找上一轮的 question.md（下级反馈），存在且未裁定则注入让 tech_lead 裁定 ──
    question_hint = ""
    question_path_found = None
    for i in range(state.iteration - 1, -1, -1):
        cand = os.path.join(work_dir, "develop", f"iter{i}", "question.md")
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as _qf:
                _qcontent = _qf.read()
            if "## tech_lead 裁定" in _qcontent:
                continue  # 已裁定过，跳过
            question_path_found = cand
            question_hint = (
                "\n" + file_hint(work_dir, cand, "下级反馈待裁定（question.md）",
                                  "核对被质疑的历史建议和硬证据，再决定 confirmed 或 rejected") +
                f"stage3 cannbot 反馈你上一轮的某条建议有误。请按当前条件任务裁定：\n"
                f"读该文件 → 对照 history 中它引用的意见 → 输出 pitfall 字段（confirmed/rejected）。\n"
            )
            break

    # ── 使用外部传入的 perf_diff（程序在 stage6 后已算好）──
    perf_diff_hint = (perf_diff.get("comparison_text", "") + perf_diff.get("diff_text", "")
                      + perf_diff.get("regression_text", ""))
    if consultation:
        perf_diff_hint = "\n程序计算的性能变化（仅作为咨询依据）：" + _json.dumps(
            {key: value for key, value in perf_diff.items() if not key.endswith("_text")}, ensure_ascii=False) + "\n"
        if question_path_found:
            question_hint = file_hint(work_dir, question_path_found, "待裁定开发异议（咨询背景）",
                                      "本次仅用于问题背景，最终复议阶段再形成裁定")

    profiling_skill_path_9 = os.path.join(os.path.dirname(roles_dir), "skills", "triton-profiling-analysis", "SKILL.md")
    profiling_skill_hint_9 = (
        "\n" + file_hint(work_dir, profiling_skill_path_9, "Profiling 分析指南",
                          "使用五文件分析流程和权重核实瓶颈结论",
                          base_dir=Path(__file__).parent, base_label="项目根目录") +
        f"分析 profiler 数据时，请加载上述 skill 文件，按五文件分析流程和权重分配进行结构化解读，"
        f"重点关注 kernel_details.csv（权重50%）中是否有多余 kernel、核是否打满、有无编译降级。\n"
    ) if scene == "all_passed" and os.path.exists(profiling_skill_path_9) else ""

    decision_output_hint = file_hint(
        work_dir, decision_path, "本次 Stage9 决策输出（待你生成）",
        "按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段")
    prompt = (
        f"工作目录：{work_dir}\n算子：{op_name}\n"
        + file_hint(work_dir, Path(work_dir) / 'task', "只读算子需求和评测基准",
                    "对照 desc.md、proto.yaml、cases.yaml、golden.py 判断建议是否保持语义和完整 case 覆盖")
        + file_hint(work_dir, Path(work_dir) / 'ANALYSIS.md', "Stage1 需求分析",
                    "核对接口、精度、硬件约束和实现难点，避免偏离需求")
        + (file_hint(work_dir, Path(roles_dir).parent / 'knowledge/anti_cheat_reference.md', "反作弊判定参考",
                    "按实际错误处理表对照错误码、kernel CSV 和真实代码定位违规或零分原因",
                    base_dir=Path(roles_dir).parent, base_label="项目根目录") if scene == "evaluation_error" or performance_scene else "")
        + file_hint(work_dir, Path(work_dir) / 'impl', "当前代码", "按本场景问题核对入口和相关实现")
        + file_hint(work_dir, history_path, "跨轮经验、性能轮次与修改账本",
                    ("只读：查看与本次问题相关的历史方向和实测；此时不提交账本" if consultation else
                     "只读：先读 suggest_next、insights、ledger 和 rounds；仅向本次 decision 提交当前轮账本，程序负责合并历史")) +
        f"{device_info_prompt}\n"
        f"{fail_hint}"
        f"{patterns_hint}"
        f"{perf_diff_hint}"
        f"{question_hint}"
        + "\n".join(data_lines) + "\n"
        + ("当前场景缺失材料（不能当作已完成结论）：\n" + "".join(
            file_hint(work_dir, item["path"], item["purpose"] + "（缺失）", "未生成或不可用；明确证据不足，不使用旧结论替代")
            for item in missing_inputs if item["key"] != "self_test") if missing_inputs else "")
        + design_hint
        + profiling_skill_hint_9
        + ("请分析本轮结果，回顾历轮设计思路；" if performance_scene
           else "请定位本轮故障，回顾与该问题有关的设计和上轮建议；")
        + f"只读 {history_path}，不要覆盖它。\n"
        + f"Stage9 输出文件：{decision_path}\n"
        + f"Stage9 请求编号：{request_id}\n"
        + f"Stage9 当前轮次：{state.iteration}\n"
        + decision_output_hint
    )
    if performance_scene and not consultation:
        prompt += ("\n本次 ledger_entry.case_analysis 必须逐一覆盖以下 case_id（原样复制，不能缩写；"
                   "每例写 observation、explanation、evidence、next_action；根因未证实须注明待验证）：\n"
                   + _json.dumps(performance_case_ids, ensure_ascii=False) + "\n")
    if state.stage9_context.get("scope_review_error") and not consultation:
        prompt += ("\nStage3 交付前检查发现旧计划缺少明确文件范围或存在冲突，尚未执行；"
                   "请重新提交本轮完整决策，并消除以下问题：\n"
                   + state.stage9_context["scope_review_error"] + "\n")

    if performance_scene:
        prompt += format_fusion_library_for_prompt(work_dir, "stage9")
    else:
        prompt += file_hint(work_dir, fusion_library_path(work_dir), "初始融合候选库（仅按需回查）", "本次先修编译或正确性；人工方向涉及融合方法时可回查概率及前提")
    prompt += format_evidence_for_prompt(work_dir)
    if performance_scene:
        prompt += format_selection_for_prompt(work_dir, comparison_context=selection_context)
        prompt += regression_action_prompt(work_dir, state.iteration)
        event = state.stage9_context.get("semantic_event")
        if event:
            prompt += (f"\n程序记录：本轮第 {event['entry_count']} 次进入 {event['scene']} 停滞场景；"
                       "这是历次触发的累计次数，与人工咨询的三次计数独立。\n")
            prompt += file_hint(work_dir, event["state_path"], "语义停滞触发及次数记录",
                                "核对当前迭代、比较组、窗口和 entry_count；恢复同一轮不重复计数")
    # Read the persisted window here too, so a Stage9 resume gets the same review
    # instructions without counting another evaluation or changing failure routes.
    selection_status = load_selection_status(work_dir, comparison_context=selection_context)
    review_hint = (fusion_review_hint(selection_status, state.iteration)
                   if fail_reason == "perf_optimize" else "")
    if review_hint:
        prompt += "\n\n" + review_hint + "\n"
        log.warning("[Stage9 提示词追加]\n%s", review_hint)
        evidence = load_evidence(work_dir)
        sources = [
            ("Stage9 role", role), ("需求分析", f"{work_dir}/ANALYSIS.md"),
            ("task", f"{work_dir}/task"), ("历史经验", history_path),
            ("硬件来源", f"{work_dir}/device_info.json"),
            ("反作弊参考", os.path.join(os.path.dirname(roles_dir), "knowledge", "anti_cheat_reference.md")),
            ("成功经验", f"{work_dir}/knowledge/proven_patterns.md"),
            ("退步教训", f"{work_dir}/knowledge/regression_patterns.md"),
            ("错题本", f"{work_dir}/knowledge/tech_lead_pitfalls.md"),
            ("当前代码与证据绑定", f"{work_dir}/selection/current_implementation.json"),
            ("语义窗口及历史索引", f"{work_dir}/selection/state.json"),
            ("历史评测快照及 case 趋势来源", f"{work_dir}/selection/records"),
            ("Profiling 指南", profiling_skill_path_9),
        ]
        sources += [(f"当前实现证据/{name}", path)
                    for name, path in evidence.get("evidence_paths", {}).items()]
        best = selection_status["best"]
        sources += [(f"最佳实现/{name}", best[name]) for name in
                    ("manifest_path", "implementation_dir", "performance_report", "precision_report")]
        if question_path_found:
            sources.append(("待裁定问题", question_path_found))
        input_lines = data_lines + all_designs + [f"{name}：{Path(path)}" for name, path in sources
                                                 if os.path.exists(path)]
        log.info("[Stage9 融合重点审查输入及证据来源]\n%s", "\n".join(input_lines))
    allow_empty = bool(selection_status.get("eligible") and selection_status.get("should_exit")
                       and selection_status.get("latest_iteration") == state.iteration
                       and fail_reason.startswith("perf_pass"))
    if allow_empty and not consultation:
        prompt += "\n程序已确认本轮语义退出，suggest_next 可为 []，完成经验记录即可。\n"
    if not consultation:
        prompt += human_prompt(work_dir, human_messages or [], bundle_path,
                               ending_reason=state.stopped_by or ("semantic_stagnation" if allow_empty else None))
    phase = "consultation" if consultation else ("feedback" if bundle_path else "decision")
    contract_paths, contract_hint = {}, ""
    if not consultation:
        contract_paths, contract_hint = _write_stage9_plan_contract(
            work_dir, request_dir, state.iteration, request_id, performance_case_ids,
            allow_empty, case_catalog, perf_diff=perf_diff, has_question=bool(question_path_found))
        prompt += "\n程序定义的填写格式（先读后填）：\n" + contract_hint
        prompt += file_hint(
            work_dir, implementation_base, "本轮建议实际实施的代码基底（只读查阅）",
            "在此核对 case 路由与待改函数；changes 仍填写实施时的 impl/... 相对路径。若本轮需回退最佳版本，不按退步代码臆造修改位置")
    atomic_write_text(role, build_scene_role(
        roles_dir, scene, phase=phase, perf_diff=perf_diff,
        has_question=bool(question_path_found), has_human=bool(human_messages)))
    if consultation:
        # A question is not a development directive and cannot mutate the ledger.
        question_output = Path(consultation["directory"]) / "question.json"
        # Retain version-bound implementation/fusion/selection evidence that follows
        # the normal output lines; remove only the final-decision contract itself.
        prompt = "\n".join(line for line in prompt.splitlines()
                           if not line.startswith(("Stage9 输出文件：", "Stage9 请求编号：", "Stage9 当前轮次："))
                           and str(decision_path) not in line)
        prompt += (f"\nStage9 咨询输出文件：{question_output}\n"
                   f"Stage9 咨询请求编号：{consultation['request_id']}\n"
                   f"程序记录的三次停滞轮次：{consultation['trigger_iterations']}\n"
                   "本次只输出问题，禁止提交最终 decision 或改写 history。不要执行 Stage3。\n"
                   + file_hint(work_dir, consultation["evidence_manifest_path"], "咨询上下文的版本快照及缺失清单",
                               "按用途和读法查看实现、最佳成绩、慢 case 趋势及历史尝试，问题引用具体证据路径")
                   + "既有主动意见：" + _json.dumps(human_messages or [], ensure_ascii=False))
    request_record = {
        "iteration": state.iteration, "request_id": request_id, "fail_reason": fail_reason,
        "perf_diff": perf_diff, "question_path": question_path_found,
        "history_before_path": str(request_dir / "history_before.json"),
        "decision_path": str(decision_path), "allow_empty_suggestions": allow_empty,
        "scene": scene, "phase": phase, "human_message_ids": [m["id"] for m in human_messages or []],
        "performance_case_ids": performance_case_ids,
        "plan_version": 2 if not consultation else None,
        "known_case_ids": known_case_ids,
        "implementation_base": str(implementation_base),
        **contract_paths,
        "require_file_scope": not bool(consultation),
        "correction_attempt": 0, "attempt_number": 1,
        "max_attempts": 3, "max_corrections": 2,
        "default_inputs": default_inputs, "missing_inputs": missing_inputs,
        "role_path": role, "prompt_path": str(request_dir / "prompt.md"),
        "output_path": str(question_output if consultation else decision_path),
        "human_bundle_path": bundle_path,
        "consultation_id": consultation["request_id"] if consultation else state.stage9_context.get("consultation_id"),
        "semantic_event": state.stage9_context.get("semantic_event"),
        "regression_action": load_regression_action(work_dir, state.iteration) if performance_scene else None,
    }
    atomic_write_json(str(request_dir / "request.json"), request_record)
    atomic_write_text(str(request_dir / "prompt.md"), prompt)
    log.info("[Stage9 场景] iter=%s scene=%s phase=%s request=%s consultation=%s human_ids=%s；role=%s；prompt=%s；输入清单=%s",
             state.iteration, scene, phase, request_id,
             consultation["request_id"] if consultation else state.stage9_context.get("consultation_id", "-"),
             [item["id"] for item in human_messages or []], role, request_dir / "prompt.md", request_dir / "request.json")
    output_path = question_output if consultation else decision_path
    log.info("[%s] 启动 tech_lead，phase=%s，输出=%s", stage, phase, output_path)
    max_attempts = 3
    root_request_dir, root_request_id = request_dir, request_id
    base_prompt, base_output_path = prompt, decision_path
    base_output_hint, base_contract_hint = decision_output_hint, contract_hint
    base_role = Path(role).read_text(encoding="utf-8")
    protected_report = (validate_profile_report_directory(work_dir, state.iteration)
                        if performance_scene else None)
    report_before = (protected_report.read_text(encoding="utf-8")
                     if protected_report is not None and protected_report.is_file() else None)

    def restore_report_input():
        # Stage9 owns only decision.json. Keep Stage7's report (or the last
        # accepted Stage9 report) unchanged even if the agent edits it directly.
        if protected_report is None:
            return
        target = validate_profile_report_directory(work_dir, state.iteration)
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current == report_before:
            return
        if report_before is None:
            target.unlink(missing_ok=True)
        else:
            atomic_write_text(str(target), report_before)
        log.warning("[Stage9 报告保护] iter=%s 已撤销 agent 对分析报告的直接改写；仅接收 decision.json：%s",
                    state.iteration, target)

    try:
        for attempt in range(max_attempts):
            # Agent/transport failure is not a malformed decision. Preserve the
            # existing interruption path instead of silently launching it again.
            try:
                agent_ok = run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N4"])
            finally:
                restore_report_input()
            if not agent_ok:
                raise RuntimeError("Stage9 agent 未成功完成")
            if consultation:
                output = _json.loads(question_output.read_text(encoding="utf-8-sig"))
                question_status = consultation.get("context", {}).get("selection_status", selection_status)
                document = render_question(output, consultation, question_status, work_dir)
                save_history(work_dir, history_before)
                atomic_write_text(consultation["question_path"], document)
                return consultation["question_path"]
            try:
                with decision_path.open("r", encoding="utf-8-sig") as decision_file:
                    decision = _json.load(decision_file)
                if not isinstance(decision, dict) or decision.get("request_id") != request_id:
                    raise ValueError("Stage9 输出不属于本次请求")
                # These checks are read-only. A task-binding error must not hide
                # an independent directory/missing-file error until the retry.
                validation_errors = []
                try:
                    validate_stage9_request_conditions(
                        decision, perf_diff=perf_diff, performance_case_ids=performance_case_ids,
                        has_question=bool(question_path_found))
                except ValueError as exc:
                    validation_errors.extend(getattr(exc, "errors", [str(exc)]))
                _h = None
                try:
                    _h = merge_tech_lead_update(
                        history_before, decision, state.iteration, reason=fail_reason,
                        allow_empty_suggestions=allow_empty, human_messages=human_messages,
                        require_evaluation_summary=True, performance_case_ids=performance_case_ids,
                        require_file_scope=True, require_structured_plan=True,
                        known_case_ids=known_case_ids, validate_experience_fields=False)
                except ValueError as exc:
                    validation_errors.extend(getattr(exc, "errors", [str(exc)]))
                try:
                    if _h is None:
                        # Inspect safe raw siblings even if another task has a
                        # schema error; this never authorizes development.
                        file_targets = decision
                    else:
                        plan_ledger = next(entry for entry in reversed(_h["ledger"])
                                           if entry["iter"] == state.iteration)
                        file_targets = {"ledger_entry": plan_ledger, "suggest_next": _h["suggest_next"]}
                    _validate_stage9_file_targets(
                        work_dir, file_targets, implementation_base=implementation_base,
                        raw_plan=_h is None)
                except ValueError as exc:
                    validation_errors.extend(getattr(exc, "errors", [str(exc)]))
                if validation_errors:
                    raise Stage9DecisionValidationError(list(dict.fromkeys(validation_errors)))
            except (ValueError, FileNotFoundError) as exc:
                save_history(work_dir, history_before)
                error_path = request_dir / "validation_error.json"
                atomic_write_json(str(error_path), {
                    "iteration": state.iteration, "request_id": request_id,
                    "decision_path": str(decision_path), "correction_attempt": attempt,
                    "error": str(exc), "errors": getattr(exc, "errors", [str(exc)]),
                    "attempt_number": attempt + 1, "max_attempts": max_attempts,
                    "will_retry": attempt + 1 < max_attempts,
                })
                will_retry = attempt + 1 < max_attempts
                next_action = (f"同轮修正{attempt + 1}/2次，不增加性能迭代次数" if will_retry
                               else "停止于 Stage9，不进入 Stage3")
                failure_message = (
                    "===== Stage9 决策校验失败：iter=%s scene=%s request=%s 尝试次数=%s/3 修正次数=%s/2；"
                    "原因=%s；下一步=%s；决策=%s；错误记录=%s =====")
                failure_args = (state.iteration, scene, request_id, attempt + 1, attempt, exc,
                                next_action, decision_path, error_path)
                log.warning(failure_message, *failure_args)
                state_log.warning(failure_message, *failure_args)
                if not will_retry:
                    stopped_message = (
                        "===== Stage9 修正失败，停止交付 ===== iter=%s request=%s；不进入 Stage3；"
                        "原历史已恢复；原因=%s；决策=%s；错误记录=%s")
                    stopped_args = (state.iteration, request_id, exc, decision_path, error_path)
                    log.error(stopped_message, *stopped_args)
                    state_log.error(stopped_message, *stopped_args)
                    raise
                previous_id = request_id
                previous_output = decision_path
                request_dir = root_request_dir / f"retry{attempt + 1}"
                request_id = uuid.uuid4().hex
                decision_path = request_dir / "decision.json"
                output_path = decision_path
                role = str(request_dir / "n4_stage9_tech_lead_guide.md")
                # Keep the full original scene/human context. Replace only the
                # output contract, and provide the rejected version as evidence.
                new_output_hint = file_hint(
                    work_dir, decision_path, "本次 Stage9 决策输出（待你生成）",
                    "按 role 写 iteration、request_id、ledger_entry 和模型结论；经验使用 proven_pattern/regression_pattern/pitfall 公共字段")
                prompt = base_prompt.replace(base_output_hint, new_output_hint)
                contract_paths, new_contract_hint = _write_stage9_plan_contract(
                    work_dir, request_dir, state.iteration, request_id, performance_case_ids,
                    allow_empty, case_catalog, perf_diff=perf_diff, has_question=bool(question_path_found))
                prompt = prompt.replace(base_contract_hint, new_contract_hint)
                prompt = prompt.replace(str(base_output_path), str(decision_path)).replace(
                    f"Stage9 请求编号：{root_request_id}", f"Stage9 请求编号：{request_id}")
                prompt += (
                    f"\n===== 同轮 Stage9 决策修正（第 {attempt + 1}/2 次，总尝试第 {attempt + 2}/3 次）=====\n"
                    "上一份输出未通过程序检查，尚未交给 Stage3，也未写入经验。"
                    "请修正完整 decision，不要只输出补丁；保持当前轮次、原场景和全部人工意见。\n"
                    + file_hint(work_dir, error_path, "上次决策的校验错误", "逐项处理 errors 中的完整字段路径及 case ID；同时核对本轮完整 schema，不能只修第一条")
                    + file_hint(work_dir, previous_output, "被拒绝的原始决策（若存在）", "只读对照，不能直接复用；本次须写新的请求编号与输出文件")
                    + f"程序拒绝原因：{exc}\n"
                    "需要修改某文件时，必须写入 changes 的 modify/create 步骤，并与目标 case 的 implementation_files 对应，"
                    "且不能在 ledger.readonly_files 中；允许修改列表由程序汇总，不要自行填写 modify_files/inspect_files。"
                    "若文件必须只读，应调整方案或安排 inspect 任务，不得将修改伪装成检查。\n"
                    "文件检查修正：changes.file 中 inspect 也必须填写实际文件，不能填 prof_data 等目录。"
                    "可先浏览目录定位对应报告再列出路径；case_scope=cases 的 inspect 任务须逐 case 包含对绑定实现文件的检查。"
                    "按 errors 列表同时修正任务绑定和文件类型问题，不猜测不存在的报告路径。\n"
                )
                if any(perf_diff.get(flag) for flag in ("has_improvement", "has_regression")):
                    prompt += (
                        "性能经验修正：按本次 schema/template 核对 proven_pattern/regression_pattern 的全部必填字段。"
                        "applicability 必须是非空文字；每条 case_analysis 都要有独立 explanation。"
                        "ledger_entry.case_analysis 的解释和总体 why_it_worked/why_it_failed 不能替代经验里的逐 case 解释。"
                        "根因不确定时如实标为推测或待验证，并说明验证方法；不得编造原因或删除 case 来绕过检查。\n"
                    )
                request_record.update({
                    "request_id": request_id, "decision_path": str(decision_path),
                    "output_path": str(decision_path), "role_path": role,
                    "prompt_path": str(request_dir / "prompt.md"),
                    "history_before_path": str(request_dir / "history_before.json"),
                    "correction_attempt": attempt + 1, "attempt_number": attempt + 2,
                    "retry_of_request_id": previous_id,
                    "rejected_decision_path": str(previous_output),
                    "validation_error_path": str(error_path),
                    **contract_paths,
                })
                atomic_write_json(str(request_dir / "history_before.json"), history_before)
                atomic_write_json(str(request_dir / "request.json"), request_record)
                atomic_write_text(role, base_role)
                atomic_write_text(str(request_dir / "prompt.md"), prompt)
                retry_message = (
                    "===== Stage9 同轮修正 ===== iter=%s scene=%s；尝试次数=%s/3 修正次数=%s/2，不增加性能迭代次数；"
                    "原请求=%s；新请求=%s；错误记录=%s；原决策=%s；prompt=%s；输入清单=%s；输出=%s")
                retry_args = (state.iteration, scene, attempt + 2, attempt + 1, previous_id, request_id, error_path,
                              previous_output, request_dir / "prompt.md", request_dir / "request.json", decision_path)
                log.info(retry_message, *retry_args)
                state_log.info(retry_message, *retry_args)
                continue
            log.info("[Stage9 决策校验通过] iter=%s request=%s 尝试次数=%s/3 修正次数=%s/2；v2任务=%s；允许修改=%s；只读=%s；决策=%s",
                     state.iteration, request_id, attempt + 1, attempt, [item["task_id"] for item in _h["suggest_next"]],
                     plan_ledger["modify_files"], plan_ledger["readonly_files"], decision_path)
            state_log.info("[Stage9 决策校验通过] iter=%s request=%s 尝试次数=%s/3 修正次数=%s/2；决策=%s",
                           state.iteration, request_id, attempt + 1, attempt, decision_path)
            break
    except BaseException as exc:
        # Restore the full pre-call state even if the agent accidentally overwrote it.
        save_history(work_dir, history_before)
        log.error("[%s] Stage9 输出未通过接收检查，历史已恢复；phase=%s，本次输出=%s", stage, phase, output_path)
        if isinstance(exc, Exception):
            raise RuntimeError(f"Stage9 决策接收失败：{exc}") from exc
        raise

    # The agent owns only decision.json. Restore accidental edits before any further IO.
    save_history(work_dir, history_before)
    if scene == "all_passed":
        # Detect competing reports before publishing history/knowledge. The
        # Markdown itself is derived only after the decision is committed.
        try:
            validate_profile_report_directory(work_dir, state.iteration)
        except (OSError, ValueError) as exc:
            log.error("[性能分析报告] iter=%s 来源=Stage9；目录检查失败，不交付：%s；决策=%s",
                      state.iteration, exc, decision_path)
            raise RuntimeError(f"Stage9 分析报告目录不可用：{exc}") from exc
    current_ledger = next(entry for entry in reversed(_h["ledger"]) if entry["iter"] == state.iteration)
    current_ledger["stage9_decision_path"] = str(decision_path)
    review_environment = build_knowledge_environment(work_dir, comparison_context=selection_context)
    performance_environment = build_knowledge_environment(
        work_dir, performance_report=perf_diff.get("current_report") or
        Path(work_dir) / "eval" / f"iter{state.iteration}" / "perf_result.json") if performance_scene else None
    current_ledger["environment"] = copy.deepcopy(performance_environment or review_environment)

    # Shared provenance and measured numbers are supplied by the program, never the model.
    evidence_files = {}
    for label, measured_iteration in (("previous", perf_diff.get("prev_iter")),
                                      ("current", state.iteration)):
        if measured_iteration is None:
            continue
        performance_path = Path(perf_diff.get(f"{label}_report") or
                                Path(work_dir) / "eval" / f"iter{measured_iteration}" / "perf_result.json")
        if performance_path.is_file():
            evidence_files[f"{label}_performance_result"] = str(performance_path)
            try:
                measured = _json.loads(performance_path.read_text(encoding="utf-8"))
                if measured.get("source_json"):
                    evidence_files[f"{label}_performance_report"] = measured["source_json"]
            except (OSError, ValueError, AttributeError):
                pass
    # Performance knowledge refers to the measured immutable revision even if
    # a late human re-review occurs after the best baseline has been restored.
    current_evidence = ((selection_status.get("current") or {}) if performance_scene
                        and selection_status.get("latest_iteration") == state.iteration
                        else load_evidence(work_dir))
    for name, path in current_evidence.get("evidence_paths", {}).items():
        evidence_files[f"current_{name}"] = path

    for flag, field, writer, filename, label in (
        ("has_improvement", "proven_pattern", append_proven_pattern, "proven_patterns.md", "成功经验"),
        ("has_regression", "regression_pattern", append_regression_pattern, "regression_patterns.md", "退步教训"),
    ):
        if not perf_diff.get(flag):
            continue
        pattern = copy.deepcopy(decision[field])
        pattern.update({
            "speedup_before": perf_diff["prev_avg_speedup"],
            "speedup_after": perf_diff["curr_avg_speedup"], "delta_pct": perf_diff["delta_pct"],
            "prev_iter": perf_diff["prev_iter"], "curr_iter": state.iteration,
            "case_diffs": perf_diff.get("case_diffs", []),
            "evidence_files": evidence_files, "decision_path": str(decision_path),
            "environment": copy.deepcopy(performance_environment),
        })
        writer(work_dir, state.iteration, pattern)
        log_knowledge_write(log, state_log, label=label, path=Path(work_dir) / "knowledge" / filename,
                            iteration=state.iteration, environment=performance_environment,
                            decision_path=decision_path,
                            detail=f"iter{perf_diff['prev_iter']}→iter{state.iteration}；"
                                   f"avg_speedup={perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']}；"
                                   f"变化={perf_diff['delta_pct']}%")

    if question_path_found:
        pitfall = copy.deepcopy(decision["pitfall"])
        pitfall["question_path"] = question_path_found
        pitfall["decision_path"] = str(decision_path)
        pitfall["environment"] = copy.deepcopy(review_environment)
        append_pitfall(work_dir, state.iteration, pitfall)
        verdict = pitfall["verdict"]
        question_path = Path(question_path_found)
        question_text = question_path.read_text(encoding="utf-8")
        if "## tech_lead 裁定" not in question_text:
            question_text += (
                f"\n\n## tech_lead 裁定（iter{state.iteration}）\n"
                f"- 结论: {'✅确认误判' if verdict == 'confirmed' else '❌驳回'}\n"
                f"- 原因: {pitfall['root_cause']}\n"
                f"- 正确{'做法' if verdict == 'confirmed' else '认知'}: {pitfall['correct_approach']}\n")
            atomic_write_text(str(question_path), question_text)
        append_work_record(work_dir, f"{stage} 裁定 question.md: {verdict}")
        log_knowledge_write(log, state_log, label="错题本", path=Path(work_dir) / "knowledge/tech_lead_pitfalls.md",
                            iteration=state.iteration, environment=review_environment, decision_path=decision_path,
                            detail=f"裁定={verdict}；主题={pitfall['topic']}；问题={question_path_found}")

    save_history(work_dir, _h)
    atomic_write_json(str(request_dir / "commit.json"), {
        "iteration": state.iteration, "request_id": request_id,
        "decision_path": str(decision_path), "history_path": history_path, "committed": True,
    })
    if scene == "all_passed":
        try:
            report_path = write_stage9_report(work_dir, state.iteration, decision, decision_path)
        except (OSError, ValueError) as exc:
            log.error("[性能分析报告] iter=%s 来源=Stage9；已接收决策，但报告保存失败：%s；决策=%s",
                      state.iteration, exc, decision_path)
            raise RuntimeError(f"Stage9 分析报告保存失败：{exc}") from exc
        _log_profile_report(log, state_log, state.iteration, 9, report_path, decision_path)
    if performance_scene:
        log.info("[历史分析] iter=%s 已保存 %s 个慢 case 的本轮结论，case_ids=%s；history=%s；decision=%s",
                 state.iteration, len(current_ledger.get("case_analysis", [])), performance_case_ids,
                 history_path, decision_path)
    log_knowledge_write(log, state_log, label="历史账本", path=history_path, iteration=state.iteration,
                        environment=current_ledger["environment"], decision_path=decision_path,
                        detail=f"程序已合并本轮账本，保留 {len(_h['ledger'])} 轮 ledger")
    if human_messages:
        log.info("[人工裁定] iter=%s request=%s responses=%s P0_ids=%s；decision=%s；history=%s",
                 state.iteration, request_id,
                 [{"id": item["message_id"], "kind": item["kind"]} for item in decision["human_responses"]],
                 [item["human_message_id"] for item in decision["suggest_next"] if item.get("source") == "human"],
                 decision_path, history_path)

    history_log = logging.getLogger("triton-ascend-history")
    if history_log.handlers:
        history_log.info(f"ITER{state.iteration} [stage9 tech_lead 更新({fail_reason})]\n{_json.dumps(_h, ensure_ascii=False, indent=2)}")
    log.info(f"[{stage}] history.json 已由程序合并 Stage9 决策")
    return str(decision_path)


def _log_profile_report(log, state_log, iteration, source_stage, path, decision_path=None):
    message = "[性能分析报告] iter=%s 来源=Stage%s；本轮唯一报告=%s；决策来源=%s"
    args = (iteration, source_stage, path, decision_path or "Stage7 分析正文")
    log.info(message, *args)
    if state_log is not None:
        state_log.info(message, *args)


def _restore_stage9_profile(work_dir, iteration, ledger):
    """Rebuild a report from the accepted decision, never the live implementation."""
    root = Path(work_dir).resolve()
    source = ledger.get("stage9_decision_path")
    if not source:
        raise ValueError("当前账本缺少已接收的 Stage9 决策路径，不能补造分析报告")
    decision_path = Path(source)
    if not decision_path.is_absolute():
        decision_path = root / decision_path
    if (not decision_path.resolve().is_relative_to(root / "knowledge" / "stage9" / f"iter{iteration}")
            or decision_path.name != "decision.json"):
        raise ValueError("分析报告来源必须是本轮已接收的 Stage9 decision.json")
    decision = _json_module.loads(decision_path.read_text(encoding="utf-8-sig"))
    commit = _json_module.loads((decision_path.parent / "commit.json").read_text(encoding="utf-8-sig"))
    if (not isinstance(commit, dict) or not isinstance(decision, dict)
            or commit.get("committed") is not True or commit.get("iteration") != iteration
            or decision.get("iteration") != iteration or ledger.get("iter") != iteration
            or not decision.get("request_id") or commit.get("request_id") != decision["request_id"]
            or commit.get("decision_path") != str(decision_path)):
        raise ValueError("分析报告来源的 Stage9 提交记录与当前轮次/请求不一致")
    authored = decision.get("ledger_entry")
    if not isinstance(authored, dict) or any(
            authored.get(field) != ledger.get(field)
            for field in ("evaluation_summary", "case_analysis")):
        raise ValueError("分析报告来源与已接收账本不一致，不能混用版本")
    # Render the accepted task view checked by current_scope, not an unrelated
    # live suggestion list or source files that Stage3 may already have edited.
    decision = dict(decision, suggest_next=ledger["action_plan"]["tasks"])
    return write_stage9_report(work_dir, iteration, decision, decision_path)


def self_goto_stage3(log, roles_dir: str, work_dir: str, op_name: str, state: State, reason: str, state_log=None, node_logs=None, iter_dirs=None, extra_context: str = "", device_info_prompt: str = "", selection_context=None):
    """
    调用阶段3（cannbot修改），根据 reason 注入不同文件路径到 prompt。
    reason: build_fail / precision_fail / perf_optimize / score_zero
    """
    stage = f"ITER{state.iteration}-阶段3-修改({reason})"
    role = os.path.join(roles_dir, "n1_stage3_fix_and_optimize.md")
    _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                           node_logs, iter_dirs, device_info_prompt, selection_context)
    # Stage3 may be resumed directly from an old or edited history. Recheck the
    # current handoff instead of letting that route bypass Stage9 validation.
    def current_scope():
        history = load_history(work_dir)
        ledger = next((item for item in reversed(history.get("ledger", []))
                       if item.get("iter") == state.iteration), None)
        suggestions = history.get("suggest_next")
        if not suggestions:
            raise ValueError("Stage3 本轮缺少可执行建议")
        scene = state.stage9_context.get("scene") or classify_scene(reason, iteration=state.iteration)
        known_ids = _stage9_case_catalog(work_dir, state.iteration, scene)["case_ids"]
        ledger, suggestions = validate_current_plan(ledger, suggestions, known_case_ids=known_ids)
        _validate_stage9_file_targets(
            work_dir, {"ledger_entry": ledger, "suggest_next": suggestions},
            allow_existing_creates=state.stage9_context.get("phase") == "developing",
            implementation_base=_stage9_implementation_base(work_dir, state.iteration))
        return ledger, suggestions

    try:
        current_scope()
    except ValueError as exc:
        log.warning("[Stage3 交付拦截] iter=%s 文件范围未通过检查，返回同轮 Stage9：%s；history=%s",
                    state.iteration, exc, Path(work_dir) / "knowledge/history.json")
        state.stage9_context["scope_review_error"] = str(exc)
        state.flush()
        run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs,
                      fail_reason=state.stage9_context.get("fail_reason") or reason,
                      device_info_prompt=device_info_prompt,
                      perf_diff=state.stage9_context.get("perf_diff"), selection_context=selection_context)
        current_scope()
    state.stage9_context.pop("scope_review_error", None)
    apply_regression_action(work_dir, state.iteration,
                            state.stage9_context.get("decision_path"), log, state_log,
                            comparison_context=selection_context)
    # The validated plan must also be applicable to the actual post-restore
    # implementation. Never carry old case/file evidence blindly across a swap.
    current_plan_ledger, current_tasks = current_scope()
    if reason == "perf_pass_optimize":
        report_path = profile_report_path(work_dir, state.iteration)
        missing_report = not report_path.is_file()
        report_path = _restore_stage9_profile(work_dir, state.iteration, current_plan_ledger)
        if missing_report:
            _log_profile_report(log, state_log, state.iteration, 9, report_path,
                                current_plan_ledger["stage9_decision_path"])
    elif reason == "perf_optimize":
        stamp_stage7_report(work_dir, state.iteration)
    human = HumanReview(work_dir)
    ids = set(state.stage9_context.get("message_ids", []))
    human_directions = [item for item in human.all_messages()
                        if item["id"] in ids and item["kind"] == "direction"
                        and item.get("status") != "executed"]
    receipt_path = Path(iter_dirs["develop"]) / "human_feedback.json"
    state.stage9_context.update(phase="developing", development_reason=reason)

    # 备份当前 impl 到 operator_iter/iter{N}/（代码追溯用，不对 agent 展示）
    import shutil
    backup_dir = os.path.join(work_dir, "operator_iter", f"iter{state.iteration}")
    impl_dir = os.path.join(work_dir, "impl")
    if os.path.isdir(impl_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(impl_dir, backup_dir)
        log.info(f"[{stage}] impl 已备份到 {backup_dir}")

    append_work_record(work_dir, stage)
    transition(state, "stage3", state_log, reason=f"回退修改代码({reason})")

    base_prompt = (
        f"工作目录：{work_dir}\n算子：{op_name}\n"
        + file_hint(work_dir, Path(work_dir) / 'impl', "当前算子实现工程",
                    "按 Tech Lead 的允许范围检查实际源文件，定位入口、tiling 和数据流")
        + file_hint(work_dir, Path(work_dir) / 'task', "只读需求、用例及参考实现",
                    "按 desc.md、proto.yaml、cases.yaml、golden.py 核对修复是否保持语义和完整覆盖")
        + file_hint(work_dir, Path(work_dir) / 'ANALYSIS.md', "Stage1 需求分析",
                    "优先核对接口、精度、shape 和硬件约束，再判断优化方向")
        + f"{device_info_prompt}\n回退原因：{reason}\n"
    )
    base_prompt += format_fusion_library_for_prompt(work_dir, "stage3")
    base_prompt += format_evidence_for_prompt(work_dir)
    base_prompt += format_selection_for_prompt(work_dir, comparison_context=selection_context)
    base_prompt += regression_action_prompt(work_dir, state.iteration)
    base_prompt += development_prompt(work_dir, iter_dirs["develop"], 3)
    base_prompt += development_human_prompt(work_dir, human_directions, receipt_path)
    if current_plan_ledger.get("stage9_decision_path"):
        base_prompt += file_hint(
            work_dir, current_plan_ledger["stage9_decision_path"], "Stage9 本轮已校验的 v2 任务单原文",
            "按 task_id 查看目标 case、对应代码依据、changes 的逐文件方法及验收要求；下方可读任务单由同一份 JSON 生成")
    log.info("[Stage3 任务交接] iter=%s tasks=%s；可修改=%s；只读=%s；来源=%s",
             state.iteration, [task["task_id"] for task in current_tasks],
             current_plan_ledger["modify_files"], current_plan_ledger["readonly_files"],
             current_plan_ledger.get("stage9_decision_path"))

    # 注入错题本（可能不存在，不存在则为空）
    pitfalls_text = format_pitfalls_for_prompt(work_dir)
    if pitfalls_text:
        base_prompt += f"\n{pitfalls_text}\n"

    # question.md 反馈通道说明（仅在确认 tech_lead 指导有误时才写）
    base_prompt += (
        f"\n📮 如实施中发现 tech_lead 建议在硬件/框架层面确实不可行（有硬证据），"
        f"可写反馈到 {iter_dirs['develop']}/question.md（严格触发条件见你的 role，"
        f"先查错题本避免重复提；这不是拒绝执行的理由，仍需用替代方案完成本轮任务）。\n"
    )

    # 从 ledger 最后一条提取 tech_lead 的文件约束指令
    fix_plan_text = get_latest_fix_plan(work_dir, current_iteration=state.iteration)
    if fix_plan_text:
        base_prompt += f"\n{fix_plan_text}\n\n"

    if reason == "build_fail":
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['build']) / 'build.log', "本轮编译日志",
                      "先查首个真实编译/安装错误及文件行号，避免只修后续连带错误") +
            f"请根据编译错误修复代码。\n"
            f"自测报告输出到 {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== 历史经验 ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['build']}/build.log"]
    elif reason == "precision_fail":
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['eval']) / 'precision_result.json', "本轮正式精度结果",
                      "先看失败 case、错误摘要和通过数量")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'precision_reports', "本轮精度原始详情",
                        "按失败 case 查误差和原始报错，与 golden.py 对照定位语义问题") +
            f"请根据精度失败信息修复代码。\n"
            f"自测报告输出到 {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== 历史经验 ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/precision_result.json"]
    elif reason == "score_zero":
        # 反作弊/零分：注入 history 经验 + score_error 信息，让 cannbot 知道问题根因
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            f"⚠️ cann-bench 性能结果异常或零分，请先核对真实原因；缺报告不等于已证实违规。\n"
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "零分/反作弊性能结果",
                        "先看 score_error_code 和 kernel 事件证据，按反作弊规则定位原因") +
            f"{extra_context}\n"
            f"若证据显示没有有效的 NPU kernel 事件，可排查以下原因；采集/报告问题先修评测链路：\n"
            f"1. 算子实际走了 CPU fallback 而非 NPU 执行\n"
            f"2. impl/build/ 目录残留旧的构建产物，或环境中安装的 cann_bench 包不是本轮版本\n"
            f"3. Triton JIT 没有正确触发（检查 @triton.jit、kernel[grid](...) 调用和 import 路径）\n"
            f"请彻底排查并修复，确保算子在 NPU 上真正执行。\n"
            f"若确认有旧构建产物，先清理本工程 impl/build/，然后在 impl/ 中执行 "
            f"python3 -m pip install . --force-reinstall --no-deps；"
            f"回到 work 目录验证 cann_bench 的导入位置，确保加载本轮安装的包。\n"
            f"设计思路文档输出到 {iter_dirs['develop']}/design_rationale.md\n"
            f"自测报告输出到 {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== 历史经验（来自之前迭代） ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/perf_result.json"]
    elif reason == "perf_pass_optimize":
        # 已达标，但程序的 x 次有效性能窗口尚未满足退出条件。
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            f"性能已达标，继续依据瓶颈和 Tech Lead 的 P0/P1/P2 意见优化。\n"
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "本轮正式性能汇总",
                        "逐 case 核对达标和剩余瓶颈，保持全部 case 不退步")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "本轮原始性能报告",
                        "按 case 回查真实计时和 kernel 证据，以新评测确认优化收益")
            + file_hint(work_dir, profile_report_path(work_dir, state.iteration), "Stage9 本轮瓶颈分析报告",
                        "先核对报告来源和轮次，再看逐 case 的现象、原因、证据及下一步；执行范围仍以已接收任务单为准") +
            f"设计思路文档输出到 {iter_dirs['develop']}/design_rationale.md\n"
            f"自测报告输出到 {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== 历史经验（来自 tech_lead） ===\n{history_context}\n\n"
            f"在保证正确性和所有 case 达标的基础上提高整体性能；可保留当前融合方案。\n"
            f"请说明本轮数据流及选择理由，以新评测检验收益，不以 kernel 数量决定优劣。"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/perf_result.json",
                  str(profile_report_path(work_dir, state.iteration))]
    else:  # perf_optimize
        # 注入跨轮记忆（insights/ledger/趋势）
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "本轮正式性能汇总",
                      "先看 avg_speedup、逐 case speedup 与 worst_6_cases，明确优化目标")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "本轮原始性能报告",
                        "按慢 case 核对原始计时和 kernel/profiler 证据")
            + file_hint(work_dir, Path(iter_dirs['profile']) / 'bottleneck_analysis.md', "Stage7 瓶颈分析",
                        "先看逐 case 根因与共性瓶颈，区分实测结论和假设")
            + file_hint(work_dir, Path(iter_dirs['search']) / 'SEARCH_REPORT.md', "Stage8 搜索方案与来源",
                        "核对官方来源、硬件前提、实现路径和风险")
            + file_hint(work_dir, Path(iter_dirs['search']) / 'FIX_DIRECTIVE.md', "Stage8 具体修改指令",
                        "结合 Tech Lead 最新 P0/P1/P2 及文件范围实施，冲突时以 Tech Lead 裁定为准") +
            f"设计思路文档输出到 {iter_dirs['develop']}/design_rationale.md\n"
            f"自测报告输出到 {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== 历史经验（来自 tech_lead，必须遵守） ===\n{history_context}\n\n"
            f"请按修改指令优化代码，重点关注 worst_6_cases 的瓶颈。\n"
            f"账本 verdict 只评价已经测过的实现；direction 是评测后提出的下一步计划。"
            f"不得因均值持平或退步就禁止新计划；结合逐 case 趋势、条件和证据执行最新建议。"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['search']}/FIX_DIRECTIVE.md", f"{iter_dirs['profile']}/bottleneck_analysis.md"]

    if Path(fusion_library_path(work_dir)).is_file():
        inputs.append(str(fusion_library_path(work_dir)))
    outputs = [f"{work_dir}/impl/"]
    log_io(log, stage, inputs, outputs)

    prompt += CANNBOT_CONSTRAINT
    if human_directions and receipt_path.exists():
        receipt_path.unlink()  # Only this generated receipt; never accept an old acknowledgement.
    previous_revisions = begin_development(work_dir, iter_dirs["develop"])
    ok = run_agent("cannbot", role, work_dir, prompt, node_log=node_logs["N1"])
    evidence = finalize_development(work_dir, iter_dirs["develop"], 3, agent_ok=ok,
                                    previous_revisions=previous_revisions)
    if not evidence.get("eligible"):
        log.warning(f"[{stage}] 方案/自测材料尚未满足最佳记录条件：{evidence.get('reason', '')}")
    if not ok:
        log.error(f"[{stage}] cannbot 失败（returncode!=0）")
        if human_directions:
            raise RuntimeError("Stage3 失败，人工 P0 尚未执行；保留原轮次供恢复")
    if human_directions:
        try:
            receipts = validate_execution_receipt(receipt_path, human_directions)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Stage3 缺少本次人工意见落实回执；保留原轮次供恢复") from exc
        for receipt in receipts:
            rationale_path = Path(iter_dirs["develop"]) / "融合方案选择决策依据.md"
            if receipt["message_id"] not in read_file_safe(str(rationale_path)):
                raise RuntimeError("Stage3 方案选择依据未说明对应人工意见编号；保留原轮次供恢复")
            if receipt["status"] == "implemented" and evidence.get("eligible"):
                human.mark_executed([receipt["message_id"]], state.iteration, str(receipt_path))
            else:
                human.mark_unexecuted([receipt["message_id"]], receipt["details"] if receipt["status"] != "implemented"
                                      else "实现自报已修改，但本轮代码绑定/自测证据不完整，尚未确认执行")
        log.info("[人工 P0] 本轮执行回执=%s；实际性能收益仍以之后正式评测为准", receipt_path)
    state.stage9_context["phase"] = "delivered"
    state.flush()


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()

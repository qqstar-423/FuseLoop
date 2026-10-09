#!/usr/bin/env python3
"""
Triton Ascend Operator Workflow — deterministic control plane
========================================
Single launch entry point. Python if/else + while controls routing; stage1.5 calls Jev for scoring.
Stage switching is fully controlled by this program; agents only execute and never decide workflow flow.

Usage:
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
            print(f"[chip detection] attempt {attempt} failed: {result.stderr[-300:]}", file=sys.stderr)
        except Exception as exc:
            print(f"[chip detection] attempt {attempt} raised: {exc}", file=sys.stderr)
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
            print(f"[chip detection] tbe query failed: {result.stderr[-300:]}", file=sys.stderr)
    except Exception as exc:
        print(f"[chip detection] tbe query error: {exc}", file=sys.stderr)
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
        "🔧 Current Ascend NPU and Triton Ascend environment (shared by all stages)\n"
        f"  Chip model: {device_info['chip_model']}\n"
        f"  SoC version: {device_info['soc_version']}\n"
        f"  NPU architecture: {device_info['npu_arch']}\n"
        f"  Framework: {FRAMEWORK}; backend: {BACKEND}; runtime versions: {_json_module.dumps(versions, ensure_ascii=False)}\n"
        f"  Programming model: {device_info.get('programming_model', PROGRAMMING_MODEL)}\n"
        f"  Backend target: {device_info.get('driver_backend', 'unknown')}/{device_info.get('target_arch', 'unknown')}\n"
        f"  AI Core count: {device_info['ai_core_num']} (Cube {device_info.get('cube_core_num', '?')} + Vector {device_info.get('vector_core_num', '?')})\n"
        f"  UB: {device_info['ub_size_kb']} KB; L1: {device_info['l1_size_kb']} KB\n"
        f"  L0A/L0B/L0C: {device_info.get('l0a_size_kb', '?')}/{device_info.get('l0b_size_kb', '?')}/{device_info.get('l0c_size_kb', '?')} KB\n"
        f"  L2 Cache: {device_info.get('l2_size_kb', '?')} KB\n"
        f"  Device ID: {device_info['device_id']} (logical index in the currently visible device mapping)\n"
        f"  Detection method: {device_info['detect_method']}\n"
        "  Use import triton, import triton.language as tl, @triton.jit and an explicit grid.\n"
        "  BLOCK tiling, mask, stride and matrix computation parameters must be chosen against these resources and local backend support; on-chip buffers are managed by the compiler.\n"
        "  Do not copy CUDA warp, shared-memory, PTX or GPU-specific async instruction assumptions; defer to the installed triton-ascend API.\n"
        + file_hint(Path(__file__).parent, Path(__file__).parent / "knowledge/arch_programming_guide.md",
                    "Triton Ascend architecture programming guide", "Verify local API, tiling, boundary masks and memory access constraints",
                    base_label="project root")
        + "  example/ samples are only for interface and basic writing reference; tiling parameters still need verification against the current operator and chip.\n"
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
    """Print the stage input/output file paths"""
    log.info(f"[{stage}] Inputs: {', '.join(inputs)}")
    log.info(f"[{stage}] Expected outputs: {', '.join(outputs)}")


def fusion_review_hint(status: dict, iteration: int) -> str:
    """Describe only this iteration's valid underperforming stagnation window."""
    if not (status.get("eligible") and status.get("review_fusion")
            and status.get("latest_iteration") == iteration):
        return ""
    window = status["window"]
    return (
        f"[Scenario 2 stagnation: focus on reviewing the fusion scheme] Some cases still have speedup < 1; "
        f"y={window['required_improvements']} valid performance iterations have completed in a row (baseline excluded), "
        f"window iter{window['start_iteration']}→iter{window['end_iteration']}, "
        f"historical best avg_speedup {window['start_best_avg_speedup']:.6g}→"
        f"{window['end_best_avg_speedup']:.6g}, cumulative valid improvement "
        f"{window['cumulative_improvement']:.2%} < {window['threshold']:.0%}.\n"
        "Focus on comparing the JSON fusion candidate library and probabilities, the current fusion selection rationale, the bottleneck analysis, search results and trends of underperforming cases; "
        "dynamically decide to keep, partially optimize or replace the scheme, and justify it within the original P0/P1/P2 suggestions."
        "If a few slow cases keep improving, continue partial optimization; do not force a scheme change merely because average improvement is under 5%."
    )


def transition(state, new_stage: str, state_log, reason: str = ""):
    """Switch state + write log (including the reason)."""
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
            raise RuntimeError("The CANN toolchain was previously verified, but a re-checkable install directory is missing; further performance comparison is forbidden.")
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
                    "reason": "Code version or evaluation protocol of precision, self-test and performance is not consistently bound; this round is excluded from the best record and stagnation window."}
    return record_evaluation(work_dir, iteration, perf=perf, precision=precision,
                             evidence=evidence, comparison_context=context, config=config)


# cannbot global constraints (appended to the end of every cannbot prompt)
CANNBOT_CONSTRAINT = (
    "\n⚠️ Constraint: all temporary files and debug scripts must go under /tmp; creating temporary files in the working directory or project root is forbidden.\n"
    + file_hint(Path(__file__).parent, Path(__file__).parent / "knowledge/anti_cheat_reference.md",
                "Anti-cheating reference", "After development, check execution paths and evaluation behavior against the actual error handling table and the self-check list", base_label="project root")
)


def allocate_work_directory(base_dir, op_name, timestamp):
    """Reserve a new run directory atomically, even for simultaneous launches."""
    if not op_name or op_name in {".", ".."} or any(char in op_name for char in "/\\"):
        raise ValueError("Operator name must be a single directory name and must not contain path separators")
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
    parser = argparse.ArgumentParser(description="Triton Ascend Operator Workflow control plane")
    parser.add_argument("--task-dir", required=True, help="Path to the cannbench task directory")
    parser.add_argument("--op-name", default=None, help="Operator name")
    parser.add_argument("--max-iter", type=int, default=None, help="Maximum number of iterations")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--work-dir", default=None, help="Working directory")
    parser.add_argument("--init-impl", default=None, help="Emergency import of an existing implementation and its development material; creates a new work directory, skips Stage1/1.5/2, and evaluates directly from Stage4")
    hint_source = parser.add_mutually_exclusive_group()
    hint_source.add_argument("--optimize-hint", default=None,
                             help="Injected into Stage1/2 for new tasks; saved as an optimization reference for emergency imports; use --optimize-hint-file for long text")
    hint_source.add_argument("--optimize-hint-file", default=None,
                             help="Reads a UTF-8 direction file in full; mutually exclusive with --optimize-hint; emergency import still evaluates first and does not modify code in advance")
    args = parser.parse_args()
    if args.optimize_hint_file:
        try:
            args.optimize_hint = Path(args.optimize_hint_file).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            parser.error(f"--optimize-hint-file cannot read UTF-8 file ({type(exc).__name__}): {args.optimize_hint_file}")

    if args.init_impl and args.work_dir:
        parser.error("--init-impl and --work-dir cannot be used together: the former creates an emergency task, the latter resumes an existing one")
    if args.init_impl:
        if not Path(args.init_impl).is_dir():
            parser.error(f"--init-impl directory does not exist: {args.init_impl}")
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
    log.info("Triton Ascend Operator Workflow started")
    log.info(f"Operator: {op_name} | Task: {task_dir}")
    log.info(f"Work: {work_dir} | max iterations: {max_iterations}")
    if args.init_impl:
        log.info("===== Emergency import of existing implementation ===== source=%s; new work=%s", args.init_impl, work_dir)
    elif optimize_hint:
        log.info(f"💡 Fusion direction hint: {optimize_hint[:200]}...")
    log.info("=" * 60)

    imported = (prepare_init_impl(args.init_impl, work_dir, task_dir,
                                 optimize_hint=optimize_hint, log=log)
                if args.init_impl else load_init_impl_manifest(work_dir))
    if imported:
        log.info("[emergency import] Source and copy manifest: %s", Path(work_dir) / "init_impl_manifest.json")
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
        log.info('[human entry] In another terminal run python tools/human_review.py --work-dir "%s" --message "your direction"; add --kind question for a pure question', work_dir)
    atexit.register(state.flush)
    signal.signal(signal.SIGTERM, lambda s, f: (state.flush(), sys.exit(1)))
    signal.signal(signal.SIGINT, lambda s, f: (state.flush(), sys.exit(1)))

    # Re-probe on every startup, including resume; an old device_info.json
    # is historical evidence, not proof that today's device is the same chip.
    device_info_path = os.path.join(work_dir, "device_info.json")
    if imported:
        log.info("[chip recheck] Keeping the imported scoring hardware file; check whether current hardware and runtime match; development nodes are not invoked")
    elif os.path.exists(device_info_path):
        log.info("[chip recheck] Checkpoint resume re-verifies the current chip; the old device_info.json is not the hardware source for this run")
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
        log.info(f"Chip info comes from manual config.yaml settings: {device_info['chip_model']}")
    else:
        device_info = detect_npu_device(device_id=hw_config.get("device_id", 0), config_path=args.config)
        device_info["device_id"] = hw_config.get("device_id", 0)

    # ── Detection validation: all key parameters must be obtained, otherwise exit ──
    missing = []
    if device_info.get("soc_version", "unknown") == "unknown":
        missing.append("soc_version (chip model unrecognized; torch_npu and environment variables both failed)")
    if device_info.get("detect_method") == "none" or device_info.get("ub_size_kb", 0) == 0:
        missing.append("UB_SIZE (tbe query failed; CANN Toolkit may not be installed correctly)")
    if device_info.get("l1_size_kb", 0) == 0:
        missing.append("L1_SIZE")
    if device_info.get("ai_core_num", 0) == 0:
        missing.append("CORE_NUM")
    if device_info.get("l0a_size_kb", 0) == 0:
        missing.append("L0A_SIZE")

    if missing:
        log.error("=" * 60)
        log.error("❌ NPU chip parameter detection failed; the workflow cannot start")
        log.error(f"Missing parameters: {', '.join(missing)}")
        log.error(f"Information obtained: {_json_module.dumps(device_info, ensure_ascii=False, indent=2)}")
        log.error("")
        log.error("Troubleshooting steps:")
        log.error("  1. Confirm the NPU device is available: python3 -c \"import torch,torch_npu; print(torch_npu.npu.get_device_properties(0).name)\"")
        log.error("  2. Confirm the CANN Toolkit is fully installed: python3 -c \"from tbe.common.platform import get_soc_spec; print('OK')\"")
        log.error("  3. Or fill in all chip parameters manually in the hardware section of config.yaml; a compatible Triton Ascend backend is still required")
        log.error("=" * 60)
        sys.exit(1)

    log.info(f"Chip detection passed: {device_info['chip_model']} (soc={device_info['soc_version']}, "
             f"cores={device_info['ai_core_num']}, UB={device_info['ub_size_kb']}KB, "
             f"L1={device_info['l1_size_kb']}KB, L0C={device_info.get('l0c_size_kb', '?')}KB, "
             f"method={device_info['detect_method']})")
    # Validate the installed backend even for a resumed/manual-hardware run.
    try:
        runtime = detect_triton_runtime(device_id=config.get("hardware", {}).get("device_id", 0),
                                        config_path=args.config)
    except (RuntimeError, ValueError, OSError) as exc:
        log.error("===== Triton Ascend runtime environment detection failed ===== %s", exc)
        state_log.error("===== Triton Ascend runtime environment detection failed ===== %s", exc)
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
    log.info("[target framework] framework=%s; backend=%s; driver=%s; versions=%s; device_id=%s; identity file=%s",
             FRAMEWORK, BACKEND, runtime["driver_backend"], runtime["runtime_versions"],
             device_info["device_id"], Path(work_dir) / "workflow_target.json")
    log.info("[CANN toolchain] Verified version files and fingerprint: %s", runtime.get("toolchain", {}))

    DEVICE_INFO_PROMPT = (
        file_hint(work_dir, Path(work_dir) / "device_info.json", "Hardware parameter source for this run",
                  "First verify the chip, programming model, core counts and UB/L1 capacities, then judge whether the scheme and tiling are feasible")
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
            log.info("[emergency import] Development evidence: eligible=%s; reason=%s; new binding=%s",
                     evidence.get("eligible"), evidence.get("reason"),
                     Path(work_dir) / "selection/current_implementation.json")
            if not evidence.get("eligible"):
                log.warning("[emergency import] Code still goes straight into formal evaluation; when development evidence does not match, scores are kept, "
                            "but they do not enter the best record or stagnation window, and later Stage3 must complete the evidence")
            transition(state, "ready_for_iter", state_log,
                       reason="Emergency import complete, skipping Stage1/1.5/2; compiling the specified implementation from Stage4")
        log.info("===== Emergency import: skipping Stage1 requirements analysis, Stage1.5 Jev scoring, Stage2 first implementation =====")
        log.info("[emergency import] Reusing imported fusion probabilities and Top N; current state=%s; old evaluations, history, best records or human feedback were not inherited",
                 state.current_stage)

    # ═══════════════════════════════════════════════════════════════
    # Stage1: requirements analysis (cannbot)
    # ═══════════════════════════════════════════════════════════════
    if state.current_stage == "N1_phase1" or stage_number(state.current_stage) == 1:
        stage = "Stage1-requirements analysis"
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
            hint_block = f"\n💡 Fusion direction hint (user-specified):\n{optimize_hint}\nIn the analysis, focus on the feasibility of this fusion direction and give concrete on-chip dataflow design suggestions.\n"
        prompt = (f"Working directory: {work_dir}\nOperator: {op_name}\n"
                  + file_hint(work_dir, Path(work_dir) / "task", "Read-only operator requirements and evaluation benchmark",
                              "Read in order: desc.md definitions, proto.yaml interfaces, cases.yaml coverage, golden.py reference semantics") +
                  f"{DEVICE_INFO_PROMPT}\n{hint_block}Please analyze the task requirements and write the output to {work_dir}/ANALYSIS.md\n"
                  f"Also output the compact English fusion requirements to {work_dir}/fusion_requirements.en.json following the role, "
                  "for stage1.5 Jev to score against real hardware and fusion methods."
                  f"The program estimates a requirements ceiling of {requirements_budget} UTF-8 JSON bytes from this round's method text, options and hardware; "
                  "organize the summary within this ceiling (including the JSON structure) and do not drop constraints that affect fusion selection."
                  "Stage1.5 validates the English inputs and checks the actual request size.") + CANNBOT_CONSTRAINT
        ok = run_agent("cannbot", role, work_dir, prompt, node_log=node_logs["N1"])
        if not ok:
            log.error(f"[{stage}] cannbot failed")
            raise RuntimeError("Stage1 failed; fusion selection requires completed requirements.")
        transition(state, "stage1.5", state_log, reason="Requirements analysis complete; starting Jev fusion scheme scoring")

    # stage1.5 can resume independently; only a valid persisted Jev result allows proceeding to the first implementation.
    if stage_number(state.current_stage) in (1.5, 2):
        stage = "Stage1.5-Jev fusion scheme selection"
        transition(state, "stage1.5", state_log, reason="Reading requirements, hardware and fusion methods; generating the candidate library")
        log_io(log, stage,
               [f"{work_dir}/ANALYSIS.md", f"{work_dir}/fusion_requirements.en.json",
                device_info_path, str(Path(__file__).parent / "knowledge/fusion_method.md"),
                str(Path(__file__).parent / "knowledge/fusion_options.json")],
               [str(fusion_library_path(work_dir)), f"{work_dir}/fusion/ranking.json"])
        append_work_record(work_dir, stage)
        run_fusion_selection(work_dir, config_path=args.config,
                             top_n=config.get("fusion_selection", {}).get("top_n", 3))
        transition(state, "stage2", state_log, reason="Jev scoring complete; the first version uses the highest-probability scheme")
    elif not imported and stage_number(state.current_stage) != 10:
        # The new workflow's checkpoint resume re-verifies input fingerprints and allows adjusting only n without re-calling Jev.
        if (Path(work_dir, "fusion_requirements.en.json").is_file()
                or Path(work_dir, "fusion").exists()):
            run_fusion_selection(work_dir, config_path=args.config,
                                 top_n=config.get("fusion_selection", {}).get("top_n", 3))
        else:
            log.warning("The old working directory has no stage1.5 requirements or candidate library yet; keeping the original iteration; new tasks will auto-generate the fusion library.")

    # ═══════════════════════════════════════════════════════════════
    # Stage2: write the first operator version (cannbot)
    # ═══════════════════════════════════════════════════════════════
    if stage_number(state.current_stage) == 2:
        stage = "Stage2-write first version"
        role = os.path.join(roles_dir, "n1_stage2_first_impl.md")
        design_dir = os.path.join(work_dir, "develop", "iter0")
        Path(design_dir).mkdir(parents=True, exist_ok=True)
        inputs = [f"{work_dir}/task/", f"{work_dir}/ANALYSIS.md", str(fusion_library_path(work_dir))]
        outputs = [f"{work_dir}/impl/", f"{design_dir}/design_rationale.md"]
        log_io(log, stage, inputs, outputs)
        append_work_record(work_dir, stage)

        hint_block = ""
        if optimize_hint:
            hint_block = f"\n💡 Fusion direction hint (user-specified):\n{optimize_hint}\nIn the implementation, focus the dataflow design on this fusion direction.\n"
        prompt = (
            f"Working directory: {work_dir}\nOperator: {op_name}\n"
            + file_hint(work_dir, Path(work_dir) / "task", "Read-only operator requirements",
                        "Verify definitions, interfaces, cases and reference results against desc.md, proto.yaml, cases.yaml and golden.py")
            + file_hint(work_dir, Path(work_dir) / "ANALYSIS.md", "Stage1 requirements analysis",
                        "Focus on the registered name, precision and shape constraints, target chip and implementation difficulties")
            + file_hint(work_dir, Path(work_dir) / "example", "Triton Ascend example project",
                        "Reference the package structure, registration and API usage; do not copy the example operator's semantics") +
            f"{DEVICE_INFO_PROMPT}\n"
            f"{hint_block}"
            f"{format_proven_patterns_for_prompt(work_dir)}"
            f"{format_regression_patterns_for_prompt(work_dir)}"
            f"{format_pitfalls_for_prompt(work_dir)}\n"
            f"Deliver a complete installable project under {work_dir}/impl/: put the core implementation in impl/cann_bench/, "
            f"export the task function through impl/cann_bench/__init__.py, and provide impl/setup.py and impl/build.sh.\n"
            f"Write the design rationale to {design_dir}/design_rationale.md\n"
            f"Write the self-test report to {design_dir}/self_test_report.md"
        ) + CANNBOT_CONSTRAINT
        prompt += format_fusion_library_for_prompt(work_dir, "stage2")
        prompt += development_prompt(work_dir, design_dir, 2)
        previous_revisions = begin_development(work_dir, design_dir)
        ok = run_agent("cannbot", role, work_dir, prompt, node_log=node_logs["N1"])
        evidence = finalize_development(work_dir, design_dir, 2, agent_ok=ok,
                                        previous_revisions=previous_revisions)
        if not evidence.get("eligible"):
            log.warning(f"[{stage}] Scheme/self-test material does not yet satisfy the best-record conditions: {evidence.get('reason', '')}")
        if not ok:
            log.error(f"[{stage}] cannbot failed")
        transition(state, "ready_for_iter", state_log, reason="First version written; entering the iteration loop")

    # ═══════════════════════════════════════════════════════════════
    # Main iteration loop: stage4→5→6→(7→8→3)→back to 4
    # Supports checkpoint resume: decides which stage to continue from based on current_stage
    # ═══════════════════════════════════════════════════════════════
    def _get_resume_stage(current_stage: str) -> int:
        """Extract the stage number to resume from current_stage. E.g. iter1_stage5 → 5"""
        if "stage" in current_stage:
            try:
                return int(current_stage.split("stage")[-1])
            except ValueError:
                pass
        return 0  # not a stage format; start a new round

    iter_dirs = None  # initialized so stage10 does not hit a reference error if the while loop never runs
    while (stage_number(state.current_stage) != 10
           and (state.iteration < state.max_iterations
                or stage_number(state.current_stage) in (4, 5, 6, 7, 8, 9)
                or (stage_number(state.current_stage) == 3
                    and state.stage9_context.get("phase") == "developing"))):
        # Decide whether to resume from mid-flow (checkpoint resume)
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
            # checkpoint resume; do not increase iteration
            log.info(f"{'─'*40} ITER {state.iteration}/{state.max_iterations} (resuming from stage{resume_stage}) {'─'*40}")
        else:
            # normal new round
            state.iteration += 1
            state.flush()
            log.info(f"{'─'*40} ITER {state.iteration}/{state.max_iterations} {'─'*40}")
            resume_stage = 4  # a new round starts at stage4

        # iter directories for the current round
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

        # ── Stage4: build & deploy (kerminal)─────────────────────
        if resume_stage <= 4:
            stage = f"ITER{state.iteration}-Stage4-build"
            role = os.path.join(roles_dir, "n2_stage4_build.md")
            # Find the most recent self-test report (stage2 in iter0, stage3 in the current iter)
            self_test_path = None
            for i in range(state.iteration, -1, -1):
                candidate = os.path.join(work_dir, "develop", f"iter{i}", "self_test_report.md")
                if os.path.exists(candidate):
                    self_test_path = candidate
                    break
            self_test_hint = (file_hint(work_dir, self_test_path, "Most recent development self-test report",
                                       "Check install, import, given cases and repeated-call results; it is a development round and not equivalent to formal precision evaluation")
                              if self_test_path else "")
            if imported:
                self_test_hint += file_hint(
                    work_dir, Path(work_dir) / "init_impl_manifest.json", "Emergency import source and copy manifest",
                    "Verify the specified code against its development round; the imported self-test is the original development evidence and does not mean this build or precision has passed")
                self_test_hint += "\nBuild the currently specified implementation directly; the existence of an imported report cannot substitute for this real build and formal evaluation.\n"
            inputs = [f"{work_dir}/impl/"]
            if self_test_path:
                inputs.append(self_test_path)
            outputs = [f"{iter_dirs['build']}/build.log"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage4", state_log, reason="Starting build and deployment")

            prompt = (f"Working directory: {work_dir}\nOperator: {op_name}\n"
                      + file_hint(work_dir, Path(work_dir) / "impl", "Current project awaiting build and deployment",
                                  "Check dependencies, registration entry points and real build errors from the project config and the cann_bench package")
                      + f"{DEVICE_INFO_PROMPT}\nPlease build and deploy, writing results to {iter_dirs['build']}/build.log\n{self_test_hint}")
            ok = run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N2"])

            build_log_path = f"{iter_dirs['build']}/build.log"
            build_success = check_build_success(build_log_path)
            if not ok and build_success:
                log.warning(f"[{stage}] kerminal exited abnormally but build.log shows SUCCESS; trusting build.log")
            if not build_success:
                log.warning(f"[{stage}] Build failed → tech_lead summary → Stage3")
                state_log.info(f"[verdict] Build failed: build.log has no STATUS: SUCCESS")
                append_work_record(work_dir, f"{stage} FAIL → Stage3")
                append_ledger_only(work_dir, state.iteration, "build_fail")
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="build_fail", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="build_fail", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

        # ── Stage5: precision evaluation (kerminal)─────────────────────
        if resume_stage <= 5:
            stage = f"ITER{state.iteration}-Stage5-precision"
            role = os.path.join(roles_dir, "n2_stage5_precision_eval.md")
            inputs = [f"task_dir={task_dir}"]
            outputs = [f"{iter_dirs['eval']}/precision_result.json"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage5", state_log, reason="Build succeeded; starting precision evaluation")

            precision_impl_hash = _implementation_hash_or_empty(work_dir)
            precision_context = comparison_context(task_dir, device_info, config)
            precision_path = Path(iter_dirs["eval"]) / "precision_result.json"
            previous_precision_revision = _file_revision(precision_path)
            binding_path = Path(iter_dirs["eval"]) / "precision_binding.json"
            atomic_write_json(str(binding_path), {"eligible": False, "reason": "Precision evaluation pending"})
            prompt = (f"Working directory: {work_dir}\nOperator: {op_name}\ntask_dir: {task_dir}\n"
                      + file_hint(work_dir, Path(work_dir) / "task", "Read-only task entry for precision evaluation (linked to the task_dir above)",
                                  "Use the full cases from cases.yaml and the golden.py reference implementation, following the proto.yaml interface")
                      + f"{DEVICE_INFO_PROMPT}\nWrite results to {iter_dirs['eval']}/precision_result.json")
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
                    log.warning(f"[{stage}] This precision run did not produce a new report; the old report is not bound for best-implementation selection")
            # Create precision report symlink (to the specific file, not the whole directory)
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
                log.warning(f"[{stage}] Precision failed({prec_passed}/{prec_total}) → tech_lead summary → Stage3")
                state_log.info(f"[verdict] Precision failed: passed={prec_passed}/{prec_total}")
                append_work_record(work_dir, f"{stage} FAIL → Stage3")
                append_ledger_only(work_dir, state.iteration, "precision_fail")
                run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="precision_fail", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="precision_fail", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
                continue

        # ── Stage6: performance evaluation (cann-bench executed directly by code)─────────
        if resume_stage <= 6:
            stage = f"ITER{state.iteration}-Stage6-performance"
            inputs = [f"task_dir={task_dir}"]
            outputs = [f"{iter_dirs['eval']}/perf_result.json"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage6", state_log, reason="Precision passed; starting performance evaluation")
            measurement_impl_hash = _implementation_hash_or_empty(work_dir)
            evaluation_context = comparison_context(task_dir, device_info, config)

            # Execute cann-bench directly (not through an agent)
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            cannbench_src = os.path.abspath(os.path.join(_script_dir, config.get("paths", {}).get("cannbench_repo", "../cann-bench"), "src"))
            if not os.path.isdir(cannbench_src):
                cannbench_src = os.path.abspath(os.path.join(_script_dir, config.get("paths", {}).get("cannbench_repo", "../cann-bench")))

            log.info(f"[{stage}] Running cann-bench performance evaluation (executed directly by code)")
            node_logs["N2"].info(f"Performance evaluation started, task={task_dir}")

            # Retry at most 2 times (occasional segfault / NPU device busy)
            max_retries = 2
            for attempt in range(1, max_retries + 1):
                # An isolated output root prevents retries/resume from using old profiler files.
                reports_dir = os.path.join(iter_dirs["eval"], "perf_reports",
                                           f"attempt_{attempt}_{uuid.uuid4().hex}")
                Path(reports_dir).mkdir(parents=True, exist_ok=True)
                log.info(f"[{stage}] Evaluation output directory for this run: {reports_dir}")
                success, eval_output, report_path = run_perf_eval(
                    task_dir, cannbench_src, device_id=config.get("hardware", {}).get("device_id", 0),
                    log_callback=lambda msg: node_logs["N2"].info(msg), config_path=args.config,
                    reports_dir=reports_dir)
                node_logs["N2"].info(f"Performance evaluation (attempt {attempt}/{max_retries}), success={success}, report={report_path}")
                log.info(f"[{stage}] attempt {attempt}: success={success}, report={'yes' if report_path else 'none'}")
                if success or report_path:
                    break
                if attempt < max_retries:
                    log.warning(f"[{stage}] Performance evaluation crashed (attempt {attempt}); retrying in 10 seconds...")
                    node_logs["N2"].warning(f"Evaluation crashed (attempt {attempt}); retrying in 10 seconds")
                    import time as _time; _time.sleep(10)
                else:
                    log.error(f"[{stage}] Performance evaluation crashed {max_retries} times in a row; workflow stopping")
                    node_logs["N2"].error(f"Evaluation crashed {max_retries} times in a row; workflow stopping")
                    append_work_record(work_dir, f"{stage} cann-bench crashed {max_retries} times in a row; workflow stopped at stage6")
                    state.flush()
                    raise RuntimeError(f"cann-bench performance evaluation crashed {max_retries} times in a row; workflow stopped at {state.current_stage}. After checking the NPU device state, rerun to resume from the stage6 checkpoint.")

            # Save the complete profiler tree first; every JSON link points at this work copy.
            source_csv_dir = archive_profiler_data(
                report_path, iter_dirs["eval"], log_callback=log.info)

            # Parse the report to generate perf_result.json
            if report_path:
                perf = parse_cannbench_report_to_perf_result(report_path, source_csv_dir)
                missing_csv = [case.get("case_id", "?") for case in perf.get("cases", [])
                               if not case.get("kernel_csv")]
                log.info(f"[{stage}] profiler association: source_csv_dir={source_csv_dir or 'none'}; "
                         f"kernel_csv associated for {len(perf.get('cases', [])) - len(missing_csv)}/"
                         f"{len(perf.get('cases', []))} cases")
                if missing_csv:
                    log.warning(f"[{stage}] kernel_csv not associated for cases={missing_csv}; "
                                "CSV missing or multiple independent CSVs; batch collection does not force per-case association; "
                                "check this round's raw prof_data files")
            elif not success:
                perf = {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                        "score_error_code": "eval_crash",
                        "score_error": f"cann-bench evaluation process crashed (returncode={eval_output[-200:] if eval_output else 'no output'})",
                        "_error": "cann-bench evaluation crashed and produced no new report"}
                log.error(f"[{stage}] cann-bench evaluation crashed with no new report (an old report may have been used; it was filtered out)")
            else:
                perf = {"perf_pass": False, "overall_score": 0.0, "avg_speedup": 0.0,
                        "_error": "cann-bench produced no report"}
                log.error(f"[{stage}] cann-bench produced no report")

            # Bind the context at measurement time. Never retrofit this stamp
            # onto old reports using the environment of a later resumed run.
            perf["comparison_context"] = copy.deepcopy(evaluation_context)
            perf["comparison_context_stable"] = (
                evaluation_context == comparison_context(task_dir, device_info, config))
            # Write perf_result.json
            atomic_write_json(f"{iter_dirs['eval']}/perf_result.json", perf)
            log.info(f"[{stage}] perf_result.json written: overall={perf.get('overall_score')}, avg_speedup={perf.get('avg_speedup')}")

            # Create performance report symlinks (to the specific file, not the whole directory)
            source_json = perf.get("source_json", "")
            if source_json and os.path.exists(source_json):
                link_dir = os.path.join(iter_dirs['eval'], "perf_reports")
                Path(link_dir).mkdir(parents=True, exist_ok=True)
                link_path = os.path.join(link_dir, os.path.basename(source_json))
                if os.path.islink(link_path) and os.path.realpath(link_path) != os.path.realpath(source_json):
                    os.unlink(link_path)
                if not os.path.exists(link_path):
                    os.symlink(source_json, link_path)
                # also link the identically named md and html files
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
            # Automatically write rounds/ledger hard data into history.json
            perf_diff = _compute_perf_diff(work_dir, state.iteration, expected_context=evaluation_context,
                                           log=log, state_log=state_log)
            prev_speedup = _history_previous_average(work_dir, state, log, comparison=perf_diff)
            append_round(work_dir, state.iteration, perf, prev_avg_speedup=prev_speedup,
                         comparison=perf_diff)
            import json as _json
            _h = load_history(work_dir)
            history_log.info(f"ITER{state.iteration} [stage6 program wrote rounds/ledger]\n{_json.dumps(_h, ensure_ascii=False, indent=2)}")
            log.info(f"[{stage}] history.json rounds/ledger updated (iter{state.iteration})")

            # ═══════════════════════════════════════════════════
            # after stage6: the program computes the performance diff (computed once, shared by later stages)
            # ═══════════════════════════════════════════════════
            perf_diff_text = (perf_diff.get("comparison_text", "") + perf_diff.get("diff_text", "")
                              + perf_diff.get("regression_text", ""))
            if perf_diff.get("has_improvement"):
                state_log.info(f"[verdict] Performance improvement: avg_speedup {perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']} (+{perf_diff['delta_pct']}%), iter{perf_diff['prev_iter']}→iter{state.iteration}")
            elif perf_diff.get("has_regression"):
                state_log.info(f"[verdict] Performance regression: avg_speedup {perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']} ({perf_diff['delta_pct']}%), iter{perf_diff['prev_iter']}→iter{state.iteration}")
            else:
                delta = perf_diff.get("delta_pct", 0)
                if delta != 0:
                    state_log.info(f"[verdict] Performance change not significant: delta={delta}% (within ±5%)")
                elif perf_diff.get("comparable"):
                    state_log.info("[verdict] Same-protocol performance flat: delta=0%, no up/down experience generated")
                else:
                    state_log.info("[verdict] Not comparing up/down this round: %s", perf_diff.get("comparison_reason", "missing comparable data"))

            # ═══════════════════════════════════════════════════
            # post-stage6 routing decisions (order matters; do not swap)
            # ① anti-cheating first → ② then performance target → ③ normal performance optimization
            # ═══════════════════════════════════════════════════

            # ── ① Anti-cheating/zero-score detection (highest priority)──
            score_error = perf.get("score_error_code") or perf.get("score_error", "")
            avg_speedup = perf.get("avg_speedup", 0.0)
            valid_average = (isinstance(avg_speedup, (int, float)) and not isinstance(avg_speedup, bool)
                             and math.isfinite(avg_speedup) and avg_speedup > 0)
            if score_error or not valid_average:
                reason_detail = score_error or ("avg_speedup=0 (all case performance data is zero)" if avg_speedup == 0.0
                                                else "avg_speedup has no valid positive measurement")
                state_log.info(f"[verdict] Performance evaluation abnormal or zero score: {reason_detail}, avg_speedup={avg_speedup}")
                csv_analysis = analyze_kernel_csv_for_anticheat(perf)
                log.warning(f"[{stage}] Anti-cheating triggered or zero score: {reason_detail} → tech_lead summary → Stage3")
                log.info(f"[{stage}] kernel_csv analysis:\n{csv_analysis}")
                append_work_record(work_dir, f"{stage} anti-cheat/zero score({reason_detail}) → Stage3")
                anticheat_context = (
                    f"score_error_code: {score_error}\n"
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "Anti-cheat/zero-score performance summary",
                                "Verify score_error_code, per-case status and kernel_csv paths")
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "Raw performance reports for the zero score",
                                "Compare error codes with raw event evidence to distinguish execution failure from ordinary performance issues")
                    + file_hint(work_dir, Path(iter_dirs['eval']) / 'prof_data', "Profiler data for the zero score",
                                "Verify whether the custom NPU kernel actually executed and whether any forbidden host-side computation occurred") +
                    f"\n=== kernel_details.csv automatic analysis ===\n{csv_analysis}\n"
                )
                anticheat_directive = (
                    f"\nThis round's scoring is abnormal; first verify the error type and raw evidence. A missing report or zero score alone does not prove a violation.\n"
                    f"score_error_code: {reason_detail}\n"
                    f"\n=== kernel_details.csv automatic analysis (parsed for you by the program) ===\n{csv_analysis}\n"
                    f"\nProceed as follows:\n"
                    f"1. Compare the actual error code, whether a report was generated, execution logs and kernel CSV to distinguish reporting/collection failures from code problems.\n"
                    f"2. If the evidence involves cheating, read knowledge/anti_cheat_reference.md against the actual error handling table; do not fabricate violation conclusions without proof.\n"
                    f"3. Give a fix direction only for root causes already located, and state hypotheses still awaiting verification.\n"
                    f"4. Keep the anti-cheating rules; caching results, sleep and similar evaluation-bypassing schemes are forbidden; do not blindly rewrite the core algorithm over a missing report.\n"
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
                log.warning(f"[{stage}] Not updating the best implementation or stagnation window: {selection_status.get('reason', '')}")
            else:
                best = selection_status["best"]
                window_status = selection_status["window"]
                best_changed = best['record_id'] != previous_best_id
                log.info(f"[best implementation] {'updated' if best_changed else 'kept'} iter{best['iteration']}: "
                         f"avg_speedup={best['avg_speedup']}, avg_speed={best['avg_speed']}, "
                         f"HAP.performance_score={best['hap'].get('performance_score')}, "
                         f"all_cases_pass={best['all_cases_pass']}, "
                         f"min_case_speedup={min(c['speedup'] for c in best['case_results'])}, "
                         f"fusion methods={best['fusion_scheme'].get('method_ids', [])}")
                log.info(f"[best implementation] code snapshot={best['implementation_dir']}; "
                         f"performance report={best['performance_report']}; performance summary={best['performance_result']}; "
                         f"manifest={best['manifest_path']}; "
                         f"scheme rationale={best['evidence_paths'].get('decision_rationale', '')}")
                if best_changed:
                    log.debug("[best implementation] HAP raw per-case metrics (not recomputed)=%s",
                              _json_module.dumps(best['hap']['cases'], ensure_ascii=False))
                event = log_semantic_trigger(work_dir, state.iteration, selection_status, log, state_log)
                if not event:
                    state_log.info(f"[semantic window] {window_status['status']}: "
                                   f"{window_status['completed_improvements']}/{window_status['required_improvements']}, "
                                   f"cumulative improvement={window_status['cumulative_improvement']:.2%}, "
                                   f"exit={selection_status['should_exit']}, "
                                   f"review fusion scheme={selection_status['review_fusion']}")

        # ── ② After meeting the target, the program window decides exit; Stage9 still completes its original knowledge accumulation ──
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
                "All cases have reached the target speedup. Keep P0/P1/P2 and review next steps against measured bottlenecks; "
                "do not force a fusion scheme change merely because of multiple kernels or HBM intermediates."
                "Exit is decided by the program's valid performance window; exit_decision does not control routing."
                + ("This round met the semantic exit condition; complete the experience records, then summarize the best evaluated implementation."
                   if should_exit else "This round has not met the semantic exit condition; continue offering evidence-backed optimization suggestions.")
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
                append_work_record(work_dir, "The passing implementation\'s cumulative valid improvements over x iterations are under 5%; exiting with the best evaluated snapshot")
                transition(state, "stage10", state_log, reason="Program semantic exit; delivering the best passing implementation")
                break
            self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="perf_pass_optimize",
                             state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs,
                             device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)
            continue

        # ── ③ Performance below target still goes stage7/8/9/3; the y window only triggers a scheme review ──
        state_log.info("[verdict] Performance below target → stage7→8→9→3; whether to change the fusion scheme is decided by stage9 based on evidence")

        # ── Stage7: Profiling analysis (kerminal)───────────────
        if resume_stage <= 7:
            stage = f"ITER{state.iteration}-Stage7-profiling"
            role = os.path.join(roles_dir, "n2_stage7_kerminal_profile.md")
            inputs = [f"{iter_dirs['eval']}/perf_result.json", f"{iter_dirs['eval']}/perf_reports/"]
            if Path(fusion_library_path(work_dir)).is_file():
                inputs.append(str(fusion_library_path(work_dir)))
            outputs = [f"{iter_dirs['profile']}/bottleneck_analysis.md"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage7", state_log, reason="Performance below target; starting profiling analysis")

            prev_design_path = None
            for i in range(state.iteration - 1, -1, -1):
                candidate = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
                if os.path.exists(candidate):
                    prev_design_path = candidate
                    break
            if prev_design_path is None:
                # Imported Stage3 material may only have its fusion-decision
                # document; never inject a nonexistent Stage2 design filename.
                imported_rationale = Path(work_dir) / "develop/iter0/fusion_scheme_rationale.md"
                if imported and imported_rationale.is_file():
                    prev_design_path = str(imported_rationale)
                else:
                    prev_design_path = os.path.join(work_dir, "develop", "iter0", "design_rationale.md")

            profiling_guide_path = os.path.join(os.path.dirname(roles_dir), "knowledge", "profiling_guide.md")
            profiling_guide_hint = (
                "\n" + file_hint(work_dir, profiling_guide_path, "Profiling analysis guide",
                                  "Read the five-file analysis process and attention weights first, then explain the bottleneck evidence",
                                  base_dir=Path(__file__).parent, base_label="project root") +
                "Read the guide above and analyze each case's profiler evidence in the order "
                "kernel_details→op_statistic→step_trace_time→api_statistic→trace_view. "
                "Allocate analysis attention at 50%/20%/15%/10%/5%; these weights guide the analysis, not official performance scoring. "
                "Use the original report for official metrics; state missing evidence and verify attribution against the evaluated code.\n"
            ) if os.path.exists(profiling_guide_path) else ""

            prompt = (
                f"Working directory: {work_dir}\nOperator: {op_name}\n"
                + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "This round's formal performance summary",
                            "Look at avg_speedup, each case's speedup and worst_6_cases first, then find performance bottlenecks")
                + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "This round's raw performance reports",
                            "Compare raw timing, kernel and profiler evidence per case to verify the summary conclusions")
                + file_hint(work_dir, prev_design_path, "Most recent usable historical design reference",
                            "Look at tiling, dataflow and change intent; it does not automatically represent the evaluated version and must be checked against the separately attached code binding") +
                f"{DEVICE_INFO_PROMPT}\n"
                f"{perf_diff_text}\n"
                f"{format_proven_patterns_for_prompt(work_dir)}"
                f"{format_regression_patterns_for_prompt(work_dir)}"
                f"{format_pitfalls_for_prompt(work_dir)}\n"
                f"{profiling_guide_hint}"
                f"Write output to {iter_dirs['profile']}/bottleneck_analysis.md"
            )
            prompt += format_fusion_library_for_prompt(work_dir, "stage7")
            prompt += format_evidence_for_prompt(work_dir)
            prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
            # A real Stage7 rerun must produce fresh output; an interrupted
            # attempt's partial report must not satisfy the output check.
            stale_report = validate_profile_report_directory(work_dir, state.iteration)
            if stale_report.exists():
                stale_report.unlink()
                log.info("[performance analysis report] iter=%s re-running Stage7; removed this round's old report and waiting for regeneration: %s",
                         state.iteration, stale_report)
            if not run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N2"]):
                raise RuntimeError(f"[{stage}] Stage7 did not complete successfully; the analysis report will not be delivered")

        # Each evaluated iteration exposes one canonical report. This also
        # stamps existing Stage7 reports when resuming at Stage8/9.
        try:
            report_path = stamp_stage7_report(work_dir, state.iteration)
        except (OSError, ValueError) as exc:
            log.error("[performance analysis report] iter=%s source=Stage7; check failed, not proceeding to later stages: %s",
                      state.iteration, exc)
            raise RuntimeError(f"Stage7 analysis report is not deliverable: {exc}") from exc
        _log_profile_report(log, state_log, state.iteration, 7, report_path)

        # ── Stage8: search for optimization schemes (hermes)───────────────────
        if resume_stage <= 8:
            stage = f"ITER{state.iteration}-Stage8-search"
            role = os.path.join(roles_dir, "n3_stage8_search.md")
            inputs = [f"{iter_dirs['profile']}/bottleneck_analysis.md", f"{work_dir}/impl/"]
            if Path(fusion_library_path(work_dir)).is_file():
                inputs.append(str(fusion_library_path(work_dir)))
            outputs = [f"{iter_dirs['search']}/SEARCH_REPORT.md", f"{iter_dirs['search']}/FIX_DIRECTIVE.md"]
            log_io(log, stage, inputs, outputs)
            append_work_record(work_dir, stage)
            transition(state, "stage8", state_log, reason="profiling complete; starting search for optimization schemes")

            prompt = (
                f"Working directory: {work_dir}\nOperator: {op_name}\n"
                + file_hint(work_dir, Path(iter_dirs['profile']) / 'bottleneck_analysis.md', "This round's bottleneck analysis",
                            "Determine the search questions from slow cases, root causes and evidence; separate proven bottlenecks from unverified hypotheses")
                + file_hint(work_dir, Path(work_dir) / 'impl', "Current operator implementation project",
                            "Locate the operator entry, tiling and dataflow in cann_bench and verify whether the search scheme applies") +
                f"{DEVICE_INFO_PROMPT}\n"
                f"{perf_diff_text}\n"
                f"{format_proven_patterns_for_prompt(work_dir)}"
                f"{format_regression_patterns_for_prompt(work_dir)}\n"
                f"Outputs: {iter_dirs['search']}/SEARCH_REPORT.md and {iter_dirs['search']}/FIX_DIRECTIVE.md"
            )
            prompt += format_fusion_library_for_prompt(work_dir, "stage8")
            prompt += format_evidence_for_prompt(work_dir)
            prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
            run_agent("hermes", role, work_dir, prompt, node_log=node_logs["N3"])

        # ── Stage9: Tech Lead experience distillation (kerminal)──────────
        if resume_stage <= 9:
            run_tech_lead(log, roles_dir, work_dir, op_name, state, state_log, node_logs, iter_dirs, fail_reason="perf_optimize", device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context, perf_diff=perf_diff)

        # ── Stage3: fix and optimize (cannbot)──────────────────────
        self_goto_stage3(log, roles_dir, work_dir, op_name, state, reason="perf_optimize", state_log=state_log, node_logs=node_logs, iter_dirs=iter_dirs, device_info_prompt=DEVICE_INFO_PROMPT, selection_context=evaluation_context)

        # one round done; iteration increments at the top of the next round

    # iteration limit
    if state.iteration >= state.max_iterations and not state.stopped_by:
        state.stopped_by = "max_iterations"
        log.warning(f"Reached the maximum iteration count {state.max_iterations}")

    # ═══════════════════════════════════════════════════════════════
    # Stage10: final report (kerminal) — N5 node
    # ═══════════════════════════════════════════════════════════════
    # Ensure iter_dirs has a value (fallback: use the last round\'s)
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
    stage = "Stage10-final report"
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
    transition(state, "stage10", state_log, reason="Generating the final report")

    # ── Scan unhandled question.md files (subordinate feedback never adjudicated by tech_lead)──
    import glob as _glob
    unhandled_questions = []
    for qpath in _glob.glob(os.path.join(work_dir, "develop", "iter*", "question.md")):
        with open(qpath, "r", encoding="utf-8") as f:
            qcontent = f.read()
        # Adjudicated ones are in the pitfall log; simple check: whether the question has a corresponding pitfall
        if "## tech_lead adjudication" not in qcontent:
            unhandled_questions.append(qpath)
    if unhandled_questions:
        log.warning(f"[{stage}] ⚠️ {len(unhandled_questions)} unadjudicated question.md files (caused by early iteration termination):")
        for q in unhandled_questions:
            log.warning(f"    {q}")
        state_log.info(f"[note] {len(unhandled_questions)} question.md files were not adjudicated by tech_lead (early iteration exit)")

    question_hint = ""
    if unhandled_questions:
        question_hint = "".join(file_hint(work_dir, path, "Unadjudicated development feedback",
                                        "Look at the questioned suggestions and hardware/framework evidence; list them as open issues in the final report rather than as settled conclusions")
                                for path in unhandled_questions)

    prompt = (
        f"Working directory: {work_dir}\nOperator: {op_name}\n"
        + file_hint(work_dir, final_precision_path, "Precision results used by the final report",
                    "Verify the pass counts, failed cases and correctness; when a best snapshot exists, use the results inside the snapshot")
        + file_hint(work_dir, final_perf_path, "Performance results used by the final report",
                    "Verify avg_speedup, HAP and per-case target attainment; do not present the last modification as the best evaluated version")
        + file_hint(work_dir, Path(work_dir) / 'WORK_RECORD.md', "Workflow execution record",
                    "Review stages, failure reasons and the exit course chronologically")
        + file_hint(work_dir, Path(work_dir) / '.state.json', "Program final state and exit reason",
                    "Read iteration, stopped_by and the stage; cross-check with the semantic window; do not infer the exit reason yourself")
        + file_hint(work_dir, Path(work_dir) / 'knowledge/history.json', "Cross-round optimization experience and ledger",
                    "Summarize measured changes, verified experience and unresolved issues from rounds, ledger and insights") +
        f"{question_hint}"
        f"{DEVICE_INFO_PROMPT}\n"
        f"Write output to {work_dir}/FINAL_REPORT.md"
    )
    if selected:
        prompt += (
            "\n" + file_hint(work_dir, selected['manifest_path'], "Manifest of the finally selected implementation",
                              "Verify the code fingerprint, metrics, fusion scheme and report associations to ensure the conclusions belong to the same snapshot")
            + file_hint(work_dir, selected['implementation_dir'], "Immutable code snapshot of the finally selected implementation",
                        "Use it as the source of the finally delivered code; do not substitute unevaluated changes from the current impl/") +
            "Use this snapshot and its evaluation reports to describe the final delivery; the current impl/ may contain unevaluated changes. "
            "State clearly whether this implementation passes all cases, and do not present the best non-passing candidate as passing.\n"
        )
    else:
        prompt += "\nNo best record yet satisfies the correctness, self-test and unified-protocol requirements. Report this fact explicitly, and do not proclaim the last modification as the best verified implementation.\n"
    prompt += format_selection_for_prompt(work_dir, comparison_context=evaluation_context)
    if human.all_messages():
        prompt += file_hint(work_dir, human.root / "state.json", "Human feedback processing and execution status",
                            "Distinguish item by item: processed, executed but awaiting formal verification, or not executed due to exit conditions; list the reasons")
        prompt += file_hint(work_dir, human.root / "inbox", "Human verbatim messages and questions", "Associate by message number with Stage9 adjudications; do not write processed as executed")
    while True:
        if not run_agent("kerminal", role, work_dir, prompt, node_log=node_logs["N5"]):
            raise RuntimeError("Stage10 summarization failed; state kept for resume; the human entry is not yet closed")
        if not human.pending_messages():
            human.close_workflow(state.stopped_by or "completed")
            if not human.pending_messages():
                break
        _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                               node_logs, iter_dirs, DEVICE_INFO_PROMPT, evaluation_context)
        human.mark_unexecuted([item["id"] for item in human.all_messages()
                               if item.get("kind") == "direction" and item.get("status") == "processed"],
                              state.stopped_by or "workflow_final_report")
        transition(state, "stage10", state_log, reason="Updating the final report after supplementing human feedback processing results")
        prompt += "\nHuman feedback received during summarization has been re-reviewed by Stage9; re-read human_review/state.json and history, and state the reasons anything was not executed.\n"
        prompt += file_hint(work_dir, human.root / "state.json", "Final processing status of human feedback", "List the reasons for non-execution by message number; do not treat processed as executed")
        prompt += file_hint(work_dir, human.root / "inbox", "The human's complete verbatim messages", "Match item by item with human_responses in state and history")

    log.info("=" * 60)
    log.info(f"Done! stopped_by={state.stopped_by}")
    log.info(f"Report: {work_dir}/FINAL_REPORT.md")
    log.info("=" * 60)
    state.flush()


def _compute_perf_diff(work_dir: str, iteration: int, *, expected_context=None, log=None, state_log=None) -> dict:
    """
    The program computes the performance diff between this round and the previous one. Returns a dict:
    {
        "has_improvement": True/False,
        "prev_iter": 2, "curr_iter": 3,
        "prev_avg_speedup": 1.5, "curr_avg_speedup": 3.0,
        "delta_pct": 100.0,
        "prev_design_path": "develop/iter2/design_rationale.md",
        "curr_design_path": "develop/iter3/design_rationale.md",
        "case_diffs": [  # sorted by improvement, descending
            {"case_id": "14", "prev_speedup": 0.69, "curr_speedup": 3.29, "delta": "+377%"},
            ...
        ],
        "diff_text": "formatted text that can be injected into a prompt directly"
    }
    """
    import json as _json
    result = {"has_improvement": False, "has_regression": False, "diff_text": "", "regression_text": "",
              "comparable": False, "comparison_status": "unavailable", "curr_iter": iteration,
              "comparison_reason": "no complete, verifiable performance report for this round"}

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
            message = ("===== Performance comparison protocol check ===== iter=%s previous_iter=%s; status=%s; reason=%s; "
                       "current report=%s; previous report=%s; check record=%s")
            args = (iteration, result.get("prev_iter"), result["comparison_status"],
                    result["comparison_reason"], result.get("current_report"),
                    result.get("previous_report"), comparison_path)
            for logger in (log, state_log if state_log is not log else None):
                if logger is not None:
                    (logger.info if result["comparable"] else logger.warning)(message, *args)
        if not result["comparable"]:
            action = ("This round becomes the new baseline; subsequent comparisons use only complete results with the same protocol."
                      if result["comparison_status"] == "new_baseline" else "A complete evaluation with a clear protocol is required before comparing.")
            result["comparison_text"] = (
                f"\n===== No performance up/down comparison this round =====\n{result['comparison_reason']}; {action}"
                "Do not infer optimization gains or regressions from this mean difference, and do not fill in proven_pattern/regression_pattern for this round.\n"
                + file_hint(work_dir, comparison_path, "The program's performance comparison protocol check",
                            "See why no comparison was made and the source of both rounds' reports; do not attribute across protocols")
            )
        if result.get("current_report_source") == "verified_selection_archive":
            result["comparison_text"] = result.get("comparison_text", "") + file_hint(
                work_dir, result["current_report"], "This round's verified performance archive used when the working copy is missing",
                "Snapshot integrity and round were verified; read the actual scores and the archive's protocol; do not use up/down conclusions from an old cache")
        return result

    def positive_number(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)

    # read this round\'s perf_result
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
        result["comparison_reason"] = current_check["reason"].replace("previous report", "current report", 1)
        result["comparison_mismatch_fields"] = [
            "current." + field[len("previous."):] if field.startswith("previous.") else field
            for field in current_check["mismatch_fields"]
        ]
        return finish()
    curr_avg = curr_perf.get("avg_speedup", 0)
    result["curr_avg_speedup"] = curr_avg

    # Find the previous round\'s valid perf_result (skipping rounds that never ran performance due to build_fail/precision_fail)
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
        result.update(comparison_status="new_baseline", comparison_reason="no valid previous-round performance result; not computing up/down")
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

    # design_rationale paths
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

    # Keep changes for all alignable cases; the prompt shows only the six most relevant.
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

    # Format the diff text
    if result["has_improvement"]:
        lines = [
            f"\n📊 Performance improvement detected (computed automatically by the program; data is trustworthy)",
            f"  avg_speedup: {prev_avg} → {curr_avg} (+{delta_pct}%)",
            f"  Compared rounds: iter{prev_iter} → iter{iteration}",
        ]
        if result.get("prev_design_path"):
            lines.append(file_hint(work_dir, result['prev_design_path'], "Historical design reference near the previous round's evaluation",
                                   "For reviewing change intent; the actually evaluated version is determined by the code-bound selection rationale and snapshot").rstrip())
        if result.get("curr_design_path"):
            lines.append(file_hint(work_dir, result['curr_design_path'], "Historical design reference near this round's evaluation",
                                   "Compare with the previous round's reference; first confirm the version via the code binding or snapshot, then attribute the success").rstrip())
        if case_diffs:
            lines.append(f"  Per-case comparison (largest improvement first):")
            for cd in case_diffs[:6]:
                lines.append(f"    case_{cd['case_id']}: {cd['prev_speedup']} → {cd['curr_speedup']} ({cd['delta']})")
        lines.append("")
        lines.append("⚡ Big performance gain this round! Compare the two rounds\' design_rationale above, summarize why this change worked, and fill in the proven_pattern field.")
        result["diff_text"] = "\n".join(lines)

    elif result["has_regression"]:
        lines = [
            f"\n⚠️ Performance regression detected (computed automatically by the program; data is trustworthy)",
            f"  avg_speedup: {prev_avg} → {curr_avg} ({delta_pct}%)",
            f"  Compared rounds: iter{prev_iter} → iter{iteration}",
        ]
        if result.get("prev_design_path"):
            lines.append(file_hint(work_dir, result['prev_design_path'], "Historical design reference near the previous round's evaluation",
                                   "For reviewing the pre-regression design intent; the actually evaluated version is determined by the binding and snapshot").rstrip())
        if result.get("curr_design_path"):
            lines.append(file_hint(work_dir, result['curr_design_path'], "Historical design reference near this round's evaluation",
                                   "First confirm the version via the code binding or snapshot, then check the change against slow cases and attribute the failure lesson").rstrip())
        if case_diffs:
            lines.append(f"  Per-case comparison (largest regression first):")
            regression_cases = sorted(case_diffs, key=lambda x: float(x["delta"].replace("%", "").replace("+", "")))
            for cd in regression_cases[:6]:
                lines.append(f"    case_{cd['case_id']}: {cd['prev_speedup']} → {cd['curr_speedup']} ({cd['delta']})")
        lines.append("")
        lines.append("🚨 Performance regression this round! Compare the two rounds\' design_rationale above, analyze the cause and fill in regression_pattern with explicit applicability conditions; avoid repeating failures under the verified conditions, and re-verify when conditions change.")
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
        log.warning("[history ledger] iter%s's previous round iter%s lacks a valid avg_speedup; the previous value is recorded as null and old state.speedup is not used instead.",
                    state.iteration, previous_iteration)
    return None


def _review_pending_humans(log, roles_dir, work_dir, op_name, state, state_log,
                           node_logs, iter_dirs, device_info_prompt, selection_context):
    if not HumanReview(work_dir).pending_messages():
        return
    context = state.stage9_context
    reason = context.get("fail_reason") or "score_zero: no valid evaluation yet; only handling human feedback and non-execution reasons"
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
    transition(state, "stage9", state_log, reason=f"keeping the original scene {scene}")
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
            raise RuntimeError("Human consultation does not match the current Stage9 scene; using the old question for a new version is forbidden")
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
            log.warning("[human consultation] Three consecutive triggers of valid improvement under 5%%; please read %s; wait 2 minutes; replying \"please wait\" extends the wait once by 10 minutes.", question_path)
            state_log.warning("[human consultation] question=%s; original scene=%s", question_path, scene)
            consultation = human.start_wait(consultation["request_id"])
        if consultation["status"] == "waiting":
            log.info("[human wait] request=%s, reusing deadline=%s", consultation["request_id"], consultation["deadline"])
            consultation = human.wait_consultation(consultation["request_id"])
        bundle_path = human.build_feedback_bundle(consultation["request_id"])
        context.update(bundle_path=bundle_path, phase="feedback")
        state.flush()
        log.info("[human re-review] status=%s, full question/dialogue/evidence=%s", consultation["status"], bundle_path)

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
        log.info("[human re-review] New feedback arrived while the Stage9 decision was being generated; re-reviewing within the same round before delivery")


def _stage9_case_catalog(work_dir, iteration, scene):
    """Bind known IDs to this evaluation; never guess case numbers from filenames."""
    report = Path(work_dir) / "eval" / f"iter{iteration}" / "perf_result.json"
    catalog = {"iteration": iteration, "case_ids": None, "source": None,
               "complete": False, "observed_case_ids": [],
               "read_hint": "No complete case ID list for this round was obtained; verify against task/cases.yaml and the actual code; do not claim the program has verified the mapping."}
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
                               read_hint="Use all of this round's case IDs as-is; do not limit to the six slowest. File mappings require separately attached routing and code evidence.")
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
        file_hint(work_dir, schema_path, "Program-defined Stage9 JSON format (read-only)",
                  "Fill in according to this round's fields, types and hierarchy; unlisted experience fields must be omitted; fill case_analysis per this round's requirements; modified files are aggregated automatically from changes")
        + file_hint(work_dir, template_path, "Decision filling template for this request (read-only)",
                    "Copy iteration/request_id, and fill in real conclusions, tasks and the listed performance experience; add benefited and harmed items to experience cases per measurements; blanks and nulls must follow the schema")
        + file_hint(work_dir, catalog_path, "Complete verifiable case ID list for this round (read-only)",
                    "When IDs are listed, select by their original names; without a list, check back against the task and do not treat the six slowest as all cases")
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
            error(f"{field}: file path must be a string, got {value!r}")
            return None
        try:
            return _path(value, field)
        except ValueError as exc:
            error(str(exc))
            return None

    def paths(value, field):
        if not isinstance(value, list):
            error(f"{field}: must be an array of file paths")
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
                error(f"{field}: Stage9 cannot check file attributes: {path} ({exc})", key=("stat", path))
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
            error(f"{field}: Stage9 must not modify read-only inputs or program-maintained files: {path}",
                  key=("protected", path))
            return None
        try:
            target = physical(path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            error(f"{field}: Stage9 cannot resolve the file path: {path} ({exc})", key=("resolve", path))
            return None
        if writing and (not target.is_relative_to(root) or target == root):
            error(f"{field}: Stage9 file scope must be inside the working directory: {path}", key=("outside", path))
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
                error(f"{field}: Stage9 modification path actually points to a read-only input or program-maintained file: {path}",
                      key=("actual_protected", path))
                return None
        info = file_stat(target, field)
        if info is stat_failed:
            return None
        if info is not None and stat.S_ISDIR(info.st_mode):
            error(f"{field}: Stage9 must list concrete files; directories are not a substitute: {path}",
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
            error(f"{field}: {message}: {path}", key=("missing", path))

    if not isinstance(decision, dict):
        raise Stage9DecisionValidationError(["Stage9 file check requires a decision object"])
    # The request requires v2 even when the model forgot/invalidated its version
    # field. Do not invent a missing legacy writable list as another error.
    raw_v2 = raw_plan or decision.get("plan_version") == 2
    ledger = decision.get("ledger_entry", {})
    if not isinstance(ledger, dict):
        error("ledger_entry: must be an object")
        ledger = {}
    suggestions = decision.get("suggest_next", [])
    if not isinstance(suggestions, list):
        error("suggest_next: must be an array")
        suggestions = []
    tasks, all_writes = [], {}
    for index, suggestion in enumerate(suggestions):
        label = f"suggest_next[{index}]"
        if not isinstance(suggestion, dict):
            error(f"{label}: must be a task object")
            continue
        task_id = suggestion.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            label += f" ({task_id})"
        changes = suggestion.get("changes", [])
        if not isinstance(changes, list):
            error(f"{label}.changes: must be an array")
            changes = []
        writes, created = {}, set()
        for change_index, change in enumerate(changes):
            field = f"{label}.changes[{change_index}]"
            if not isinstance(change, dict):
                error(f"{field}: must be an object")
                continue
            path = path_value(change.get("file"), f"{field}.file")
            operation = change.get("operation")
            if operation not in ("create", "modify", "inspect"):
                error(f"{field}.operation: must be create, modify or inspect")
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
                    error(f"{record[3]}: declared as create but the file already exists; use modify instead: {path}",
                          key=("create_exists", path))
            else:
                require_file(record, "the actual file to inspect/modify does not exist")
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
            error(f"{field}: Stage9 writable path and read-only path actually point to the same file: {path}; "
                  f"{conflict[3]}={conflict[2]}", key=("readonly_conflict", target))
        if info is not None and info.st_nlink > 1:
            error(f"{field}: Stage9 file to write has multiple hard links; use an independent copy first: {path}",
                  key=("hardlink", target))
        for parent in target.parents:
            if parent in writable or parent in all_writes:
                error(f"{field}: Stage9 file to write cannot simultaneously be the directory of another file to write: {parent}; {path}",
                      key=("write_parent", parent, target))
            parent_info = file_stat(parent, field)
            if parent_info is not None and parent_info is not stat_failed and not stat.S_ISDIR(parent_info.st_mode):
                error(f"{field}: Stage9 parent of the file to write is already a file; cannot create or modify: {parent}; {path}",
                      key=("file_parent", parent, target))
                break
            if parent_info is stat_failed or parent == root:
                break

    for suggestion, label, writes, created in tasks:
        for target, record in writes.items():
            if target not in writable:
                error(f"{record[3]}: modification target conflicts with the actual file permission scope: {record[2]}",
                      key=("undeclared_write", target))
        bindings = suggestion.get("case_bindings", [])
        if not isinstance(bindings, list):
            error(f"{label}.case_bindings: must be an array")
            continue
        for index, binding in enumerate(bindings):
            field = f"{label}.case_bindings[{index}]"
            if not isinstance(binding, dict):
                error(f"{field}: must be an object")
                continue
            for path, location in paths(binding.get("implementation_files", []), f"{field}.implementation_files"):
                record = resolve(path, location)
                if path not in created:
                    require_file(record, "the implementation file associated with the case does not exist")
            evidence_items = binding.get("route_evidence", [])
            if not isinstance(evidence_items, list):
                error(f"{field}.route_evidence: must be an array")
                continue
            for evidence_index, evidence in enumerate(evidence_items):
                location = f"{field}.route_evidence[{evidence_index}].file"
                if not isinstance(evidence, dict):
                    error(f"{field}.route_evidence[{evidence_index}]: must be an object")
                    continue
                path = path_value(evidence.get("file"), location)
                if path is not None:
                    require_file(resolve(path, location), "the routing evidence file does not exist")
    if errors:
        raise Stage9DecisionValidationError(errors)


def _run_tech_lead_decision(log, roles_dir, work_dir, op_name, state, state_log, node_logs,
                            iter_dirs, fail_reason="", device_info_prompt="", perf_diff=None,
                            selection_context=None, *, human_messages=None, bundle_path=None,
                            consultation=None, scene=None):
    """
    Receives this round's Stage9 decision; the program merges history and saves experience.
    Supports being invoked at any fallback point (build_fail/precision_fail/score_zero/perf_optimize).
    Assembles the prompt flexibly based on the files that actually exist.
    """
    import json as _json
    stage = f"ITER{state.iteration}-Stage9-tech_lead({fail_reason})"
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
    transition(state, "stage9", state_log, reason=f"tech_lead experience summary({fail_reason})")

    # Collect all existing design_rationale files
    all_designs = []
    for i in (range(0, state.iteration + 1) if performance_scene
              else range(max(0, state.iteration - 1), state.iteration + 1)):
        candidate = os.path.join(work_dir, "develop", f"iter{i}", "design_rationale.md")
        if os.path.exists(candidate):
            all_designs.append(file_hint(work_dir, candidate, f"Implementation design rationale for iter{i}",
                                         "Compare algorithm, tiling, dataflow and change reasons across rounds, then verify against actual performance").rstrip())
    design_hint = "Design rationale from previous rounds:\n" + "\n".join(all_designs) + "\n" if all_designs else ""

    # Assemble the prompt from available files
    data_lines = []
    default_inputs = []
    missing_inputs = []
    for key, name, path, reading in [
        ("fusion_library", "JSON fusion operator library", str(fusion_library_path(work_dir)), "Compare candidate dataflows, prerequisites and initial probabilities; use measured evidence to decide keep or replace"),
        ("perf_result", "This round's performance result", f"{iter_dirs['eval']}/perf_result.json", "Look at avg_speedup, each case's speedup and the slowest cases, not just the average"),
        ("profiler", "This round's profiler data (check on demand)", f"{iter_dirs['eval']}/prof_data/", "Check per case only when conclusions conflict or evidence is missing"),
        ("perf_reports", "This round's performance reports", f"{iter_dirs['eval']}/perf_reports/", "Check raw timing and scoring to verify the performance summary and anti-cheating signals"),
        ("bottleneck", "This round's bottleneck analysis", f"{iter_dirs['profile']}/bottleneck_analysis.md", "Read Stage7 conclusions first; separate proven root causes from speculation"),
        ("fix_directive", "This round's modification directive", f"{iter_dirs['search']}/FIX_DIRECTIVE.md", "Read Stage8 conclusions first and convert them to the original P0/P1/P2 priorities"),
        ("search_report", "This round's search report", f"{iter_dirs['search']}/SEARCH_REPORT.md", "Check candidate methods, evidence and applicability conditions as needed"),
        ("build_log", "This round's build log", f"{iter_dirs['build']}/build.log", "Look at STATUS and the first real error to locate build/interface problems"),
        ("precision_result", "This round's precision result", f"{iter_dirs['eval']}/precision_result.json", "Verify totals, pass counts and failed cases; prioritize correctness"),
        ("precision_reports", "This round's precision report details", f"{iter_dirs['eval']}/precision_reports/", "For failed cases, check errors, inputs and raw error messages"),
        ("self_test", "Self-test report of the same-numbered development round (if present)", f"{iter_dirs['develop']}/self_test_report.md", "Review the development self-test records; the basis for evaluated code is the separately attached hash binding file"),
    ]:
        if key in scene_input_keys(scene):
            entry = {"key": key, "path": str(Path(path)), "purpose": name, "read_hint": reading}
            if os.path.exists(path):
                default_inputs.append(entry)
                data_lines.append(file_hint(work_dir, path, name, reading).rstrip())
            else:
                missing_inputs.append(entry)

    fail_hint = f"This round\'s failure reason: {fail_reason}\n" if fail_reason else ""

    # Inject verified success experience + failure lessons + the pitfall log for tech_lead reference
    patterns_hint = ((format_proven_patterns_for_prompt(work_dir)
                     + format_regression_patterns_for_prompt(work_dir)
                     + format_pitfalls_for_prompt(work_dir)) if performance_scene
                     else format_pitfalls_for_prompt(work_dir))

    # ── Find the previous round\'s question.md (subordinate feedback); if present and unadjudicated, inject it for tech_lead to adjudicate ──
    question_hint = ""
    question_path_found = None
    for i in range(state.iteration - 1, -1, -1):
        cand = os.path.join(work_dir, "develop", f"iter{i}", "question.md")
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as _qf:
                _qcontent = _qf.read()
            if "## tech_lead adjudication" in _qcontent:
                continue  # already adjudicated; skip
            question_path_found = cand
            question_hint = (
                "\n" + file_hint(work_dir, cand, "Subordinate feedback awaiting adjudication (question.md)",
                                  "Verify the questioned historical suggestion and the hard evidence, then decide confirmed or rejected") +
                f"stage3 cannbot reports that one of your previous-round suggestions was wrong. Adjudicate as a task under current conditions:\n"
                f"Read the file → compare the opinion it cites in history → output the pitfall field (confirmed/rejected).\n"
            )
            break

    # ── Use the externally supplied perf_diff (already computed by the program after stage6)──
    perf_diff_hint = (perf_diff.get("comparison_text", "") + perf_diff.get("diff_text", "")
                      + perf_diff.get("regression_text", ""))
    if consultation:
        perf_diff_hint = "\nPerformance change computed by the program (consultation reference only): " + _json.dumps(
            {key: value for key, value in perf_diff.items() if not key.endswith("_text")}, ensure_ascii=False) + "\n"
        if question_path_found:
            question_hint = file_hint(work_dir, question_path_found, "Development objection awaiting adjudication (consultation background)",
                                      "Used only as question background this time; the adjudication is formed in the final re-review phase")

    profiling_guide_path_9 = os.path.join(os.path.dirname(roles_dir), "knowledge", "profiling_guide.md")
    profiling_guide_hint_9 = (
        "\n" + file_hint(work_dir, profiling_guide_path_9, "Profiling analysis guide",
                          "Use the five-file analysis process and attention weights to verify bottleneck conclusions",
                          base_dir=Path(__file__).parent, base_label="project root") +
        "Read the guide above when interpreting profiler data. Follow its 50%/20%/15%/10%/5% analysis attention allocation, "
        "with kernel_details.csv receiving 50% for redundant kernels, saturated cores and compilation downgrades. "
        "Then use operator statistics, compute/idle records, API calls and the timeline to verify "
        "kernel attribution, resource use and dependencies against the evaluated code.\n"
    ) if scene == "all_passed" and os.path.exists(profiling_guide_path_9) else ""

    decision_output_hint = file_hint(
        work_dir, decision_path, "This round's Stage9 decision output (to be generated by you)",
        "Write iteration, request_id, ledger_entry and model conclusions per the role; use the common proven_pattern/regression_pattern/pitfall fields for experience")
    prompt = (
        f"Working directory: {work_dir}\nOperator: {op_name}\n"
        + file_hint(work_dir, Path(work_dir) / 'task', "Read-only operator requirements and evaluation benchmark",
                    "Judge against desc.md, proto.yaml, cases.yaml and golden.py whether suggestions preserve semantics and complete case coverage")
        + file_hint(work_dir, Path(work_dir) / 'ANALYSIS.md', "Stage1 requirements analysis",
                    "Verify interfaces, precision, hardware constraints and implementation difficulties; avoid deviating from requirements")
        + (file_hint(work_dir, Path(roles_dir).parent / 'knowledge/anti_cheat_reference.md', "Anti-cheating verdict reference",
                    "Use the actual error handling table to compare error codes, kernel CSV and real code to locate violation or zero-score causes",
                    base_dir=Path(roles_dir).parent, base_label="project root") if scene == "evaluation_error" or performance_scene else "")
        + file_hint(work_dir, Path(work_dir) / 'impl', "Current code", "Verify the entry point and related implementation against this scene's question")
        + file_hint(work_dir, history_path, "Cross-round experience, performance rounds and modification ledger",
                    ("Read-only: review historical directions and measurements related to this question; do not submit the ledger now" if consultation else
                     "Read-only: read suggest_next, insights, ledger and rounds first; submit only the current round's ledger with this decision; the program merges history")) +
        f"{device_info_prompt}\n"
        f"{fail_hint}"
        f"{patterns_hint}"
        f"{perf_diff_hint}"
        f"{question_hint}"
        + "\n".join(data_lines) + "\n"
        + ("Materials missing for the current scene (cannot be treated as finished conclusions):\n" + "".join(
            file_hint(work_dir, item["path"], item["purpose"] + " (missing)", "Not generated or unavailable; state clearly that evidence is insufficient and do not substitute old conclusions")
            for item in missing_inputs if item["key"] != "self_test") if missing_inputs else "")
        + design_hint
        + profiling_guide_hint_9
        + ("Analyze this round's results and review previous rounds' design rationale;" if performance_scene
           else "Locate this round's failure and review the design and previous-round suggestions related to this question;")
        + f"read {history_path} read-only; do not overwrite it.\n"
        + f"Stage9 output file: {decision_path}\n"
        + f"Stage9 request ID: {request_id}\n"
        + f"Stage9 current iteration: {state.iteration}\n"
        + decision_output_hint
    )
    if performance_scene and not consultation:
        prompt += ("\nThis round's ledger_entry.case_analysis must cover each of the following case_ids one by one (copy verbatim, no abbreviation; "
                   "for each case write observation, explanation, evidence, next_action; unproven root causes must be marked as pending verification):\n"
                   + _json.dumps(performance_case_ids, ensure_ascii=False) + "\n")
    if state.stage9_context.get("scope_review_error") and not consultation:
        prompt += ("\nThe pre-delivery check in Stage3 found the old plan lacks a clear file scope or has conflicts and was not executed; "
                   "please resubmit this round's complete decision and resolve the following problems:\n"
                   + state.stage9_context["scope_review_error"] + "\n")

    if performance_scene:
        prompt += format_fusion_library_for_prompt(work_dir, "stage9")
    else:
        prompt += file_hint(work_dir, fusion_library_path(work_dir), "Initial fusion candidate library (consult on demand only)", "Fix build or correctness first this time; when a human direction involves fusion methods, check probabilities and prerequisites")
    prompt += format_evidence_for_prompt(work_dir)
    if performance_scene:
        prompt += format_selection_for_prompt(work_dir, comparison_context=selection_context)
        prompt += regression_action_prompt(work_dir, state.iteration)
        event = state.stage9_context.get("semantic_event")
        if event:
            prompt += (f"\nProgram record: this is entry {event['entry_count']} into the {event['scene']} stagnation scene; "
                       "this is the cumulative count across triggers, independent of the human consultation's three-count.\n")
            prompt += file_hint(work_dir, event["state_path"], "Semantic stagnation trigger and count record",
                                "Verify the current iteration, comparison group, window and entry_count; resuming the same round does not double count")
    # Read the persisted window here too, so a Stage9 resume gets the same review
    # instructions without counting another evaluation or changing failure routes.
    selection_status = load_selection_status(work_dir, comparison_context=selection_context)
    review_hint = (fusion_review_hint(selection_status, state.iteration)
                   if fail_reason == "perf_optimize" else "")
    if review_hint:
        prompt += "\n\n" + review_hint + "\n"
        log.warning("[Stage9 prompt addition]\n%s", review_hint)
        evidence = load_evidence(work_dir)
        sources = [
            ("Stage9 role", role), ("Requirements analysis", f"{work_dir}/ANALYSIS.md"),
            ("task", f"{work_dir}/task"), ("Historical experience", history_path),
            ("Hardware source", f"{work_dir}/device_info.json"),
            ("Anti-cheating reference", os.path.join(os.path.dirname(roles_dir), "knowledge", "anti_cheat_reference.md")),
            ("Success experience", f"{work_dir}/knowledge/proven_patterns.md"),
            ("Regression lessons", f"{work_dir}/knowledge/regression_patterns.md"),
            ("Pitfall log", f"{work_dir}/knowledge/tech_lead_pitfalls.md"),
            ("Current code and evidence binding", f"{work_dir}/selection/current_implementation.json"),
            ("Semantic window and history index", f"{work_dir}/selection/state.json"),
            ("Historical evaluation snapshots and case trend source", f"{work_dir}/selection/records"),
            ("Profiling guide", profiling_guide_path_9),
        ]
        sources += [(f"current implementation evidence/{name}", path)
                    for name, path in evidence.get("evidence_paths", {}).items()]
        best = selection_status["best"]
        sources += [(f"best implementation/{name}", best[name]) for name in
                    ("manifest_path", "implementation_dir", "performance_report", "precision_report")]
        if question_path_found:
            sources.append(("question awaiting adjudication", question_path_found))
        input_lines = data_lines + all_designs + [f"{name}: {Path(path)}" for name, path in sources
                                                 if os.path.exists(path)]
        log.info("[Stage9 fusion review inputs and evidence sources]\n%s", "\n".join(input_lines))
    allow_empty = bool(selection_status.get("eligible") and selection_status.get("should_exit")
                       and selection_status.get("latest_iteration") == state.iteration
                       and fail_reason.startswith("perf_pass"))
    if allow_empty and not consultation:
        prompt += "\nThe program has confirmed this round\'s semantic exit; suggest_next may be []; just complete the experience records.\n"
    if not consultation:
        prompt += human_prompt(work_dir, human_messages or [], bundle_path,
                               ending_reason=state.stopped_by or ("semantic_stagnation" if allow_empty else None))
    phase = "consultation" if consultation else ("feedback" if bundle_path else "decision")
    contract_paths, contract_hint = {}, ""
    if not consultation:
        contract_paths, contract_hint = _write_stage9_plan_contract(
            work_dir, request_dir, state.iteration, request_id, performance_case_ids,
            allow_empty, case_catalog, perf_diff=perf_diff, has_question=bool(question_path_found))
        prompt += "\nProgram-defined filling format (read first, then fill):\n" + contract_hint
        prompt += file_hint(
            work_dir, implementation_base, "Code base on which this round's suggestions will actually be implemented (read-only reference)",
            "Verify case routing and functions to change here; changes still use impl/... relative paths as implemented. If this round restores the best version, do not invent modification locations from the regressed code")
    atomic_write_text(role, build_scene_role(
        roles_dir, scene, phase=phase, perf_diff=perf_diff,
        has_question=bool(question_path_found), has_human=bool(human_messages)))
    if consultation:
        # A question is not a development directive and cannot mutate the ledger.
        question_output = Path(consultation["directory"]) / "question.json"
        # Retain version-bound implementation/fusion/selection evidence that follows
        # the normal output lines; remove only the final-decision contract itself.
        prompt = "\n".join(line for line in prompt.splitlines()
                           if not line.startswith(("Stage9 output file:", "Stage9 request ID:", "Stage9 current iteration:"))
                           and str(decision_path) not in line)
        prompt += (f"\nStage9 consultation output file: {question_output}\n"
                   f"Stage9 consultation request ID: {consultation['request_id']}\n"
                   f"The three stagnation rounds recorded by the program: {consultation['trigger_iterations']}\n"
                   "Output only the question this time; submitting a final decision or rewriting history is forbidden. Do not execute Stage3.\n"
                   + file_hint(work_dir, consultation["evidence_manifest_path"], "Version snapshot and missing-item list of the consultation context",
                               "Review the implementation, best scores, slow case trends and historical attempts by purpose and reading hint; cite concrete evidence paths in the question")
                   + "Existing proactive feedback: " + _json.dumps(human_messages or [], ensure_ascii=False))
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
    log.info("[Stage9 scene] iter=%s scene=%s phase=%s request=%s consultation=%s human_ids=%s; role=%s; prompt=%s; input list=%s",
             state.iteration, scene, phase, request_id,
             consultation["request_id"] if consultation else state.stage9_context.get("consultation_id", "-"),
             [item["id"] for item in human_messages or []], role, request_dir / "prompt.md", request_dir / "request.json")
    output_path = question_output if consultation else decision_path
    log.info("[%s] launching tech_lead, phase=%s, output=%s", stage, phase, output_path)
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
        log.warning("[Stage9 report protection] iter=%s reverted the agent's direct rewrite of the analysis report; only decision.json is accepted: %s",
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
                raise RuntimeError("Stage9 agent did not complete successfully")
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
                    raise ValueError("Stage9 output does not belong to this request")
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
                next_action = (f"in-round correction {attempt + 1}/2, without increasing performance iterations" if will_retry
                               else "stop at Stage9; do not enter Stage3")
                failure_message = (
                    "===== Stage9 decision validation failed: iter=%s scene=%s request=%s attempts=%s/3 corrections=%s/2; "
                    "reason=%s; next step=%s; decision=%s; error record=%s =====")
                failure_args = (state.iteration, scene, request_id, attempt + 1, attempt, exc,
                                next_action, decision_path, error_path)
                log.warning(failure_message, *failure_args)
                state_log.warning(failure_message, *failure_args)
                if not will_retry:
                    stopped_message = (
                        "===== Stage9 correction failed; delivery stopped ===== iter=%s request=%s; not entering Stage3; "
                        "original history restored; reason=%s; decision=%s; error record=%s")
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
                    work_dir, decision_path, "This round's Stage9 decision output (to be generated by you)",
                    "Write iteration, request_id, ledger_entry and model conclusions per the role; use the common proven_pattern/regression_pattern/pitfall fields for experience")
                prompt = base_prompt.replace(base_output_hint, new_output_hint)
                contract_paths, new_contract_hint = _write_stage9_plan_contract(
                    work_dir, request_dir, state.iteration, request_id, performance_case_ids,
                    allow_empty, case_catalog, perf_diff=perf_diff, has_question=bool(question_path_found))
                prompt = prompt.replace(base_contract_hint, new_contract_hint)
                prompt = prompt.replace(str(base_output_path), str(decision_path)).replace(
                    f"Stage9 request ID: {root_request_id}", f"Stage9 request ID: {request_id}")
                prompt += (
                    f"\n===== In-round Stage9 decision correction (correction {attempt + 1}/2, overall attempt {attempt + 2}/3) =====\n"
                    "The previous output failed the program's checks; it was not handed to Stage3 nor written into experience."
                    "Fix the complete decision; do not output only a patch; keep the current iteration, original scene and all human feedback.\n"
                    + file_hint(work_dir, error_path, "Validation errors of the previous decision", "Address each full field path and case ID in errors; also recheck this round's complete schema; do not fix only the first item")
                    + file_hint(work_dir, previous_output, "Rejected original decision (if present)", "Read-only reference; it cannot be reused directly; this time write a new request ID and output file")
                    + f"Program rejection reason: {exc}\n"
                    "When a file must be modified, it must be written as a modify/create step in changes and correspond to the target case's implementation_files, "
                    "and it must not be in ledger.readonly_files; the writable list is aggregated by the program; do not fill in modify_files/inspect_files yourself."
                    "If a file must stay read-only, adjust the scheme or arrange an inspect task; do not disguise modifications as inspections.\n"
                    "File-check correction: in changes.file, inspect must also name an actual file, not a directory like prof_data."
                    "You may browse the directory first to locate the corresponding report and then list paths; inspect tasks with case_scope=cases must include a check of the bound implementation files per case."
                    "Fix both task bindings and file type issues together per the errors list; do not guess nonexistent report paths.\n"
                )
                if any(perf_diff.get(flag) for flag in ("has_improvement", "has_regression")):
                    prompt += (
                        "Performance experience correction: verify all required fields of proven_pattern/regression_pattern against this schema/template."
                        "applicability must be non-empty text; each case_analysis needs its own explanation."
                        "The explanations in ledger_entry.case_analysis and the overall why_it_worked/why_it_failed cannot replace the per-case explanations in the experience."
                        "When the root cause is uncertain, honestly mark it as speculation or pending verification and state the verification method; do not fabricate causes or drop cases to bypass checks.\n"
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
                    "===== Stage9 in-round correction ===== iter=%s scene=%s; attempts=%s/3 corrections=%s/2, without increasing performance iterations; "
                    "original request=%s; new request=%s; error record=%s; original decision=%s; prompt=%s; input list=%s; output=%s")
                retry_args = (state.iteration, scene, attempt + 2, attempt + 1, previous_id, request_id, error_path,
                              previous_output, request_dir / "prompt.md", request_dir / "request.json", decision_path)
                log.info(retry_message, *retry_args)
                state_log.info(retry_message, *retry_args)
                continue
            log.info("[Stage9 decision validation passed] iter=%s request=%s attempts=%s/3 corrections=%s/2; v2 tasks=%s; writable=%s; read-only=%s; decision=%s",
                     state.iteration, request_id, attempt + 1, attempt, [item["task_id"] for item in _h["suggest_next"]],
                     plan_ledger["modify_files"], plan_ledger["readonly_files"], decision_path)
            state_log.info("[Stage9 decision validation passed] iter=%s request=%s attempts=%s/3 corrections=%s/2; decision=%s",
                           state.iteration, request_id, attempt + 1, attempt, decision_path)
            break
    except BaseException as exc:
        # Restore the full pre-call state even if the agent accidentally overwrote it.
        save_history(work_dir, history_before)
        log.error("[%s] Stage9 output failed the acceptance check; history restored; phase=%s, this output=%s", stage, phase, output_path)
        if isinstance(exc, Exception):
            raise RuntimeError(f"Stage9 decision acceptance failed: {exc}") from exc
        raise

    # The agent owns only decision.json. Restore accidental edits before any further IO.
    save_history(work_dir, history_before)
    if scene == "all_passed":
        # Detect competing reports before publishing history/knowledge. The
        # Markdown itself is derived only after the decision is committed.
        try:
            validate_profile_report_directory(work_dir, state.iteration)
        except (OSError, ValueError) as exc:
            log.error("[performance analysis report] iter=%s source=Stage9; directory check failed; not delivering: %s; decision=%s",
                      state.iteration, exc, decision_path)
            raise RuntimeError(f"Stage9 analysis report directory unusable: {exc}") from exc
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
        ("has_improvement", "proven_pattern", append_proven_pattern, "proven_patterns.md", "success experience"),
        ("has_regression", "regression_pattern", append_regression_pattern, "regression_patterns.md", "regression lesson"),
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
                            detail=f"iter{perf_diff['prev_iter']}→iter{state.iteration}; "
                                   f"avg_speedup={perf_diff['prev_avg_speedup']}→{perf_diff['curr_avg_speedup']}; "
                                   f"change={perf_diff['delta_pct']}%")

    if question_path_found:
        pitfall = copy.deepcopy(decision["pitfall"])
        pitfall["question_path"] = question_path_found
        pitfall["decision_path"] = str(decision_path)
        pitfall["environment"] = copy.deepcopy(review_environment)
        append_pitfall(work_dir, state.iteration, pitfall)
        verdict = pitfall["verdict"]
        question_path = Path(question_path_found)
        question_text = question_path.read_text(encoding="utf-8")
        if "## tech_lead adjudication" not in question_text:
            question_text += (
                f"\n\n## tech_lead adjudication (iter{state.iteration})\n"
                f"- Verdict: {'✅ misjudgment confirmed' if verdict == 'confirmed' else '❌ rejected'}\n"
                f"- Reason: {pitfall['root_cause']}\n"
                f"- Correct {'practice' if verdict == 'confirmed' else 'understanding'}: {pitfall['correct_approach']}\n")
            atomic_write_text(str(question_path), question_text)
        append_work_record(work_dir, f"{stage} adjudicated question.md: {verdict}")
        log_knowledge_write(log, state_log, label="pitfall log", path=Path(work_dir) / "knowledge/tech_lead_pitfalls.md",
                            iteration=state.iteration, environment=review_environment, decision_path=decision_path,
                            detail=f"verdict={verdict}; topic={pitfall['topic']}; question={question_path_found}")

    save_history(work_dir, _h)
    atomic_write_json(str(request_dir / "commit.json"), {
        "iteration": state.iteration, "request_id": request_id,
        "decision_path": str(decision_path), "history_path": history_path, "committed": True,
    })
    if scene == "all_passed":
        try:
            report_path = write_stage9_report(work_dir, state.iteration, decision, decision_path)
        except (OSError, ValueError) as exc:
            log.error("[performance analysis report] iter=%s source=Stage9; decision accepted but saving the report failed: %s; decision=%s",
                      state.iteration, exc, decision_path)
            raise RuntimeError(f"Stage9 analysis report save failed: {exc}") from exc
        _log_profile_report(log, state_log, state.iteration, 9, report_path, decision_path)
    if performance_scene:
        log.info("[history analysis] iter=%s saved this round's conclusions for %s slow cases, case_ids=%s; history=%s; decision=%s",
                 state.iteration, len(current_ledger.get("case_analysis", [])), performance_case_ids,
                 history_path, decision_path)
    log_knowledge_write(log, state_log, label="history ledger", path=history_path, iteration=state.iteration,
                        environment=current_ledger["environment"], decision_path=decision_path,
                        detail=f"The program merged this round's ledger; keeping {len(_h['ledger'])} rounds of ledger")
    if human_messages:
        log.info("[human adjudication] iter=%s request=%s responses=%s P0_ids=%s; decision=%s; history=%s",
                 state.iteration, request_id,
                 [{"id": item["message_id"], "kind": item["kind"]} for item in decision["human_responses"]],
                 [item["human_message_id"] for item in decision["suggest_next"] if item.get("source") == "human"],
                 decision_path, history_path)

    history_log = logging.getLogger("triton-ascend-history")
    if history_log.handlers:
        history_log.info(f"ITER{state.iteration} [stage9 tech_lead update({fail_reason})]\n{_json.dumps(_h, ensure_ascii=False, indent=2)}")
    log.info(f"[{stage}] history.json merged with the Stage9 decision by the program")
    return str(decision_path)


def _log_profile_report(log, state_log, iteration, source_stage, path, decision_path=None):
    message = "[performance analysis report] iter=%s source=Stage%s; this round\'s only report=%s; decision source=%s"
    args = (iteration, source_stage, path, decision_path or "Stage7 analysis body")
    log.info(message, *args)
    if state_log is not None:
        state_log.info(message, *args)


def _restore_stage9_profile(work_dir, iteration, ledger):
    """Rebuild a report from the accepted decision, never the live implementation."""
    root = Path(work_dir).resolve()
    source = ledger.get("stage9_decision_path")
    if not source:
        raise ValueError("The current ledger lacks the accepted Stage9 decision path; the analysis report cannot be fabricated")
    decision_path = Path(source)
    if not decision_path.is_absolute():
        decision_path = root / decision_path
    if (not decision_path.resolve().is_relative_to(root / "knowledge" / "stage9" / f"iter{iteration}")
            or decision_path.name != "decision.json"):
        raise ValueError("The analysis report source must be this round\'s accepted Stage9 decision.json")
    decision = _json_module.loads(decision_path.read_text(encoding="utf-8-sig"))
    commit = _json_module.loads((decision_path.parent / "commit.json").read_text(encoding="utf-8-sig"))
    if (not isinstance(commit, dict) or not isinstance(decision, dict)
            or commit.get("committed") is not True or commit.get("iteration") != iteration
            or decision.get("iteration") != iteration or ledger.get("iter") != iteration
            or not decision.get("request_id") or commit.get("request_id") != decision["request_id"]
            or commit.get("decision_path") != str(decision_path)):
        raise ValueError("The analysis report source\'s Stage9 commit record does not match the current iteration/request")
    authored = decision.get("ledger_entry")
    if not isinstance(authored, dict) or any(
            authored.get(field) != ledger.get(field)
            for field in ("evaluation_summary", "case_analysis")):
        raise ValueError("The analysis report source does not match the accepted ledger; versions cannot be mixed")
    # Render the accepted task view checked by current_scope, not an unrelated
    # live suggestion list or source files that Stage3 may already have edited.
    decision = dict(decision, suggest_next=ledger["action_plan"]["tasks"])
    return write_stage9_report(work_dir, iteration, decision, decision_path)


def self_goto_stage3(log, roles_dir: str, work_dir: str, op_name: str, state: State, reason: str, state_log=None, node_logs=None, iter_dirs=None, extra_context: str = "", device_info_prompt: str = "", selection_context=None):
    """
    Invokes stage3 (cannbot modification), injecting different file paths into the prompt based on reason.
    reason: build_fail / precision_fail / perf_optimize / score_zero
    """
    stage = f"ITER{state.iteration}-Stage3-modify({reason})"
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
            raise ValueError("Stage3 lacks executable suggestions for this round")
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
        log.warning("[Stage3 delivery intercept] iter=%s file scope failed the check; returning to the same round's Stage9: %s; history=%s",
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

    # Back up the current impl to operator_iter/iter{N}/ (for code tracing; not shown to the agent)
    import shutil
    backup_dir = os.path.join(work_dir, "operator_iter", f"iter{state.iteration}")
    impl_dir = os.path.join(work_dir, "impl")
    if os.path.isdir(impl_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(impl_dir, backup_dir)
        log.info(f"[{stage}] impl backed up to {backup_dir}")

    append_work_record(work_dir, stage)
    transition(state, "stage3", state_log, reason=f"reverting to modify code({reason})")

    base_prompt = (
        f"Working directory: {work_dir}\nOperator: {op_name}\n"
        + file_hint(work_dir, Path(work_dir) / 'impl', "Current operator implementation project",
                    "Inspect the actual source files within the Tech Lead's allowed scope; locate the entry point, tiling and dataflow")
        + file_hint(work_dir, Path(work_dir) / 'task', "Read-only requirements, cases and reference implementation",
                    "Verify against desc.md, proto.yaml, cases.yaml and golden.py that fixes preserve semantics and complete coverage")
        + file_hint(work_dir, Path(work_dir) / 'ANALYSIS.md', "Stage1 requirements analysis",
                    "Verify interfaces, precision, shapes and hardware constraints first, then judge the optimization direction")
        + f"{device_info_prompt}\nRevert reason: {reason}\n"
    )
    base_prompt += format_fusion_library_for_prompt(work_dir, "stage3")
    base_prompt += format_evidence_for_prompt(work_dir)
    base_prompt += format_selection_for_prompt(work_dir, comparison_context=selection_context)
    base_prompt += regression_action_prompt(work_dir, state.iteration)
    base_prompt += development_prompt(work_dir, iter_dirs["develop"], 3)
    base_prompt += development_human_prompt(work_dir, human_directions, receipt_path)
    if current_plan_ledger.get("stage9_decision_path"):
        base_prompt += file_hint(
            work_dir, current_plan_ledger["stage9_decision_path"], "Stage9's validated v2 task list for this round (original)",
            "Look up by task_id the target case, corresponding code evidence, per-file methods in changes and acceptance requirements; the readable task list below is generated from the same JSON")
    log.info("[Stage3 task handoff] iter=%s tasks=%s; writable=%s; read-only=%s; source=%s",
             state.iteration, [task["task_id"] for task in current_tasks],
             current_plan_ledger["modify_files"], current_plan_ledger["readonly_files"],
             current_plan_ledger.get("stage9_decision_path"))

    # Inject the pitfall log (may not exist; empty if absent)
    pitfalls_text = format_pitfalls_for_prompt(work_dir)
    if pitfalls_text:
        base_prompt += f"\n{pitfalls_text}\n"

    # question.md feedback channel note (write only when tech_lead\'s guidance is confirmed wrong)
    base_prompt += (
        f"\n📮 If during implementation you find tech_lead's suggestion is genuinely infeasible at the hardware/framework level (with hard evidence), "
        f"you may write feedback to {iter_dirs['develop']}/question.md (strict trigger conditions are in your role; "
        f"check the pitfall log first to avoid duplicates; this is not grounds for refusing to execute — you must still complete this round's task with an alternative scheme).\n"
    )

    # Extract tech_lead's file constraint directive from the last ledger entry
    fix_plan_text = get_latest_fix_plan(work_dir, current_iteration=state.iteration)
    if fix_plan_text:
        base_prompt += f"\n{fix_plan_text}\n\n"

    if reason == "build_fail":
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['build']) / 'build.log', "This round's build log",
                      "Check the first real build/install error and file line number first; avoid fixing only downstream follow-on errors") +
            f"Please fix the code according to the build errors.\n"
            f"Write the self-test report to {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== Historical experience ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['build']}/build.log"]
    elif reason == "precision_fail":
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['eval']) / 'precision_result.json', "This round's formal precision result",
                      "Look at failed cases, error summaries and pass counts first")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'precision_reports', "This round's raw precision details",
                        "For failed cases, check errors and raw error messages and compare with golden.py to locate semantic problems") +
            f"Please fix the code according to the precision failure information.\n"
            f"Write the self-test report to {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== Historical experience ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/precision_result.json"]
    elif reason == "score_zero":
        # anti-cheat/zero score: inject history experience + score_error info so cannbot knows the root cause
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            f"⚠️ cann-bench performance results are abnormal or zero; verify the real cause first; a missing report does not prove a violation.\n"
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "Zero-score/anti-cheat performance result",
                        "Look at score_error_code and kernel event evidence first and locate the cause per the anti-cheating rules") +
            f"{extra_context}\n"
            f"If the evidence shows no valid NPU kernel events, investigate the following causes; fix the evaluation pipeline first for collection/reporting problems:\n"
            f"1. The operator actually took a CPU fallback instead of executing on the NPU\n"
            f"2. impl/build/ still contains old build artifacts, or the cann_bench package installed in the environment is not this round's version\n"
            f"3. Triton JIT was not triggered correctly (check the @triton.jit, kernel[grid](...) calls and import paths)\n"
            f"Investigate thoroughly and fix so the operator truly executes on the NPU.\n"
            f"If old build artifacts are confirmed, clean this project's impl/build/ first, then run in impl/: "
            f"python3 -m pip install . --force-reinstall --no-deps; "
            f"return to the work directory to verify cann_bench's import location, ensuring this round's installed package loads.\n"
            f"Write the design rationale to {iter_dirs['develop']}/design_rationale.md\n"
            f"Write the self-test report to {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== Historical experience (from previous iterations) ===\n{history_context}\n"
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/perf_result.json"]
    elif reason == "perf_pass_optimize":
        # Target met, but the program\'s x valid-performance window has not yet satisfied the exit condition.
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            f"Performance target met; continue optimizing based on the bottlenecks and the Tech Lead's P0/P1/P2 feedback.\n"
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "This round's formal performance summary",
                        "Verify target attainment and remaining bottlenecks per case; keep all cases from regressing")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "This round's raw performance reports",
                        "Check real timing and kernel evidence per case; confirm optimization gains with a new evaluation")
            + file_hint(work_dir, profile_report_path(work_dir, state.iteration), "Stage9 bottleneck analysis report for this round",
                        "Verify the report's source and round first, then read each case's symptoms, causes, evidence and next steps; the execution scope still follows the accepted task list") +
            f"Write the design rationale to {iter_dirs['develop']}/design_rationale.md\n"
            f"Write the self-test report to {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== Historical experience (from tech_lead) ===\n{history_context}\n\n"
            f"Improve overall performance while keeping correctness and all cases passing; the current fusion scheme may be kept.\n"
            f"Explain this round's dataflow and selection rationale; validate gains with a new evaluation, and do not judge by kernel count alone."
        )
        inputs = [f"{work_dir}/impl/", f"{iter_dirs['eval']}/perf_result.json",
                  str(profile_report_path(work_dir, state.iteration))]
    else:  # perf_optimize
        # Inject cross-round memory (insights/ledger/trends)
        history_context = format_for_prompt(work_dir, current_iteration=state.iteration)
        prompt = base_prompt + (
            file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_result.json', "This round's formal performance summary",
                      "Look at avg_speedup, per-case speedup and worst_6_cases first to define the optimization target")
            + file_hint(work_dir, Path(iter_dirs['eval']) / 'perf_reports', "This round's raw performance reports",
                        "Verify raw timing and kernel/profiler evidence per slow case")
            + file_hint(work_dir, Path(iter_dirs['profile']) / 'bottleneck_analysis.md', "Stage7 bottleneck analysis",
                        "Read per-case root causes and common bottlenecks first; separate measured conclusions from hypotheses")
            + file_hint(work_dir, Path(iter_dirs['search']) / 'SEARCH_REPORT.md', "Stage8 search schemes and sources",
                        "Verify official sources, hardware prerequisites, implementation paths and risks")
            + file_hint(work_dir, Path(iter_dirs['search']) / 'FIX_DIRECTIVE.md', "Stage8 concrete modification directives",
                        "Implement together with the Tech Lead's latest P0/P1/P2 and the file scope; on conflict, the Tech Lead's ruling prevails") +
            f"Write the design rationale to {iter_dirs['develop']}/design_rationale.md\n"
            f"Write the self-test report to {iter_dirs['develop']}/self_test_report.md\n"
            f"\n=== Historical experience (from tech_lead, must follow) ===\n{history_context}\n\n"
            f"Optimize the code according to the modification directives, focusing on worst_6_cases bottlenecks.\n"
            f"The ledger verdict only evaluates implementations already measured; direction is the next-step plan proposed after evaluation."
            f"Do not ban new plans merely because the mean is flat or regressed; execute the latest suggestions using per-case trends, conditions and evidence."
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
        log.warning(f"[{stage}] Scheme/self-test material does not yet satisfy the best-record conditions: {evidence.get('reason', '')}")
    if not ok:
        log.error(f"[{stage}] cannbot failed (returncode!=0)")
        if human_directions:
            raise RuntimeError("Stage3 failed; human P0 not yet executed; original round kept for resume")
    if human_directions:
        try:
            receipts = validate_execution_receipt(receipt_path, human_directions)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Stage3 lacks the implementation receipt for this round\'s human feedback; original round kept for resume") from exc
        for receipt in receipts:
            rationale_path = Path(iter_dirs["develop"]) / "fusion_scheme_rationale.md"
            if receipt["message_id"] not in read_file_safe(str(rationale_path)):
                raise RuntimeError("Stage3\'s scheme-selection rationale does not reference the corresponding human feedback ID; original round kept for resume")
            if receipt["status"] == "implemented" and evidence.get("eligible"):
                human.mark_executed([receipt["message_id"]], state.iteration, str(receipt_path))
            else:
                human.mark_unexecuted([receipt["message_id"]], receipt["details"] if receipt["status"] != "implemented"
                                      else "The implementation self-reports modification, but this round\'s code binding/self-test evidence is incomplete; execution not yet confirmed")
        log.info("[human P0] This round\'s execution receipt=%s; actual performance gains are still subject to the subsequent formal evaluation", receipt_path)
    state.stage9_context["phase"] = "delivered"
    state.flush()


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()

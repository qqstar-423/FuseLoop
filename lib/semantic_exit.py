"""Select immutable, measured implementations and evaluate semantic exit windows.

Only Stage6 calls ``record_evaluation``.  Jev scores and agent prose are evidence,
never performance measurements.  Failed evaluations do not advance a window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from decimal import Decimal
from typing import Any


SCHEMA_VERSION = 1
IMPROVEMENT_THRESHOLD = 0.05
log = logging.getLogger("triton-ascend-workflow")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".selection-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _windows(config: dict | None) -> tuple[int, int]:
    settings = config or {}
    if "workflow" in settings:
        settings = settings["workflow"]
    if "semantic_exit" in settings:
        settings = settings["semantic_exit"]
    values = tuple(settings.get(key, 3) for key in ("passed_window", "underperforming_window"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("Semantic exit windows must be positive integers")
    return values


def _positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _case_baselines(cases: list[dict]) -> dict:
    """Extract validated baseline values, independent of case ordering."""
    baselines = {}
    for case in sorted(cases, key=lambda item: item["case_id"]):
        value = case.get("baseline_perf_us", case.get("baseline_us"))
        # JSON's 100 and 100.0 describe the same measured baseline. Preserve
        # exact integer values while giving those representations one group.
        baselines[case["case_id"]] = int(value) if value == int(value) else value
    return baselines


def _implementation_hash(path: Path) -> str:
    # Shared with the evidence binder; no circular dependency on this module.
    from .fusion_evidence import implementation_hash
    return implementation_hash(str(path))


def _empty(reason: str = "No valid Stage6 evaluation has been recorded") -> dict:
    return {"eligible": False, "reason": reason, "should_exit": False,
            "review_fusion": False, "best": None, "case_trends": [], "window": None}


def _validate(perf: dict, precision: dict, evidence: dict, impl: Path) -> str:
    if precision.get("precision_overall") is not True or precision.get("_error"):
        return "Precision validation has not passed"
    total = perf.get("total_cases")
    cases = perf.get("cases")
    if (isinstance(total, bool) or not isinstance(total, int) or total < 1
            or not isinstance(cases, list) or len(cases) != total):
        return "Performance result does not contain the complete case set"
    if (precision.get("total_cases") != total or precision.get("passed_cases") != total
            or precision.get("failed_cases", 0) != 0):
        return "Precision and performance case counts do not match"
    if (perf.get("_error") or perf.get("score_error") or perf.get("score_error_code")
            or not _positive(perf.get("avg_speedup"))):
        return "Performance result is invalid or has no positive finite avg_speedup"
    ids = []
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                or not case["case_id"] or case.get("status") != "success"
                or not _positive(case.get("speedup")) or not _positive(case.get("elapsed_us"))
                or not _positive(case.get("baseline_perf_us", case.get("baseline_us")))):
            return "Performance contains a missing, failed, or invalid case"
        ids.append(case["case_id"])
    if len(set(ids)) != len(ids):
        return "Performance case IDs are not unique"
    source = Path(perf.get("source_json") or "")
    if not source.is_file():
        return "Original performance report is missing"
    # Ensure the parsed figures really belong to the report being archived.
    try:
        raw = _read_json(source)
        operators = raw.get("operators", [])
        operator = operators[0] if operators else {}
        raw_cases = {case["case_id"]: case for case in operator.get("cases", [])}
        if (operator.get("total_cases") != total or operator.get("avg_speedup") != perf["avg_speedup"]
                or len(operator.get("cases", [])) != total or set(raw_cases) != set(ids)
                or operator.get("score_error") or operator.get("score_error_code")):
            return "Parsed performance does not match the original report"
        for case in cases:
            raw_case = raw_cases[case["case_id"]]
            if any(raw_case.get(key) != case.get(key) for key in
                   ("status", "speedup", "elapsed_us", "perf_score", "t_hw_us")):
                return "Parsed case metrics do not match the original report"
            baseline = case.get("baseline_perf_us", case.get("baseline_us"))
            if (raw_case.get("baseline_perf_us") != baseline
                    or ("baseline_us" in case and case["baseline_us"] != baseline)):
                return "Parsed baseline timing does not match the original report"
    except (OSError, ValueError, KeyError, TypeError):
        return "Original performance report is unreadable or malformed"
    if evidence.get("eligible") is not True:
        return f"Implementation evidence is not valid: {evidence.get('reason', 'missing self-test evidence')}"
    if not isinstance(evidence.get("implementation_plan"), str) or not evidence["implementation_plan"].strip():
        return "Implementation plan is missing"
    if not isinstance(evidence.get("fusion_scheme"), dict) or not evidence["fusion_scheme"]:
        return "Fusion scheme is missing"
    paths = evidence.get("evidence_paths")
    if not isinstance(paths, dict) or not paths:
        return "Implementation evidence files are missing"
    if any(not isinstance(path, str) or not Path(path).is_file() for path in paths.values()):
        return "An implementation evidence file is missing"
    if not impl.is_dir() or not any(p.is_file() for p in impl.rglob("*")):
        return "Implementation directory is missing or empty"
    try:
        if not evidence.get("impl_sha256") or _implementation_hash(impl) != evidence["impl_sha256"]:
            return "Implementation changed after its evidence was validated"
    except (OSError, ValueError):
        return "Implementation cannot be hashed safely"
    return ""


def _safe_name(name: str) -> str:
    return "".join(c if c.isascii() and (c.isalnum() or c in "-_.") else "_" for c in name)


def _snapshot(work: Path, iteration: int, perf: dict, precision: dict, evidence: dict,
              context: dict, group_id: str, record_id: str, expected_hashes: dict) -> dict:
    root = work / "selection" / "records"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / record_id
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    impl = work / "impl"
    files = {}

    def copied(source: str | Path, relative: str) -> str:
        src = Path(source)
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        files[relative] = _file_hash(target)
        if str(src) in expected_hashes and files[relative] != expected_hashes[str(src)]:
            raise ValueError("Evaluation evidence changed while taking its snapshot")
        return str(destination / relative)

    def written(relative: str, content: dict) -> str:
        target = temporary / relative
        _write_json(target, content)
        files[relative] = _file_hash(target)
        return str(destination / relative)

    try:
        # Copy real bytes, never links to a mutable working implementation.
        if any(p.is_symlink() for p in impl.rglob("*")):
            raise ValueError("Implementation contains symlinks; a complete independent snapshot is required")
        shutil.copytree(impl, temporary / "impl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git"))
        if (_implementation_hash(temporary / "impl") != evidence["impl_sha256"]
                or _implementation_hash(impl) != evidence["impl_sha256"]):
            raise ValueError("Implementation changed while taking its snapshot")
        perf_copy = dict(perf)
        perf_copy["source_json"] = copied(perf["source_json"], "reports/performance_source.json")
        perf_copy["source_md"] = ""
        if perf.get("source_md") and Path(perf["source_md"]).is_file():
            perf_copy["source_md"] = copied(perf["source_md"], "reports/performance_source.md")
        perf_copy["original_source_csv_dir"] = perf.get("source_csv_dir", "")
        perf_copy["source_csv_dir"] = ""
        perf_copy["cases"] = []
        for index, case in enumerate(perf["cases"]):
            archived = dict(case)
            if case.get("kernel_csv") and Path(case["kernel_csv"]).is_file():
                archived["source_kernel_csv"] = case["kernel_csv"]
                archived["kernel_csv"] = copied(case["kernel_csv"], f"reports/case_{index}/kernel_details.csv")
                perf_copy["source_csv_dir"] = str(destination / "reports")
            elif case.get("kernel_csv"):
                archived["source_kernel_csv"] = case["kernel_csv"]
                archived["kernel_csv"] = ""
            perf_copy["cases"].append(archived)
        perf_copy["worst_6_cases"] = sorted(perf_copy["cases"], key=lambda c: c["speedup"])[:6]
        perf_path = written("reports/perf_result.json", perf_copy)
        precision_copy = dict(precision)
        if precision.get("source_report") and Path(precision["source_report"]).is_file():
            precision_copy["source_report"] = copied(precision["source_report"], "reports/precision_source.json")
        precision_path = written("reports/precision_result.json", precision_copy)
        evidence_paths = {}
        for index, (key, source) in enumerate(sorted(evidence["evidence_paths"].items())):
            evidence_paths[key] = copied(source, f"evidence/{index}_{_safe_name(key)}{Path(source).suffix}")
        # Re-check the archived raw report against parsed figures, closing the
        # validation/copy race even when the source changed before hashing.
        archived_perf = {**perf_copy, "source_json": str(temporary / "reports/performance_source.json")}
        archived_evidence = {**evidence, "evidence_paths": {
            key: str(temporary / Path(path).relative_to(destination)) for key, path in evidence_paths.items()}}
        validation_error = _validate(archived_perf, precision_copy, archived_evidence, temporary / "impl")
        if validation_error:
            raise ValueError(validation_error)
        all_passed = all(case["speedup"] >= 1.0 for case in perf["cases"])
        # HAP has no fabricated aggregate: preserve exactly what the tool reports.
        hap = {"performance_score": perf.get("performance_score"), "cases": [
            {"case_id": c["case_id"], **{key: c[key] for key in ("perf_score", "t_hw_us", "op_times") if key in c}}
            for c in perf["cases"]]}
        manifest = {
            "schema_version": SCHEMA_VERSION, "record_id": record_id, "iteration": iteration,
            "group_id": group_id, "comparison_context": context,
            "case_ids": sorted(c["case_id"] for c in perf["cases"]),
            "all_cases_pass": all_passed, "selection_label": "best_passed" if all_passed else "best_available",
            "avg_speedup": perf["avg_speedup"], "avg_speed": perf["avg_speedup"], "hap": hap,
            "implementation_plan": evidence["implementation_plan"], "fusion_scheme": evidence["fusion_scheme"],
            "impl_sha256": evidence["impl_sha256"], "implementation_dir": str(destination / "impl"),
            "original_implementation_dir": str(impl), "manifest_path": str(destination / "manifest.json"),
            "performance_report": perf_copy["source_json"], "original_performance_report": perf["source_json"],
            "performance_result": perf_path, "precision_report": precision_path,
            "perf_result_path": perf_path, "precision_result_path": precision_path,
            "evidence_paths": evidence_paths,
            "case_results": perf_copy["cases"], "files_sha256": files,
        }
        _write_json(temporary / "manifest.json", manifest)
        if temporary.resolve().parent != root.resolve() or destination.resolve().parent != root.resolve():
            raise ValueError("Snapshot directories escaped the selection archive")
        os.replace(temporary, destination)
        log.debug("语义退出：已创建 iter%s 评测快照，目录=%s", iteration, destination)
        return manifest
    finally:
        # This temporary directory is created above, exclusively inside selection/records.
        if temporary.exists():
            if temporary.resolve().parent != root.resolve() or not temporary.name.startswith(".pending-"):
                raise ValueError("Refusing to remove a directory outside the temporary snapshot area")
            shutil.rmtree(temporary)


def _load_records(work: Path, state: dict, group_id: str) -> list[dict]:
    records = []
    group = state.get("groups", {}).get(group_id, {})
    baselines = group.get("case_baselines")
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError("Selection group has no archived case baselines; a fresh evaluation is required")
    if (set(baselines) != set(group.get("case_ids", []))
            or any(not _positive(value) for value in baselines.values())
            or group_id != _digest({"comparison_context": group.get("comparison_context"),
                                    "case_ids": group.get("case_ids"), "case_baselines": baselines})):
        raise ValueError("Selection group baseline identity is invalid")
    root = (work / "selection" / "records").resolve()
    for entry in group.get("records", []):
        record_dir = root / entry["record_id"]
        if record_dir.resolve().parent != root:
            raise ValueError("Selection index contains an invalid record path")
        manifest_path = record_dir / "manifest.json"
        if _file_hash(manifest_path) != entry["manifest_sha256"]:
            raise ValueError("Selection manifest changed after evaluation")
        record = _read_json(manifest_path)
        if _case_baselines(record["case_results"]) != baselines:
            raise ValueError("Archived evaluation baselines do not match their comparison group")
        if _implementation_hash(record_dir / "impl") != record["impl_sha256"]:
            raise ValueError("Archived implementation changed after evaluation")
        for relative, expected in record["files_sha256"].items():
            artifact = record_dir / relative
            if not artifact.resolve().is_relative_to(record_dir.resolve()):
                raise ValueError("Selection manifest contains an invalid artifact path")
            if _file_hash(artifact) != expected:
                raise ValueError("Archived evaluation evidence changed after evaluation")
        records.append(record)
    return sorted(records, key=lambda record: record["iteration"])


def _decision(records: list[dict], config: dict | None, group_id: str) -> dict:
    if not records:
        return _empty()
    passed_window, under_window = _windows(config)
    best = max(records, key=lambda record: (record["all_cases_pass"], record["avg_speedup"]))
    best_by_status = {True: 0.0, False: 0.0}
    running = {}
    for record in records:
        passed = record["all_cases_pass"]
        best_by_status[passed] = max(best_by_status[passed], record["avg_speedup"])
        running[record["record_id"]] = best_by_status[passed]
    # A rerun can contribute another measured best, but never an extra window
    # step for the same iteration. The most recent result is its active sample.
    records = list({record["iteration"]: record for record in records}.values())
    current = records[-1]
    required = passed_window if current["all_cases_pass"] else under_window
    segment = []
    previous_status = None
    for record in records:
        passed = record["all_cases_pass"]
        if passed != previous_status:
            segment = []
        segment.append(record)
        previous_status = passed
    enough = len(segment) >= required + 1
    start = segment[-required - 1] if enough else segment[0]
    start_best = running[start["record_id"]]
    end_best = running[current["record_id"]]
    gain = Decimal(str(end_best)) / Decimal(str(start_best)) - 1
    stagnated = enough and gain < Decimal("0.05")
    window = {"status": "passed" if current["all_cases_pass"] else "underperforming",
              "required_improvements": required, "valid_samples": len(segment),
              "completed_improvements": min(len(segment) - 1, required),
              "enough_samples": enough, "start_iteration": start["iteration"],
              "end_iteration": current["iteration"], "start_best_avg_speedup": start_best,
              "end_best_avg_speedup": end_best, "cumulative_improvement": float(gain),
              "threshold": IMPROVEMENT_THRESHOLD, "stagnated": stagnated}
    trends = []
    for case_id in current["case_ids"]:
        history = []
        for record in records:
            case = next(c for c in record["case_results"] if c["case_id"] == case_id)
            history.append({"iteration": record["iteration"], "speedup": case["speedup"],
                            "gap_to_one": max(0.0, 1.0 - case["speedup"])})
        speedup = history[-1]["speedup"]
        trends.append({"case_id": case_id, "current_speedup": speedup,
                       "currently_underperforming": speedup < 1, "gap_to_one": max(0.0, 1.0 - speedup),
                       "change_from_previous": speedup - history[-2]["speedup"] if len(history) > 1 else None,
                       "history": history})
    return {"eligible": True, "reason": "Valid measured implementation recorded",
            "group_id": group_id, "current_iteration": current["iteration"], "latest_iteration": current["iteration"],
            "should_exit": bool(stagnated and current["all_cases_pass"]),
            "review_fusion": bool(stagnated and not current["all_cases_pass"]),
            "best": best, "current": current, "case_trends": trends, "window": window}


def load_selection_status(work_dir, comparison_context=None, config=None) -> dict:
    """Read selection status without creating files or mixing comparison groups."""
    work = Path(work_dir).resolve()
    state_path = work / "selection" / "state.json"
    if not state_path.is_file():
        return _empty()
    try:
        state = _read_json(state_path)
        group_id = state.get("current_group")
        if comparison_context is not None:
            matching = [key for key, group in state.get("groups", {}).items()
                        if group.get("comparison_context") == comparison_context]
            if group_id not in matching:
                return _empty("No current evaluation matches the requested hardware/task/timing context")
        effective_config = config if config is not None else state.get("semantic_exit", {})
        result = _decision(_load_records(work, state, group_id), effective_config, group_id)
        attempt = state.get("last_attempt", {})
        if attempt.get("eligible") is False:
            result.update(eligible=False, reason=attempt["reason"], should_exit=False, review_fusion=False)
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _empty(f"Selection archive is unavailable or invalid: {exc}")


def record_evaluation(work_dir, iteration, *, perf, precision, evidence,
                      comparison_context, config=None) -> dict:
    """Archive a valid fresh Stage6 result, update best.json, and evaluate x/y.

    A retry of the same evaluation is idempotent. Re-evaluating the same iteration
    replaces its active sample, while retaining both immutable archives on disk.
    No working implementation is restored or overwritten.
    """
    _windows(config)
    work = Path(work_dir).resolve()
    state_path = work / "selection" / "state.json"

    def rejected(reason: str) -> dict:
        log.debug("语义退出：iter%s 未入库，原因=%s", iteration, reason)
        result = load_selection_status(work, comparison_context, config)
        result.update(eligible=False, reason=reason, should_exit=False, review_fusion=False)
        if state_path.is_file():
            try:
                state = _read_json(state_path)
            except (OSError, ValueError):
                pass  # Preserve a corrupt index for diagnosis; never overwrite it.
            else:
                state["last_attempt"] = {"eligible": False, "reason": reason, "iteration": iteration}
                _write_json(state_path, state)
        return result

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if not isinstance(comparison_context, dict) or not comparison_context:
        return rejected("Hardware/task/timing comparison context is missing")
    reason = _validate(perf, precision, evidence, work / "impl")
    if reason:
        return rejected(reason)
    ids = sorted(case["case_id"] for case in perf["cases"])
    baselines = _case_baselines(perf["cases"])
    try:
        group_id = _digest({"comparison_context": comparison_context, "case_ids": ids,
                            "case_baselines": baselines})
        expected_hashes = {str(Path(path)): _file_hash(Path(path)) for path in
                           [perf["source_json"], *evidence["evidence_paths"].values()]}
        fingerprint = _digest({"group_id": group_id, "iteration": iteration,
                               "perf": perf, "precision": precision,
                               "implementation_plan": evidence["implementation_plan"],
                               "fusion_scheme": evidence["fusion_scheme"], "impl_sha256": evidence["impl_sha256"],
                               "report_sha256": expected_hashes[str(Path(perf["source_json"]))],
                               "evidence_sha256": {key: expected_hashes[str(Path(path))] for key, path in evidence["evidence_paths"].items()}})
    except (OSError, ValueError, TypeError) as exc:
        return rejected(f"Evaluation evidence cannot be fingerprinted: {exc}")
    record_id = f"iter{iteration}-{fingerprint[:20]}"
    try:
        state = _read_json(state_path) if state_path.is_file() else {"schema_version": SCHEMA_VERSION, "groups": {}}
        if not isinstance(state.get("groups"), dict):
            raise ValueError("Selection groups are malformed")
    except (OSError, ValueError) as exc:
        return rejected(f"Selection archive is unavailable or invalid: {exc}")
    group = state["groups"].setdefault(group_id, {
        "comparison_context": comparison_context, "case_ids": ids,
        "case_baselines": baselines, "records": [],
    })
    manifest_path = work / "selection" / "records" / record_id / "manifest.json"
    existing = next((entry for entry in group["records"] if entry["record_id"] == record_id), None)
    if existing and (not manifest_path.is_file() or _file_hash(manifest_path) != existing["manifest_sha256"]):
        return rejected("Selection manifest changed after evaluation")
    if not manifest_path.is_file():
        try:
            _snapshot(work, iteration, perf, precision, evidence, comparison_context, group_id, record_id, expected_hashes)
        except (OSError, ValueError) as exc:
            return rejected(f"Unable to archive measured implementation: {exc}")
    else:
        log.debug("语义退出：iter%s 复用已有评测快照，清单=%s", iteration, manifest_path)
    group["records"] = [entry for entry in group["records"] if entry["record_id"] != record_id]
    group["records"].append({"iteration": iteration, "record_id": record_id,
                             "manifest_sha256": _file_hash(manifest_path)})
    state["current_group"] = group_id
    passed_window, under_window = _windows(config)
    state["semantic_exit"] = {"passed_window": passed_window, "underperforming_window": under_window}
    state["last_attempt"] = {"eligible": True, "iteration": iteration, "group_id": group_id}
    try:
        result = _decision(_load_records(work, state, group_id), config, group_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return rejected(f"Selection archive is unavailable or invalid: {exc}")
    _write_json(state_path, state)
    _write_json(work / "selection" / "best.json", result["best"])
    return result


def format_selection_for_prompt(work_dir, comparison_context=None) -> str:
    """Include complete valid case trends and snapshot paths for reviewing agents."""
    from lib.prompt_files import file_hint

    status = load_selection_status(work_dir, comparison_context)
    if status.get("best") is None:
        return "No eligible measured best implementation is available. " + status["reason"]
    # Avoid repeating whole raw metrics and long implementation prose in prompts;
    # archived evidence remains accessible by its explicit path.
    best = status["best"]
    payload = {"eligible": status["eligible"], "reason": status["reason"],
               "should_exit": status["should_exit"], "review_fusion": status["review_fusion"],
               "best": {key: best[key] for key in
                        ("iteration", "selection_label", "all_cases_pass", "avg_speedup", "hap",
                         "implementation_dir", "manifest_path", "performance_report", "precision_report",
                         "fusion_scheme")},
               "window": status["window"], "case_trends": status["case_trends"]}
    descriptions = {
        "implementation_dir": ("最佳已验证实现的独立代码快照目录", "读取该目录的实现，勿把当前工作代码当作历史最佳版本"),
        "manifest_path": ("最佳实现快照清单", "核对 iteration、融合方案、avg_speedup、hap 及代码和评测证据路径"),
        "performance_report": ("最佳实现对应的原始性能报告", "按 case 看 speedup、耗时和 HAP，确认使用同一评测口径"),
        "performance_result": ("最佳实现的结构化性能结果", "看 avg_speedup、cases 和最慢用例，配合窗口和趋势判断"),
        "precision_report": ("最佳实现对应的精度结果", "核对 precision_overall、通过数量和原始精度报告定位"),
    }
    hints = file_hint(work_dir, Path(work_dir) / "selection/state.json", "有效评测索引与语义窗口配置",
                      "以下 window 和 case_trends 由程序按同口径历史快照计算，不将失败轮计入窗口")
    hints += "".join(file_hint(work_dir, best[key], purpose, read_hint)
                    for key, (purpose, read_hint) in descriptions.items() if best.get(key))
    evidence_descriptions = {
        "decision_rationale": ("最佳实现的融合选择依据快照", "看当时的选择理由，与当前方案区别及实测结果核对"),
        "fusion_library": ("最佳实现的当轮融合方案库快照", "看 selection 中实际方法和实现方案，不把初始概率当性能"),
        "self_test_report": ("最佳实现的自测报告快照", "核对给定 case 及连续调用的执行范围与结果"),
        "self_test_result": ("最佳实现的自测资格 JSON 快照", "核对两类自测状态；其中历史日志路径以本清单中的归档日志为准"),
        "selftest_log_provided": ("最佳实现的给定 case 原始自测日志快照", "核对原始执行输出是否支持报告结论"),
        "selftest_log_continuous": ("最佳实现的连续调用原始日志快照", "核对同 shape 更换参数后逐次比较参考实现的结果"),
    }
    hints += "".join(file_hint(work_dir, path, *evidence_descriptions.get(
        key, ("最佳实现的补充证据快照", "与快照清单及性能结果交叉核对")))
        for key, path in best.get("evidence_paths", {}).items())
    return ("Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).\n"
            "Underperforming stagnation requests review, never mandatory fusion replacement.\n"
            + hints
            + json.dumps(payload, ensure_ascii=False, indent=2))

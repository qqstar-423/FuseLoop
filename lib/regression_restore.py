"""Record regression policy, then restore a measured baseline after Stage9 commits.

The evaluated regression remains immutable. Directory swaps are journaled so a
restart can finish a partial restore without overwriting subsequent Stage3 work.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from lib.fusion_evidence import implementation_hash
from lib.handoff import atomic_write_json
from lib.prompt_files import file_hint
from lib.semantic_exit import load_selection_status


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(work, path):
    candidate = Path(path)
    resolved = candidate.resolve()
    if resolved == work or not resolved.is_relative_to(work):
        raise ValueError(f"回退路径必须位于工作目录内：{candidate}")
    # Do not follow directory symlinks/junctions for copy or rename operations.
    for part in (candidate, *candidate.parents):
        if part == work:
            break
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"回退路径不能经过链接：{candidate}")
    return resolved


def _tree(work, path):
    root = _inside(work, path)
    for item in root.rglob("*"):
        _inside(work, item)
    return root


def _notice(log, state_log, text):
    log.warning(text)
    if state_log is not None and state_log is not log:
        state_log.warning(text)


def load_regression_action(work_dir, iteration):
    work = Path(work_dir).resolve()
    path = _inside(work, work / "selection" / "rollbacks" / f"iter{iteration}" / "record.json")
    return _read(path) if path.is_file() else None


def prepare_regression_action(work_dir, iteration, perf_diff, selection, log, state_log):
    """Freeze the branch from this evaluation, never from a historical best's flag."""
    if not perf_diff.get("has_regression"):
        return None
    previous = load_regression_action(work_dir, iteration)
    if previous:
        return previous
    work = Path(work_dir).resolve()
    path = _inside(work, work / "selection" / "rollbacks" / f"iter{iteration}" / "record.json")
    current = selection.get("current") or {}
    best = selection.get("best") or {}
    valid = (selection.get("eligible") is True and selection.get("latest_iteration") == iteration
             and current.get("iteration") == iteration)
    restore = bool(valid and current.get("all_cases_pass") is True
                   and best.get("all_cases_pass") is True)
    if not valid:
        reason = "当前轮缺少有效且版本绑定的评测证据，只记录退步教训，不自动恢复"
    elif not current.get("all_cases_pass"):
        reason = "本轮仍有 case < 1，保留当前实现，只记录退步教训"
    elif not restore:
        reason = "没有同口径的最佳达标快照，只记录退步教训，不自动恢复"
    else:
        reason = "本轮全部 case >= 1，Stage9 记录退步教训后恢复最佳达标实现"
    if restore and implementation_hash(_tree(work, work / "impl")) != current["impl_sha256"]:
        raise ValueError("本轮评测后代码已变动，拒绝覆盖未经保存的修改")
    record = {
        "schema_version": 1, "iteration": iteration,
        "action": "restore_best" if restore else "keep_current",
        "status": "planned" if restore else "recorded", "reason": reason,
        "record_path": str(path), "perf_diff": perf_diff,
        "all_cases_pass": current.get("all_cases_pass") if valid else None,
        "current_record": current, "best_record": best,
        "comparison_context": current.get("comparison_context"),
        "source_iteration": best.get("iteration"),
        "best_implementation_dir": best.get("implementation_dir"),
        "failed_implementation_dir": str(path.parent / "failed_impl"),
        "knowledge_path": str(work / "knowledge" / "regression_patterns.md"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(str(path), record)
    _notice(log, state_log,
            f"===== 性能退步处置：iter{iteration} =====\n"
            f"iter{perf_diff.get('prev_iter')}→iter{iteration}；avg_speedup "
            f"{perf_diff.get('prev_avg_speedup')}→{perf_diff.get('curr_avg_speedup')}；"
            f"变化={perf_diff.get('delta_pct')}%（沿用退步 >= 5% 规则）。\n"
            f"{reason}。\n当前全部达标={record['all_cases_pass']}；"
            f"最佳=iter{best.get('iteration')}，avg_speedup={best.get('avg_speedup')}；"
            f"最佳目录={best.get('implementation_dir')}\n"
            f"退步知识={record['knowledge_path']}；处置记录={path}\n"
            "===== 性能退步处置记录结束 =====")
    return record


def apply_regression_action(work_dir, iteration, decision_path, log, state_log, *, comparison_context=None):
    """Apply a frozen action only after a final Stage9 decision saved its knowledge."""
    work = Path(work_dir).resolve()
    record = load_regression_action(work, iteration)
    if not record or record["status"] == "restored":
        return record
    path = _inside(work, record["record_path"])
    decision = _inside(work, decision_path)
    commit = _read(decision.parent / "commit.json")
    output = _read(decision)
    if (commit.get("committed") is not True or commit.get("iteration") != iteration
            or output.get("iteration") != iteration
            or commit.get("request_id") != output.get("request_id")
            or not output.get("regression_pattern")
            or not _inside(work, record["knowledge_path"]).is_file()):
        raise ValueError("必须先由 Stage9 提交本轮退步教训，再恢复代码")
    if record["action"] == "keep_current":
        record.update(decision_path=str(decision), knowledge_written=True)
        atomic_write_json(str(path), record)
        return record
    if comparison_context is not None and comparison_context != record["comparison_context"]:
        raise ValueError("恢复时硬件或评测口径已变化，拒绝使用旧口径的最佳代码")
    selection = load_selection_status(work, comparison_context=record["comparison_context"])
    best, current = selection.get("best") or {}, selection.get("current") or {}
    if (not selection.get("eligible") or selection.get("latest_iteration") != iteration
            or best.get("record_id") != record["best_record"].get("record_id")
            or current.get("record_id") != record["current_record"].get("record_id")):
        raise ValueError("最佳或本轮评测快照失效，拒绝恢复代码；原始代码和记录保留")
    live = _inside(work, work / "impl")
    failed = _inside(work, record["failed_implementation_dir"])
    source = _tree(work, best["implementation_dir"])
    expected = best["impl_sha256"]
    if record["status"] == "planned":
        # The uniquely named staging directory is never used by Stage3.
        staging = _inside(work, path.parent / f"prepared-{uuid.uuid4().hex}")
        shutil.copytree(source, staging)
        if implementation_hash(staging) != expected:
            raise ValueError("最佳实现复制校验失败；尚未替换当前代码")
        binding = work / "selection" / "current_implementation.json"
        if binding.is_file():
            shutil.copy2(binding, path.parent / "failed_binding.json")
        record.update(status="prepared", prepared_dir=str(staging), decision_path=str(decision))
        atomic_write_json(str(path), record)
    staging = _inside(work, record["prepared_dir"])
    # Two renames with a durable journal. On restart, each completed rename is
    # recognized from paths and hashes rather than repeated over newer contents.
    if not failed.exists():
        if implementation_hash(_tree(work, live)) != current["impl_sha256"]:
            raise ValueError("当前代码与退步版本不一致，拒绝覆盖开发修改")
        if implementation_hash(_tree(work, staging)) != expected:
            raise ValueError("待恢复的最佳代码校验失败")
        live.rename(failed)
    if implementation_hash(_tree(work, failed)) != current["impl_sha256"]:
        raise ValueError("退步代码备份校验失败，停止恢复")
    if not live.exists():
        if implementation_hash(_tree(work, staging)) != expected:
            raise ValueError("待恢复的最佳代码校验失败")
        staging.rename(live)
    elif staging.exists() or implementation_hash(_tree(work, live)) != expected:
        raise ValueError("工作代码已被修改，拒绝重复恢复覆盖")
    # Archived self-test JSON contains relative log paths. Invalidate the old
    # binding instead of pretending those tests were produced by a new Stage3.
    atomic_write_json(str(work / "selection" / "current_implementation.json"), {
        "schema_version": 1, "eligible": False, "stage": "regression_restore",
        "reason": "已恢复最佳已评测快照；Stage3 修改后须重新生成方案依据和自测绑定",
        "impl_sha256": expected, "restoration_record": str(path),
    })
    record.update(status="restored", restored_at_utc=datetime.now(timezone.utc).isoformat(),
                  decision_path=str(decision))
    atomic_write_json(str(path), record)
    _notice(log, state_log,
            f"===== 最佳实现恢复完成：iter{iteration} → 最佳 iter{best['iteration']} =====\n"
            f"avg_speedup={best['avg_speedup']}；avg_speed={best['avg_speed']}；"
            f"HAP.performance_score={best['hap'].get('performance_score')}\n"
            f"最佳来源={source}；工作代码={live}；退步代码保留={failed}\n"
            f"最佳性能报告={best['performance_report']}；退步教训={record['knowledge_path']}；"
            f"Stage9 决策={decision}；处置记录={path}\n===== 最佳实现恢复记录结束 =====")
    return record


def regression_action_prompt(work_dir, iteration):
    record = load_regression_action(work_dir, iteration)
    if not record:
        return ""
    text = "\n【性能退步处置】\n" + record["reason"] + "。\n"
    text += file_hint(work_dir, record["record_path"], "本轮退步处置与版本对应关系",
                      "核对 action/status、当前评测版本与最佳来源；本轮退步成绩不会被恢复后的成绩覆盖")
    if record["action"] == "restore_best":
        restored = record["status"] == "restored"
        text += ("工作 impl 已恢复最佳基线；" if restored else "Stage9 先分析当前退步版本并记录知识；程序随后恢复最佳基线；")
        text += ("后续修改建议应以最佳基线为起点，避免照搬只适用于退步代码的修改。"
                 "本轮退步报告属于恢复前代码；不把恢复当作新的有效评测，Stage3 修改后必须重新自测。\n")
        for name, item in (("退步版本", record["current_record"]), ("最佳基线", record["best_record"])):
            text += file_hint(work_dir, item["implementation_dir"], name + "的已评测代码快照", "先核对版本，再定位对应实现")
            text += file_hint(work_dir, item["manifest_path"], name + "的性能和方案清单", "查看 avg_speedup、HAP、融合方案及精度/性能报告地址")
            for key, value in item.get("evidence_paths", {}).items():
                text += file_hint(work_dir, value, name + "的实现依据/" + key, "只读归档，理解该版本方案选择和自测；不替代本轮新自测")
    return text

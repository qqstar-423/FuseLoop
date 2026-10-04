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
        raise ValueError(f"the rollback path must stay inside the working directory: {candidate}")
    # Do not follow directory symlinks/junctions for copy or rename operations.
    for part in (candidate, *candidate.parents):
        if part == work:
            break
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError(f"the rollback path must not pass through links: {candidate}")
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
        reason = "the current round lacks valid, version-bound evaluation evidence; only the regression lesson is recorded, no automatic restore"
    elif not current.get("all_cases_pass"):
        reason = "this round still has cases < 1; keeping the current implementation and only recording the regression lesson"
    elif not restore:
        reason = "no best passing snapshot with the same protocol exists; only the regression lesson is recorded, no automatic restore"
    else:
        reason = "all cases >= 1 this round; after Stage9 records the regression lesson, the best passing implementation is restored"
    if restore and implementation_hash(_tree(work, work / "impl")) != current["impl_sha256"]:
        raise ValueError("the code changed after this round\'s evaluation; refusing to overwrite unsaved modifications")
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
            f"===== Regression handling: iter{iteration} =====\n"
            f"iter{perf_diff.get('prev_iter')}→iter{iteration}; avg_speedup "
            f"{perf_diff.get('prev_avg_speedup')}→{perf_diff.get('curr_avg_speedup')}; "
            f"change={perf_diff.get('delta_pct')}% (the regression >= 5% rule still applies).\n"
            f"{reason}.\nAll cases passing now={record['all_cases_pass']}; "
            f"best=iter{best.get('iteration')}, avg_speedup={best.get('avg_speedup')}; "
            f"best directory={best.get('implementation_dir')}\n"
            f"regression knowledge={record['knowledge_path']}; handling record={path}\n"
            "===== end of regression handling record =====")
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
        raise ValueError("Stage9 must first commit this round\'s regression lesson before the code is restored")
    if record["action"] == "keep_current":
        record.update(decision_path=str(decision), knowledge_written=True)
        atomic_write_json(str(path), record)
        return record
    if comparison_context is not None and comparison_context != record["comparison_context"]:
        raise ValueError("the hardware or evaluation protocol changed at restore time; refusing to use best code from the old protocol")
    selection = load_selection_status(work, comparison_context=record["comparison_context"])
    best, current = selection.get("best") or {}, selection.get("current") or {}
    if (not selection.get("eligible") or selection.get("latest_iteration") != iteration
            or best.get("record_id") != record["best_record"].get("record_id")
            or current.get("record_id") != record["current_record"].get("record_id")):
        raise ValueError("the best or this round\'s evaluation snapshot is invalid; refusing to restore code; the original code and record are kept")
    live = _inside(work, work / "impl")
    failed = _inside(work, record["failed_implementation_dir"])
    source = _tree(work, best["implementation_dir"])
    expected = best["impl_sha256"]
    if record["status"] == "planned":
        # The uniquely named staging directory is never used by Stage3.
        staging = _inside(work, path.parent / f"prepared-{uuid.uuid4().hex}")
        shutil.copytree(source, staging)
        if implementation_hash(staging) != expected:
            raise ValueError("best implementation copy verification failed; the current code has not been replaced")
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
            raise ValueError("the current code does not match the regressed version; refusing to overwrite development changes")
        if implementation_hash(_tree(work, staging)) != expected:
            raise ValueError("verification of the best code to restore failed")
        live.rename(failed)
    if implementation_hash(_tree(work, failed)) != current["impl_sha256"]:
        raise ValueError("regressed code backup verification failed; stopping the restore")
    if not live.exists():
        if implementation_hash(_tree(work, staging)) != expected:
            raise ValueError("verification of the best code to restore failed")
        staging.rename(live)
    elif staging.exists() or implementation_hash(_tree(work, live)) != expected:
        raise ValueError("the working code was modified; refusing to overwrite it with a repeated restore")
    # Archived self-test JSON contains relative log paths. Invalidate the old
    # binding instead of pretending those tests were produced by a new Stage3.
    atomic_write_json(str(work / "selection" / "current_implementation.json"), {
        "schema_version": 1, "eligible": False, "stage": "regression_restore",
        "reason": "The best evaluated snapshot was restored; after Stage3 modifications the scheme rationale and self-test binding must be regenerated",
        "impl_sha256": expected, "restoration_record": str(path),
    })
    record.update(status="restored", restored_at_utc=datetime.now(timezone.utc).isoformat(),
                  decision_path=str(decision))
    atomic_write_json(str(path), record)
    _notice(log, state_log,
            f"===== Best implementation restore finished: iter{iteration} → best iter{best['iteration']} =====\n"
            f"avg_speedup={best['avg_speedup']}; avg_speed={best['avg_speed']}; "
            f"HAP.performance_score={best['hap'].get('performance_score')}\n"
            f"best source={source}; working code={live}; regressed code kept={failed}\n"
            f"best performance report={best['performance_report']}; regression lesson={record['knowledge_path']}; "
            f"Stage9 decision={decision}; handling record={path}\n===== end of best implementation restore record =====")
    return record


def regression_action_prompt(work_dir, iteration):
    record = load_regression_action(work_dir, iteration)
    if not record:
        return ""
    text = "\n[Regression handling]\n" + record["reason"] + ".\n"
    text += file_hint(work_dir, record["record_path"], "This round's regression handling and version correspondence",
                      "Verify action/status, the current evaluated version and the best source; this round's regressed scores will not be overwritten by post-restore scores")
    if record["action"] == "restore_best":
        restored = record["status"] == "restored"
        text += ("The working impl has been restored to the best baseline;" if restored else "Stage9 first analyzes the current regressed version and records knowledge; the program then restores the best baseline;")
        text += ("later modification suggestions should start from the best baseline; avoid copying changes that only suit the regressed code."
                 "This round's regression report belongs to the pre-restore code; do not treat the restore as a new valid evaluation, and Stage3 modifications must be re-tested.\n")
        for name, item in (("regressed version", record["current_record"]), ("best baseline", record["best_record"])):
            text += file_hint(work_dir, item["implementation_dir"], "Evaluated code snapshot of the " + name, "Verify the version first, then locate the corresponding implementation")
            text += file_hint(work_dir, item["manifest_path"], "Performance and scheme manifest of the " + name, "Look at avg_speedup, HAP, the fusion scheme and precision/performance report paths")
            for key, value in item.get("evidence_paths", {}).items():
                text += file_hint(work_dir, value, name + " implementation evidence/" + key, "Read-only archive for understanding that version's scheme selection and self-test; it does not replace this round's new self-test")
    return text

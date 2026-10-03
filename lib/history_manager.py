"""
History Manager — 管理跨轮迭代的上下文记忆
存储位置: <work>/knowledge/history.json

history 对象结构:
{
    "insights": [],          # 大模型每轮整体重写的洞察（字符串数组）
    "ledger": [],            # 累积追加的假设追踪账本（程序写硬数据）
    "rounds": [],            # 程序自动追加的每轮硬数据
    "bottleneck_now": "",    # 大模型：当前主要瓶颈
    "suggest_next": [],        # 大模型：下一轮建议方向（P0/P1/P2 分优先级列表）
    "worst_cases_tracker": {}, # 大模型：最慢 case 的持续追踪
    "fusion_kernel_strategy": []  # 大模型累加：融合算子策略追踪（每条含 iter/strategy/evidence/status）
}
"""
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from lib.handoff import atomic_write_json, atomic_write_text
from lib.knowledge_metadata import environment_summary
from lib.prompt_files import file_hint


def _history_path(work_dir: str) -> str:
    return os.path.join(work_dir, "knowledge", "history.json")


def load_history(work_dir: str) -> Dict[str, Any]:
    path = _history_path(work_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "insights": [],
        "ledger": [],
        "rounds": [],
        "bottleneck_now": "",
        "suggest_next": [],
        "worst_cases_tracker": {},
        "fusion_kernel_strategy": [],
    }


def save_history(work_dir: str, history: Dict[str, Any]):
    atomic_write_json(_history_path(work_dir), history)


def _finite_metric(value: Any, *, nonnegative: bool = False) -> Optional[float]:
    """Keep missing/invalid report metrics unknown instead of inventing a score."""
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or (nonnegative and value < 0)):
        return None
    return value


def _rounded_metric(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _apply_round_comparison(history, iteration, comparison):
    """Only the program may attach a cross-round comparison to measured rows."""
    can_compare = comparison.get("comparable") is True
    before = comparison.get("prev_avg_speedup") if can_compare else None
    after = comparison.get("curr_avg_speedup")
    for key in ("rounds", "ledger"):
        for row in history.get(key, []):
            if row.get("iter") != iteration:
                continue
            row.update(comparison_status=comparison.get("comparison_status", "unavailable"),
                       comparison_reason=comparison.get("comparison_reason", "缺少可比数据"),
                       comparison_previous_iter=comparison.get("prev_iter"))
            if key == "rounds":
                row["improved"] = after > before if can_compare else None
            else:
                row["avg_speedup_before"] = _rounded_metric(before, 4)
                row["delta"] = round(after - before, 4) if can_compare else None
                row["verdict"] = (_classify_verdict(after, before) if can_compare else
                                  "baseline" if comparison.get("comparison_status") == "new_baseline" else None)


def refresh_round_comparison(work_dir, iteration, comparison):
    """Recheck a resumed review without rewriting its measured data or tasks."""
    history = load_history(work_dir)
    original = json.dumps(history, ensure_ascii=False, sort_keys=True)
    _apply_round_comparison(history, iteration, comparison)
    if json.dumps(history, ensure_ascii=False, sort_keys=True) != original:
        save_history(work_dir, history)
        _save_rounds_snapshot(work_dir, history.get("rounds", []))


def append_round(work_dir: str, iteration: int, perf_result: Dict[str, Any],
                 prev_avg_speedup: float = 0.0, *, comparison=None) -> Dict[str, Any]:
    """
    程序自动追加本轮硬数据到 rounds[] 和 ledger[]。
    perf_result 来自 parse_perf_result() 的返回值。
    返回更新后的 history。
    """
    history = load_history(work_dir)

    # 防止断点续跑时重复追加——如果这轮已有条目，先删掉旧的
    history["rounds"] = [r for r in history.get("rounds", []) if r.get("iter") != iteration]
    history["ledger"] = [e for e in history.get("ledger", []) if e.get("iter") != iteration]

    avg_speedup = _finite_metric(perf_result.get("avg_speedup"), nonnegative=True)
    prev_avg_speedup = _finite_metric(prev_avg_speedup, nonnegative=True)
    overall_score = _finite_metric(perf_result.get("overall_score"))
    performance_score = _finite_metric(perf_result.get("performance_score"))
    cases = perf_result.get("cases") or perf_result.get("worst_6_cases") or []

    # 异常 case 优先展示，但保持 null，不能伪造成 0 分或参与正常速度排序。
    case_metrics = [
        {"case_id": c.get("case_id", "?"),
         "speedup": _finite_metric(c.get("speedup"), nonnegative=True)}
        for c in cases if isinstance(c, dict)
    ]
    worst_6 = sorted(case_metrics, key=lambda c: (
        c["speedup"] is not None, c["speedup"] if c["speedup"] is not None else 0
    ))[:6]

    comparable = avg_speedup is not None and prev_avg_speedup is not None
    improved = (avg_speedup > prev_avg_speedup if prev_avg_speedup > 0 else True) if comparable else None

    # rounds: 纯硬数据，程序写
    round_entry = {
        "iter": iteration,
        "avg_speedup": _rounded_metric(avg_speedup, 4),
        "overall_score": _rounded_metric(overall_score, 2),
        "performance_score": _rounded_metric(performance_score, 2),
        "improved": improved,
        "worst_case": {
            "case_id": worst_6[0].get("case_id", "?") if worst_6 else "?",
            "speedup": _rounded_metric(worst_6[0]["speedup"], 4) if worst_6 else None,
        },
        "worst_6_speedups": [
            {"case_id": c["case_id"], "speedup": _rounded_metric(c["speedup"], 4)}
            for c in worst_6
        ],
    }
    history["rounds"].append(round_entry)

    # ledger: 每轮迭代记录（reason + 方向 + 修改文件 + 只读文件，由 tech_lead 补充）
    ledger_entry = {
        "iter": iteration,
        "reason": "(pending tech_lead)",
        "direction": "(pending tech_lead)",
        "modify_files": [],
        "readonly_files": [],
        "avg_speedup_before": _rounded_metric(prev_avg_speedup, 4),
        "avg_speedup_after": _rounded_metric(avg_speedup, 4),
        "delta": round(avg_speedup - prev_avg_speedup, 4) if comparable else None,
        "verdict": _classify_verdict(avg_speedup, prev_avg_speedup) if comparable else None,
    }
    history["ledger"].append(ledger_entry)

    if comparison is not None:
        _apply_round_comparison(history, iteration, comparison)

    save_history(work_dir, history)
    # 备份 rounds（纯硬数据，不允许被模型篡改）
    _save_rounds_snapshot(work_dir, history["rounds"])
    return history


def append_ledger_only(work_dir: str, iteration: int, fail_reason: str) -> Dict[str, Any]:
    """
    编译失败/精度失败时调用——没走到 stage6，没有性能数据，
    但仍需追加一条 ledger 条目让 tech_lead 填写 direction/modify_files/readonly_files。
    """
    history = load_history(work_dir)
    # 防止断点续跑时重复追加
    history["ledger"] = [e for e in history.get("ledger", []) if e.get("iter") != iteration]
    ledger_entry = {
        "iter": iteration,
        "reason": fail_reason,
        "direction": "(pending tech_lead)",
        "modify_files": [],
        "readonly_files": [],
        "avg_speedup_before": None,
        "avg_speedup_after": None,
        "delta": None,
        "verdict": fail_reason,
    }
    history["ledger"].append(ledger_entry)
    save_history(work_dir, history)
    return history


def _classify_verdict(after: float, before: float) -> str:
    if before <= 0:
        return "baseline"
    delta_pct = (after - before) / before * 100
    if delta_pct >= 5:
        return "big_win"
    elif delta_pct >= 1:
        return "small_win"
    elif delta_pct >= -1:
        return "no_change"
    else:
        return "regression"


def update_from_tech_lead(work_dir: str, tech_lead_output: Dict[str, Any],
                          iteration: int) -> Dict[str, Any]:
    """
    用 tech_lead 的输出更新 history 的大模型字段。
    tech_lead_output 应包含: insights, bottleneck_now, suggest_next,
                             worst_cases_tracker, direction_for_ledger
    """
    if "plan_version" in tech_lead_output:
        raise ValueError("带 plan_version 的决策必须通过 merge_tech_lead_update 完整校验，不能使用旧 history 更新入口")
    history = load_history(work_dir)

    # insights: 整体替换
    if "insights" in tech_lead_output:
        history["insights"] = tech_lead_output["insights"]

    # bottleneck_now: 替换
    if "bottleneck_now" in tech_lead_output:
        history["bottleneck_now"] = tech_lead_output["bottleneck_now"]

    # suggest_next: 替换
    if "suggest_next" in tech_lead_output:
        history["suggest_next"] = tech_lead_output["suggest_next"]

    # worst_cases_tracker: 替换
    if "worst_cases_tracker" in tech_lead_output:
        history["worst_cases_tracker"] = tech_lead_output["worst_cases_tracker"]

    # 补充本轮 ledger 的 direction 字段
    if "direction_for_ledger" in tech_lead_output and history["ledger"]:
        for entry in reversed(history["ledger"]):
            if entry.get("iter") == iteration:
                entry["direction"] = tech_lead_output["direction_for_ledger"]
                break

    # fusion_kernel_strategy: 累加追加（不覆盖已有条目）
    if "fusion_kernel_strategy" in tech_lead_output:
        new_fds = tech_lead_output["fusion_kernel_strategy"]
        if isinstance(new_fds, list):
            existing = history.get("fusion_kernel_strategy", [])
            for fd in new_fds:
                if isinstance(fd, dict) and fd not in existing:
                    existing.append(fd)
            history["fusion_kernel_strategy"] = existing
        elif isinstance(new_fds, dict):
            # 单条也接受
            existing = history.get("fusion_kernel_strategy", [])
            if new_fds not in existing:
                existing.append(new_fds)
            history["fusion_kernel_strategy"] = existing

    # Retain legacy history for compatibility; program semantic windows own exit routing.
    if "exit_decision" in tech_lead_output:
        history["exit_decision"] = tech_lead_output["exit_decision"]

    # proven_pattern: 大幅提升时的成功经验（tech_lead 填写，程序追加到文件）
    if "proven_pattern" in tech_lead_output:
        history["_pending_proven_pattern"] = tech_lead_output["proven_pattern"]

    # regression_pattern: 大幅退步时的失败教训（tech_lead 填写，程序追加到文件）
    if "regression_pattern" in tech_lead_output:
        history["_pending_regression_pattern"] = tech_lead_output["regression_pattern"]

    # pitfall: tech_lead 对 question.md 的裁定（确认误判 or 驳回），程序追加到错题本
    if "pitfall" in tech_lead_output:
        history["_pending_pitfall"] = tech_lead_output["pitfall"]

    save_history(work_dir, history)
    return history


_PATTERN_DETAILS = {
    "environment": "框架与芯片",
    "prev_iter": "对比轮次",
    "case_analysis": "逐 case 分析",
    "evidence": "证据",
    "applicability": "适用条件",
    "next_action": "后续动作",
    "case_diffs": "程序计算的逐 case 变化",
    "evidence_files": "程序提供的证据文件",
    "decision_path": "本轮决策文件",
}
_JSON_FIELDS = {"environment", "prev_iter", "case_analysis", "evidence", "case_diffs", "evidence_files"}


def _record_blocks(content: str):
    """Keep unrelated Markdown blocks verbatim when replacing one record."""
    starts = list(re.finditer(r"^## iter\d+\b", content, re.MULTILINE))
    if not starts:
        return content, []
    return content[:starts[0].start()], [
        content[match.start():starts[i + 1].start() if i + 1 < len(starts) else len(content)]
        for i, match in enumerate(starts)
    ]


def _write_record(path: Path, preamble: str, record: str, matches):
    """Replace matching records once (including historical duplicates)."""
    content = path.read_text(encoding="utf-8-sig") if path.exists() else preamble
    prefix, blocks = _record_blocks(content)
    updated, replaced = [], False
    for block in blocks:
        if matches(block):
            if not replaced:
                updated.append(record)
                replaced = True
        else:
            updated.append(block)
    output = prefix + "".join(updated)
    if not replaced:
        output += ("\n\n" if output and not output.endswith("\n\n") else "") + record
    atomic_write_text(str(path), output)


def _field_block(label: str, value: Any, *, structured=False) -> str:
    # Multiline values are indented so embedded headings/field labels cannot
    # become record delimiters. Structured detail is stored once, under its
    # human-readable field label; no duplicate whole-record JSON is needed.
    if structured:
        body = "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
        return f"- **{label}**:\n\n" + "".join("  " + line + "\n" for line in body.split("\n")) + "\n"
    else:
        body = str(value)
    lines = body.split("\n")
    return f"- **{label}**: {lines[0]}\n" + "".join("  " + line + "\n" for line in lines[1:]) + "\n"


def _read_fields(block: str) -> Dict[str, str]:
    fields = {}
    markers = list(re.finditer(r"^- \*\*(.+?)\*\*:[ \t]?(.*)$", block, re.MULTILINE))
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(block)
        continuation = block[marker.end():end].split("\n")[1:]
        # Remove only the unindented Markdown separators. Indented empty lines
        # belong to the value, including intentional trailing newlines.
        while continuation and not continuation[-1]:
            continuation.pop()
        lines = [marker.group(2)] + [line[2:] if line.startswith("  ") else line for line in continuation]
        fields[marker.group(1)] = "\n".join(lines)
    return fields


def _detail_value(value: str):
    structured = value.strip()
    if structured.startswith("```json\n") and structured.endswith("\n```"):
        try:
            return json.loads(structured[len("```json\n"):-len("\n```")])
        except ValueError:
            pass  # Keep manually edited or legacy prose readable.
    return value


def _pattern_entry(block: str, *, regression=False):
    header = block.split("\n", 1)[0]
    match = re.match(r"## iter(\d+):\s*speedup\s+(\S+)\s*→\s*(\S+)\s*\(([^%]+)%\)(.*)", header)
    if not match:
        return None
    try:
        numbers = [0 if value.lstrip("+") == "?" else float(value) for value in match.group(2, 3, 4)]
    except ValueError:
        return None
    fields = _read_fields(block)
    why = "why_it_failed" if regression else "why_it_worked"
    entry = {"iter": int(match.group(1)), "speedup_before": numbers[0], "speedup_after": numbers[1],
             "delta_pct": numbers[2], "fusion_related": "融合" in match.group(5),
             "what_changed": fields.get("改了什么", ""),
             why: fields.get("为什么退步" if regression else "为什么有效", "")}
    entry.update({key: _detail_value(fields[label]) for key, label in _PATTERN_DETAILS.items() if label in fields})
    return entry


def _append_pattern(work_dir: str, iteration: int, pattern: Dict[str, Any], *, regression=False):
    filename = "regression_patterns.md" if regression else "proven_patterns.md"
    title = "已验证的失败教训（退步记录）" if regression else "已验证的成功优化经验"
    preamble = f"# {title}\n\n> Stage9 填写分析，程序验证性能变化达到 5% 后保存；同轮重试更新原记录。\n\n"
    tag = " 🔥融合相关" if pattern.get("fusion_related") else ""
    delta = str(pattern.get("delta_pct", "?"))
    if not regression and not delta.startswith(("+", "-")):
        delta = "+" + delta
    record = (f"## iter{iteration}: speedup {pattern.get('speedup_before', '?')} → "
              f"{pattern.get('speedup_after', '?')} ({delta}%){tag}\n\n")
    why = "why_it_failed" if regression else "why_it_worked"
    record += _field_block("改了什么", pattern.get("what_changed", "?"))
    record += _field_block("为什么退步" if regression else "为什么有效", pattern.get(why, "?"))
    for key, label in _PATTERN_DETAILS.items():
        if key in pattern and pattern[key] is not None:
            record += _field_block(label, pattern[key], structured=key in _JSON_FIELDS)
    _write_record(Path(work_dir) / "knowledge" / filename, preamble, record,
                  lambda block: bool(re.match(rf"## iter{iteration}:\s*speedup\b", block)))


def _load_patterns(work_dir: str, *, regression=False):
    path = Path(work_dir) / "knowledge" / ("regression_patterns.md" if regression else "proven_patterns.md")
    if not path.exists():
        return []
    _, blocks = _record_blocks(path.read_text(encoding="utf-8-sig"))
    return [entry for block in blocks if (entry := _pattern_entry(block, regression=regression)) is not None]


def append_proven_pattern(work_dir: str, iteration: int, pattern: Dict[str, Any]):
    """Save a verified improvement, updating the same iteration on retries."""
    _append_pattern(work_dir, iteration, pattern)


def load_proven_patterns(work_dir: str) -> List[Dict[str, Any]]:
    """Read both legacy short entries and multiline detailed Markdown records."""
    return _load_patterns(work_dir)


def format_proven_patterns_for_prompt(work_dir: str) -> str:
    """
    把 proven_patterns 格式化为可注入任意 stage prompt 的文本。
    用途：stage3（写代码参考）、stage7（分析时对照）、stage8（搜索时参考）、stage9（总结时引用）。

    注入说明会告诉 agent 这个文档是什么、怎么用。
    """
    patterns = load_proven_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\n📚 已验证的成功优化经验（knowledge/proven_patterns.md）",
        file_hint(work_dir, Path(work_dir) / "knowledge/proven_patterns.md", "以下成功经验摘要的来源",
                  "先核对经验中的框架与芯片，再核对改动、适用条件和提升证据；旧记录未注明环境时须先验证再复用").rstrip(),
        "这是历轮迭代中性能大幅提升（≥5%）时自动记录的成功经验。经验内容由 stage9 tech_lead 填写，orchestrator 验证提升幅度后才写入。",
        "参考方式：",
        "  - 写代码（stage3）：延续有效方向，避免回退已验证的改动",
        "  - 分析性能（stage7）：对照成功经验判断当前瓶颈是否已有解法",
        "  - 搜索方案（stage8）：搜索时优先搜和已验证方向相关的深入优化",
        "  - Tech Lead（stage9）：引用成功经验作为 insights 和 suggest_next 的证据",
        "",
    ]
    for p in patterns:
        fusion_tag = " 🔥[融合相关]" if p.get("fusion_related") else ""
        lines.append(
            f"  ✅ iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → "
            f"{p.get('speedup_after', '?')} (+{p.get('delta_pct', '?')}%){fusion_tag}"
        )
        lines.append(f"     改了什么: {p.get('what_changed', '?')}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     为什么有效: {p.get('why_it_worked', '?')}")
        if p.get("applicability"):
            lines.append(f"     适用条件: {p['applicability']}")
        lines.append("")

    return "\n".join(lines)


def append_regression_pattern(work_dir: str, iteration: int, pattern: Dict[str, Any]):
    """Save a verified regression, updating the same iteration on retries."""
    _append_pattern(work_dir, iteration, pattern, regression=True)


def load_regression_patterns(work_dir: str) -> List[Dict[str, Any]]:
    """Read both legacy and detailed regression records."""
    return _load_patterns(work_dir, regression=True)


def format_regression_patterns_for_prompt(work_dir: str) -> str:
    """把退步教训格式化为可注入任意 stage prompt 的文本。"""
    patterns = load_regression_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\n🚨 已验证的失败教训（knowledge/regression_patterns.md）",
        file_hint(work_dir, Path(work_dir) / "knowledge/regression_patterns.md", "以下退步教训摘要的来源",
                  "先核对经验中的框架与芯片，再查看退步改动、根因和受影响 case；旧记录未注明环境时须先验证再复用").rstrip(),
        "这是历轮迭代中性能大幅退步（≥5%）时自动记录的失败教训。先核对适用条件，在已验证条件下避免重复失败；改变条件后须重新验证。",
        "参考方式：",
        "  - 写代码（stage2/3）：避免重复相同条件下已导致退步的改动",
        "  - 分析性能（stage7）：核对当前瓶颈和记录中的失败条件是否一致",
        "  - 搜索方案（stage8）：排除已验证条件下的失败改法，条件改变须提供新证据",
        "  - Tech Lead（stage9）：引用失败教训作为否决 insight 的证据",
        "",
    ]
    for p in patterns:
        fusion_tag = " 🔥[融合相关]" if p.get("fusion_related") else ""
        lines.append(
            f"  ❌ iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → "
            f"{p.get('speedup_after', '?')} ({p.get('delta_pct', '?')}%){fusion_tag}"
        )
        lines.append(f"     改了什么: {p.get('what_changed', '?')}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     为什么退步: {p.get('why_it_failed', '?')}")
        if p.get("applicability"):
            lines.append(f"     适用条件: {p['applicability']}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 错题本（tech_lead_pitfalls）：下级反馈 → 上级裁定 → 沉淀
# ═══════════════════════════════════════════════════════════════

def append_pitfall(work_dir: str, iteration: int, pitfall: Dict[str, Any]):
    """Upsert this iteration's ruling by question path; retain older rulings."""
    path = Path(work_dir) / "knowledge/tech_lead_pitfalls.md"
    preamble = ("# Tech Lead 决策错题本\n\n"
                "> 记录下级反馈及裁定；同问题同轮重试更新原记录。\n"
                "> ✅确认误判：参考正确做法；❌驳回：参考正确认知。\n\n")
    verdict = pitfall.get("verdict", "confirmed")
    mark = "✅确认误判" if verdict == "confirmed" else "❌驳回"
    topic = str(pitfall.get("topic", "?"))
    record = f"## iter{iteration} {mark}: {' '.join(topic.splitlines())}\n\n"
    record += _field_block("主题", topic)
    for key, label in (("target_advice", "涉及的意见"), ("feedback", "cannbot 反馈"),
                       ("root_cause", "确认误判原因" if verdict == "confirmed" else "驳回原因"),
                       ("correct_approach", "正确做法" if verdict == "confirmed" else "正确认知")):
        record += _field_block(label, pitfall.get(key, "?"))
    if pitfall.get("question_path"):
        record += _field_block("问题文件", pitfall["question_path"])
    if pitfall.get("decision_path"):
        record += _field_block("本轮决策文件", pitfall["decision_path"])
    if pitfall.get("environment") is not None:
        record += _field_block("框架与芯片", pitfall["environment"], structured=True)

    def identity(value):
        value = os.fspath(value)
        return os.path.normcase(os.path.abspath(value if os.path.isabs(value) else os.path.join(work_dir, value)))

    def matches(block):
        previous = _pitfall_entry(block)
        if previous is None or previous["iter"] != iteration:
            return False
        if previous.get("question_path") and pitfall.get("question_path"):
            return identity(previous["question_path"]) == identity(pitfall["question_path"])
        # Old records have no question path. Preserve the old API by matching
        # their topic only when both records use that legacy form.
        return not previous.get("question_path") and not pitfall.get("question_path") and previous["topic"] == topic

    _write_record(path, preamble, record, matches)


def _pitfall_entry(block):
    match = re.match(r"## iter(\d+)\s+(✅确认误判|❌驳回):\s*([^\n]*)", block)
    if not match:
        return None
    fields = _read_fields(block)
    entry = {"iter": int(match.group(1)), "verdict": "confirmed" if "确认误判" in match.group(2) else "rejected",
             "topic": fields.get("主题", match.group(3).strip()),
             "target_advice": fields.get("涉及的意见", ""), "feedback": fields.get("cannbot 反馈", ""),
             "root_cause": fields.get("确认误判原因", fields.get("驳回原因", "")),
             "correct_approach": fields.get("正确做法", fields.get("正确认知", ""))}
    if "问题文件" in fields:
        entry["question_path"] = fields["问题文件"]
    if "本轮决策文件" in fields:
        entry["decision_path"] = fields["本轮决策文件"]
    if "框架与芯片" in fields:
        entry["environment"] = _detail_value(fields["框架与芯片"])
    return entry


def load_pitfalls(work_dir: str) -> List[Dict[str, Any]]:
    """读取错题本（从 md 解析回结构化数据）。"""
    path = Path(work_dir) / "knowledge/tech_lead_pitfalls.md"
    if not path.exists():
        return []
    _, blocks = _record_blocks(path.read_text(encoding="utf-8-sig"))
    return [entry for block in blocks if (entry := _pitfall_entry(block)) is not None]


def format_pitfalls_for_prompt(work_dir: str) -> str:
    """把错题本格式化为可注入 stage2/3/7/9 prompt 的文本。"""
    pitfalls = load_pitfalls(work_dir)
    if not pitfalls:
        return ""
    lines = [
        "\n📕 Tech Lead 决策错题本（knowledge/tech_lead_pitfalls.md）",
        file_hint(work_dir, Path(work_dir) / "knowledge/tech_lead_pitfalls.md", "以下历史裁定摘要的来源",
                  "先核对经验中的框架与芯片，再区分 confirmed/rejected；旧记录未注明环境时须先验证，再按正确做法或正确认知修正建议").rstrip(),
        "以下是历轮中 tech_lead 决策失误或 cannbot 无效反馈的记录。",
        "  - ✅确认误判：tech_lead 曾建议错，提意见/写代码时避开，按「正确做法」来",
        "  - ❌驳回：cannbot 曾误提 question，不要重复提，按「正确认知」理解",
        "",
    ]
    for p in pitfalls:
        mark = "✅" if p["verdict"] == "confirmed" else "❌"
        lines.append(f"  {mark} iter{p['iter']}: {p['topic']}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     涉及意见: {p['target_advice']}")
        lines.append(f"     正确{'做法' if p['verdict']=='confirmed' else '认知'}: {p['correct_approach']}")
        lines.append("")
    return "\n".join(lines)


def _format_action_task(item: Dict[str, Any]) -> List[str]:
    """Render a checked v2 task without replacing its executable details by prose."""
    pri = item.get("priority", "P2")
    marker = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(pri, "⚪")
    task_type = {"inspect": "检查", "modify": "修改"}.get(item.get("task_type"), "未知")
    title = f"### {marker} [{pri}] {item.get('task_id', '?')} · {task_type}：{item.get('action', '')}"
    if item.get("source") == "human":
        title += f" [人工意见 {item.get('human_message_id', '?')}]"
    lines = [title, f"原因: {item.get('reason', '')}"]
    if item.get("case_scope") == "operator":
        lines.append(f"算子整体/工程任务（不冒充某个 case 的优化）: {item.get('operator_reason', '')}")
    else:
        lines.append("目标 case（完整编号）: " + ", ".join(item.get("target_cases", [])))
        for binding in item.get("case_bindings", []):
            lines.append(f"- case→实现: {binding.get('case_id', '?')} → "
                         + ", ".join(binding.get("implementation_files", [])))
            for evidence in binding.get("route_evidence", []):
                lines.append(f"  路由证据: {evidence.get('file', '?')} · {evidence.get('location', '?')}"
                             f" — {evidence.get('explanation', '')}")
    lines.append("逐文件执行方法（这是实际任务范围）:")
    for change in item.get("changes", []):
        operation = {"inspect": "检查，不修改", "modify": "修改", "create": "新建"}.get(
            change.get("operation"), "未知")
        lines.append(f"- [{operation}] {change.get('file', '?')} · {change.get('location', '?')}")
        lines.append(f"  方法: {change.get('method', '')}")
    lines.append("验收要点:")
    lines.extend(f"- {check}" for check in item.get("acceptance_checks", []))
    return lines


def format_for_prompt(work_dir: str, max_ledger: int = 10,
                      current_iteration: Optional[int] = None) -> str:
    """
    将 history 格式化为可注入 stage3 prompt 的文本。
    每个字段带使用说明，让 cannbot 知道怎么用。
    """
    history = load_history(work_dir)
    ledger = history.get("ledger", [])
    current_entry = (next((entry for entry in reversed(ledger)
                           if entry.get("iter") == current_iteration), None)
                     if current_iteration is not None else ledger[-1] if ledger else None)
    current_label = f"iter{current_iteration}" if current_iteration is not None else "最后一轮"

    parts = []

    # ── suggest_next（最优先看，分优先级）──
    suggest = history.get("suggest_next", [])
    if current_iteration is not None:
        current_plan = current_entry.get("action_plan", {}) if current_entry else {}
        if isinstance(current_plan, dict) and current_plan.get("version") == 2:
            suggest = current_plan.get("tasks", [])
        elif not current_entry or current_entry is not ledger[-1]:
            suggest = []
            parts.append(f"当前 {current_label} 尚无可展示的任务单，不能沿用另一轮建议；需 Stage9 重新裁定。")
    if suggest:
        parts.append(
            "## ★ 本轮修改指令（由 tech_lead 指定，按优先级执行）\n"
            "P0 必须优先完成，包括最新人工指导；遵循 Tech Lead 给出的顺序和约束，不按是否融合自行提级。\n"
            "P1 次之；P2 是锦上添花。\n"
            "suggest_next 是本轮可执行事项的唯一清单。v2 任务按 task_id 列明目标 case、路由证据、逐文件方法和验收要点。\n"
            "只执行 changes 中声明的操作；inspect 不授予修改权，也不限制回查其他相关只读证据。\n"
            "direction 是方向概览，case_analysis.next_action 和 tracker 后续动作是研究线索，均不额外授权修改文件。\n"
        )
        if isinstance(suggest, list):
            for item in suggest:
                if isinstance(item, dict):
                    if item.get("task_id") and isinstance(item.get("changes"), list):
                        parts.extend(_format_action_task(item))
                        continue
                    pri = item.get("priority", "P2")
                    action = item.get("action", "")
                    reason = item.get("reason", "")
                    marker = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(pri, "⚪")
                    line = f"- {marker} [{pri}] {action}"
                    if item.get("source") == "human":
                        line += f" [人工意见 {item.get('human_message_id', '?')}]"
                    if reason:
                        line += f"（原因：{reason}）"
                    parts.append(line)
                    inspect = item.get("inspect_files")
                    modify = item.get("modify_files")
                    if (isinstance(inspect, list) and isinstance(modify, list)
                            and all(isinstance(path, str) and path.strip() for path in inspect + modify)
                            and (inspect or modify)):
                        parts.append("  ⚠️ 旧版任务仅作回顾；缺少 v2 的 case 映射、逐文件方法和验收，恢复执行前需 Stage9 重做任务单。")
                        parts.append(f"  检查文件（不授予修改权）: {', '.join(inspect) if inspect else '无'}")
                        parts.append(f"  修改文件（仍须在本轮允许范围内）: {', '.join(modify) if modify else '无，仅检查'}")
                    else:
                        parts.append("  ⚠️ 旧记录缺少完整的逐条文件范围，未按新规则校验；需 Stage9 补全，不可凭建议文字扩大修改范围。")
                else:
                    parts.append(f"- {item}\n  ⚠️ 旧建议未按新规则校验，缺少逐条文件范围；需 Stage9 补全。")
        elif isinstance(suggest, str) and suggest.strip():
            # 兼容旧格式（单字符串）
            parts.append(f"- 🔴 [P0] {suggest}\n  ⚠️ 旧建议未按新规则校验，缺少逐条文件范围；需 Stage9 补全。")

    # ── insights（按证据与适用条件参考的知识库）──
    if history.get("insights"):
        parts.append(
            "\n## 历史经验知识库（insights）\n"
            "每条记录了一个优化方向的尝试结果。格式：[方向] 轮次 | 证据 | 结论 | 状态\n"
            "- ✅已验证：核对适用条件后保留有效改动\n"
            "- ❌已否决：避免在相同条件下重复失败；条件或证据改变时按最新计划重新验证\n"
            "- 🔄待继续：有潜力，本轮可以继续深入\n"
        )
        for i, insight in enumerate(history["insights"], 1):
            parts.append(f"{i}. {insight}")

    # ── bottleneck_now（当前瓶颈）──
    if history.get("bottleneck_now"):
        parts.append(
            f"\n## 当前性能瓶颈\n"
            f"这是理解当前任务的背景；具体改法与权限以 suggest_next.changes 为准。\n\n"
            f"{history['bottleneck_now']}"
        )

    # ── worst_cases_tracker（最慢 case 追踪）──
    if history.get("worst_cases_tracker"):
        parts.append(
            "\n## 最慢 case 追踪\n"
            "每个 case 的历史 speedup 变化和原因。「硬件限制」是需核对条件和证据的历史判断，不能据此永久跳过该 case。\n"
            "后续动作只作研究线索；本轮是否执行及能改哪些文件，以 suggest_next 对应条目为准。\n"
        )
        tracker = history["worst_cases_tracker"]
        if isinstance(tracker, dict):
            for case_id, info in tracker.items():
                if isinstance(info, dict):
                    iteration = info.get("iteration", "?")
                    parts.append(f"- {case_id} [iter{iteration}]")
                    for key, label in (("explanation", "结论"), ("next_action", "后续动作"), ("evidence", "证据")):
                        if info.get(key):
                            value = info[key]
                            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            parts.append(f"  {label}: {text}")
                    parts.append("  完整观察与历史数据见 knowledge/history.json 的 worst_cases_tracker。")
                else:
                    parts.append(f"- {case_id}: {info}")
        elif isinstance(tracker, list):
            for item in tracker:
                parts.append(f"- {item}")

    # ── fusion_kernel_strategy（融合算子策略追踪）──
    fds = history.get("fusion_kernel_strategy", [])
    if fds:
        parts.append(
            "\n## 融合算子策略追踪（fusion_kernel_strategy）\n"
            "每条记录了一次融合方案的尝试。格式：iter/策略/证据/状态\n"
            "- ✅有效：当前方案的融合数据流正确，保持\n"
            "- ❌无效：实测或正确性证据证明该方案存在问题，需要改进\n"
            "- 🔄待验证：方案已提出，尚未验证效果\n"
            "结合当前代码、未达标 case 趋势和实测瓶颈审查方案；多 kernel 或 HBM 中间读写本身不代表无效。\n"
        )
        for fd in fds:
            if isinstance(fd, dict):
                parts.append(
                    f"- iter{fd.get('iter', '?')}: [{fd.get('status', '?')}] {fd.get('direction') or fd.get('strategy') or '?'}\n"
                    f"  证据: {fd.get('evidence', '无')}"
                )
            else:
                parts.append(f"- {fd}")

    # ── proven_patterns（已验证的成功经验）──
    patterns = load_proven_patterns(work_dir)
    if patterns:
        parts.append(
            "\n## 已验证的成功优化经验（knowledge/proven_patterns.md，相对工作目录）\n"
            "以下是历轮中性能大幅提升时记录的成功经验。**这些方向已被证明有效，修改时可以参考和延续。**\n"
        )
        parts.append(file_hint(work_dir, Path(work_dir) / "knowledge/proven_patterns.md",
                               "成功经验完整记录", "先核对框架与芯片再复用；以下保留全部历史改动与原因，逐 case 分析与证据按对应 iter 回看；旧记录未注明环境时须先验证"))
        for p in patterns:
            fusion_tag = " [融合相关]" if p.get("fusion_related") else ""
            parts.append(
                f"- iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → {p.get('speedup_after', '?')} "
                f"(+{p.get('delta_pct', '?')}%){fusion_tag}\n"
                f"  {environment_summary(p.get('environment'))}\n"
                f"  改了什么: {p.get('what_changed', '?')}\n"
                f"  为什么有效: {p.get('why_it_worked', '?')}"
            )
            if p.get("applicability"):
                parts.append(f"  适用条件: {p['applicability']}")

    # ── regression_patterns（已验证的失败教训）──
    regressions = load_regression_patterns(work_dir)
    if regressions:
        parts.append(
            "\n## 已验证的失败教训（knowledge/regression_patterns.md，相对工作目录）\n"
            "以下是历轮中性能大幅退步时记录的失败教训。在已验证条件下避免重复失败；改变条件后须重新验证。\n"
        )
        parts.append(file_hint(work_dir, Path(work_dir) / "knowledge/regression_patterns.md",
                               "退步教训完整记录", "先核对框架与芯片再复用；以下保留全部历史改动与原因，逐 case 分析与证据按对应 iter 回看；旧记录未注明环境时须先验证"))
        for p in regressions:
            fusion_tag = " [融合相关]" if p.get("fusion_related") else ""
            parts.append(
                f"- iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → {p.get('speedup_after', '?')} "
                f"({p.get('delta_pct', '?')}%){fusion_tag}\n"
                f"  {environment_summary(p.get('environment'))}\n"
                f"  改了什么: {p.get('what_changed', '?')}\n"
                f"  为什么退步: {p.get('why_it_failed', '?')}"
            )
            if p.get("applicability"):
                parts.append(f"  适用条件: {p['applicability']}")

    # ── ledger（迭代追踪账本）──
    if ledger:
        recent = ledger[-max_ledger:]
        if current_entry and all(entry is not current_entry for entry in recent):
            recent = recent + [current_entry]
        parts.append(
            f"\n## 迭代追踪账本（最近 {len(recent)} 轮）\n"
            "verdict 和 speedup 描述该轮已评测的实现；新记录的 direction 是评测后提出的下一步计划，不能把它当作已失败的改动。\n"
            "旧记录未区分方向时序，须回查设计和评测证据，不能直接给 direction 贴成功或失败标签。\n"
            f"旧轮文件范围仅供回顾；只有 {current_label} 已裁定计划的 modify_files/readonly_files 对本次修改生效。\n"
            "regression/no_change 本身不能否决下一步计划，也不能代替逐 case 趋势判断。\n"
        )
        for entry in recent:
            reason = entry.get('reason', '?')
            direction = entry.get('direction', '?')
            modify = entry.get('modify_files', [])
            readonly = entry.get('readonly_files', [])
            verdict = entry.get('verdict', '?')
            speedup_before = entry.get('avg_speedup_before')
            speedup_after = entry.get('avg_speedup_after')
            before_text = speedup_before if speedup_before is not None else '未评测/缺失'
            after_text = speedup_after if speedup_after is not None else '未评测/缺失'
            is_current = entry is current_entry and direction not in ('?', '', '(pending tech_lead)')

            parts.append(f"### iter{entry['iter']}（{reason}）已评测结果: {verdict or '未评测/缺失'}")
            if entry.get("evaluation_summary"):
                parts.append(f"本轮评测回顾: {entry['evaluation_summary']}")
                parts.append(f"评测后提出的下一步计划（尚未被本条 verdict 评价）: {direction}")
            else:
                parts.append(f"旧 direction（当时未区分已执行改动/下一步计划）: {direction}")
            if entry.get("case_analysis"):
                parts.append("本轮逐 case 分析见 knowledge/history.json 中本条 ledger.case_analysis。")
            parts.append(f"speedup: {before_text} → {after_text}")
            if entry.get("comparison_reason"):
                parts.append(f"程序比较结论: {entry['comparison_reason']}")
            if modify:
                label = '当前计划可修改' if is_current else '当时计划修改（仅回顾）'
                parts.append(f"{label}: {', '.join(modify)}")
            if readonly:
                label = '当前计划不可修改' if is_current else '当时计划只读（仅回顾）'
                visible = readonly if is_current else readonly[:5]
                parts.append(f"{label}: {', '.join(visible)}{'...' if not is_current and len(readonly) > 5 else ''}")
            parts.append("")

    # ── rounds 趋势 ──
    rounds = history.get("rounds", [])
    if rounds:
        parts.append(
            f"\n## 性能趋势（{len(rounds)} 轮）\n"
            "每轮的 avg_speedup 和最慢 case。↑=提升；未提升=持平或下降；?=缺少比较依据。"
            "新基线不能与前轮直接比较；不同硬件、case、基准或计时口径的数字不能推断涨跌。\n"
        )
        for r in rounds:
            mark = "↑" if r.get("improved") is True else "未提升" if r.get("improved") is False else "?"
            if r.get("comparison_status") in {"new_baseline", "unavailable"}:
                mark = "新基线，不比较涨跌" if r["comparison_status"] == "new_baseline" else "不可比较"
            avg = r.get("avg_speedup")
            worst = r.get("worst_case") or {}
            worst_speedup = worst.get("speedup")
            parts.append(
                f"- iter{r['iter']}: avg_speedup={avg if avg is not None else '未知'} {mark} "
                f"| worst={worst.get('case_id', '?')}({worst_speedup if worst_speedup is not None else '未知'})"
            )

    return (file_hint(work_dir, _history_path(work_dir), "以下跨轮经验与修改账本的来源",
                      f"先按任务编号阅读 case、文件、方法和验收，再核对 ledger 中 {current_label} 的只读范围；历史证据只供回顾")
            + "\n".join(parts)) if parts else "(无历史记录)"


def get_latest_fix_plan(work_dir: str, current_iteration: Optional[int] = None) -> str:
    """
    从 ledger 最后一条提取 modify_files/readonly_files/direction，
    格式化为 stage3 的文件约束指令。
    如果 tech_lead 没填（pending），返回空字符串。
    """
    history = load_history(work_dir)
    ledger = history.get("ledger", [])
    if not ledger:
        return ""

    latest = (next((entry for entry in reversed(ledger)
                    if entry.get("iter") == current_iteration), None)
              if current_iteration is not None else ledger[-1])
    if latest is None:
        return ""
    current_label = f"iter{current_iteration}" if current_iteration is not None else "最后一条"
    direction = latest.get("direction", "")
    modify = latest.get("modify_files", [])
    readonly = latest.get("readonly_files", [])
    reason = latest.get("reason", "")

    action_plan = latest.get("action_plan", {})
    is_v2 = isinstance(action_plan, dict) and action_plan.get("version") == 2
    if direction == "(pending tech_lead)" or (not modify and not readonly and not is_v2):
        return ""

    lines = []
    lines.append("=== Tech Lead 本轮修改指令（必须严格遵守）===")
    lines.append(file_hint(work_dir, _history_path(work_dir), "以下文件范围与修改方向的来源",
                           f"读取 ledger 中 {current_label} 已裁定计划，严格遵守 modify_files/readonly_files").rstrip())
    lines.append(f"问题类型: {reason}")
    lines.append(f"方向概览（不追加执行指令或修改权限）: {direction}")
    lines.append("具体事项只执行 suggest_next 的 v2 任务单；changes 按文件列明操作、位置、方法，按 acceptance_checks 验收。")
    lines.append("允许修改文件由 changes 的 modify/create 自动汇总，不是模型另填的一份授权；inspect 只读，不限制回查其他证据。")
    if modify:
        lines.append(f"可修改的文件（只改这些）:")
        for f in modify:
            lines.append(f"  - {f}")
    else:
        lines.append("可修改的文件: 无，本轮仅检查或已由程序确认结束。")
    if readonly:
        lines.append(f"不可修改的文件（绝对不能动）:")
        for f in readonly:
            lines.append(f"  - {f}")
    lines.append("⚠️ 严格按上述文件范围修改，不在 modify_files 里的文件一律不准碰。")
    if not is_v2:
        lines.append("⚠️ 旧任务未经 v2 校验，仅供回顾；恢复执行前需 Stage9 重新填写，不猜测补齐字段。")
    lines.append("direction、case_analysis.next_action、旧 fix_plan 均不能自行扩大权限。case 映射虽有证据，仍须结合实际路由核对；发现冲突先反馈，不越界执行。")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# rounds 保护：备份 + stage9 后恢复
# ═══════════════════════════════════════════════════════════════

def _rounds_snapshot_path(work_dir: str) -> str:
    return os.path.join(work_dir, "knowledge", "_rounds_snapshot.json")


def _fusion_snapshot_path(work_dir: str) -> str:
    return os.path.join(work_dir, "knowledge", "_fusion_snapshot.json")


def _save_rounds_snapshot(work_dir: str, rounds: List[Dict[str, Any]]):
    path = _rounds_snapshot_path(work_dir)
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rounds, f, ensure_ascii=False, indent=2)


def save_pre_tech_lead_snapshot(work_dir: str):
    """stage9 之前调用：备份 rounds 和 fusion_kernel_strategy（程序保护字段）。"""
    history = load_history(work_dir)
    _save_rounds_snapshot(work_dir, history.get("rounds", []))
    # 备份 fusion_kernel_strategy
    fd_path = _fusion_snapshot_path(work_dir)
    Path(os.path.dirname(fd_path)).mkdir(parents=True, exist_ok=True)
    with open(fd_path, "w", encoding="utf-8") as f:
        json.dump(history.get("fusion_kernel_strategy", []), f, ensure_ascii=False, indent=2)


def restore_program_fields(work_dir: str) -> Dict[str, Any]:
    """
    stage9 tech_lead 写完后调用。
    从备份恢复 rounds（程序硬数据，不允许被模型改），
    合并 fusion_kernel_strategy（模型追加的新条目保留，但不允许删除旧条目），
    保留 tech_lead 写的 insights/suggest/bottleneck/worst_cases_tracker/ledger.direction。
    """
    history = load_history(work_dir)

    # 恢复 rounds
    snapshot_path = _rounds_snapshot_path(work_dir)
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            history["rounds"] = json.load(f)

    # 合并 fusion_kernel_strategy：旧条目不能丢，新条目追加
    fd_snapshot_path = _fusion_snapshot_path(work_dir)
    if os.path.exists(fd_snapshot_path):
        with open(fd_snapshot_path, "r", encoding="utf-8") as f:
            old_fds = json.load(f)
        current_fds = history.get("fusion_kernel_strategy", [])
        # 以 old_fds 为基础，追加 current_fds 中不在 old_fds 里的新条目
        merged = list(old_fds)
        for fd in current_fds:
            if fd not in merged:
                merged.append(fd)
        history["fusion_kernel_strategy"] = merged

    save_history(work_dir, history)
    return history

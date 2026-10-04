"""
History Manager — manages cross-round contextual memory.
Storage location: <work>/knowledge/history.json

history object structure:
{
    "insights": [],          # model-rewritten insights each round (array of strings)
    "ledger": [],            # cumulatively appended hypothesis-tracking ledger (program writes hard data)
    "rounds": [],            # program-appended hard data for each round
    "bottleneck_now": "",    # model: the current main bottleneck
    "suggest_next": [],        # model: next-round suggested directions (P0/P1/P2 priority list)
    "worst_cases_tracker": {}, # model: continuous tracking of the slowest cases
    "fusion_kernel_strategy": []  # model-appended: fusion operator strategy tracking (each entry has iter/strategy/evidence/status)
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
                       comparison_reason=comparison.get("comparison_reason", "missing comparable data"),
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
    The program appends this round's hard data to rounds[] and ledger[] automatically.
    perf_result comes from parse_perf_result()'s return value.
    Returns the updated history.
    """
    history = load_history(work_dir)

    # avoid duplicate appends on checkpoint resume — if this round already has entries, delete the old ones first
    history["rounds"] = [r for r in history.get("rounds", []) if r.get("iter") != iteration]
    history["ledger"] = [e for e in history.get("ledger", []) if e.get("iter") != iteration]

    avg_speedup = _finite_metric(perf_result.get("avg_speedup"), nonnegative=True)
    prev_avg_speedup = _finite_metric(prev_avg_speedup, nonnegative=True)
    overall_score = _finite_metric(perf_result.get("overall_score"))
    performance_score = _finite_metric(perf_result.get("performance_score"))
    cases = perf_result.get("cases") or perf_result.get("worst_6_cases") or []

    # abnormal cases are shown first but stay null; they must not be faked as 0 scores or join normal speed sorting.
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

    # rounds: pure hard data, written by the program
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

    # ledger: per-iteration record (reason + direction + modified files + read-only files, completed by tech_lead)
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
    # back up rounds (pure hard data; the model must not tamper with it)
    _save_rounds_snapshot(work_dir, history["rounds"])
    return history


def append_ledger_only(work_dir: str, iteration: int, fail_reason: str) -> Dict[str, Any]:
    """
    Called on build/precision failure — stage6 was never reached, so there is no performance data,
    but a ledger entry is still appended for tech_lead to fill direction/modify_files/readonly_files.
    """
    history = load_history(work_dir)
    # avoid duplicate appends on checkpoint resume
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
    Update the history's model fields from tech_lead's output.
    tech_lead_output should contain: insights, bottleneck_now, suggest_next,
                                     worst_cases_tracker, direction_for_ledger
    """
    if "plan_version" in tech_lead_output:
        raise ValueError("decisions with plan_version must pass full validation through merge_tech_lead_update; the legacy history update entry cannot be used")
    history = load_history(work_dir)

    # insights: wholesale replacement
    if "insights" in tech_lead_output:
        history["insights"] = tech_lead_output["insights"]

    # bottleneck_now: replacement
    if "bottleneck_now" in tech_lead_output:
        history["bottleneck_now"] = tech_lead_output["bottleneck_now"]

    # suggest_next: replacement
    if "suggest_next" in tech_lead_output:
        history["suggest_next"] = tech_lead_output["suggest_next"]

    # worst_cases_tracker: replacement
    if "worst_cases_tracker" in tech_lead_output:
        history["worst_cases_tracker"] = tech_lead_output["worst_cases_tracker"]

    # complete this round's ledger direction field
    if "direction_for_ledger" in tech_lead_output and history["ledger"]:
        for entry in reversed(history["ledger"]):
            if entry.get("iter") == iteration:
                entry["direction"] = tech_lead_output["direction_for_ledger"]
                break

    # fusion_kernel_strategy: append cumulatively (existing entries are not overwritten)
    if "fusion_kernel_strategy" in tech_lead_output:
        new_fds = tech_lead_output["fusion_kernel_strategy"]
        if isinstance(new_fds, list):
            existing = history.get("fusion_kernel_strategy", [])
            for fd in new_fds:
                if isinstance(fd, dict) and fd not in existing:
                    existing.append(fd)
            history["fusion_kernel_strategy"] = existing
        elif isinstance(new_fds, dict):
            # a single entry is also accepted
            existing = history.get("fusion_kernel_strategy", [])
            if new_fds not in existing:
                existing.append(new_fds)
            history["fusion_kernel_strategy"] = existing

    # Retain legacy history for compatibility; program semantic windows own exit routing.
    if "exit_decision" in tech_lead_output:
        history["exit_decision"] = tech_lead_output["exit_decision"]

    # proven_pattern: success experience on big improvements (filled by tech_lead, appended to the file by the program)
    if "proven_pattern" in tech_lead_output:
        history["_pending_proven_pattern"] = tech_lead_output["proven_pattern"]

    # regression_pattern: failure lesson on big regressions (filled by tech_lead, appended to the file by the program)
    if "regression_pattern" in tech_lead_output:
        history["_pending_regression_pattern"] = tech_lead_output["regression_pattern"]

    # pitfall: tech_lead\'s adjudication of question.md (misjudgment confirmed or rejected); the program appends it to the pitfall log
    if "pitfall" in tech_lead_output:
        history["_pending_pitfall"] = tech_lead_output["pitfall"]

    save_history(work_dir, history)
    return history


_PATTERN_DETAILS = {
    "environment": "Framework and chip",
    "prev_iter": "Compared rounds",
    "case_analysis": "Per-case analysis",
    "evidence": "Evidence",
    "applicability": "Applicability",
    "next_action": "Next action",
    "case_diffs": "Per-case changes computed by the program",
    "evidence_files": "Evidence files provided by the program",
    "decision_path": "This round's decision file",
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
             "delta_pct": numbers[2], "fusion_related": "fusion-related" in match.group(5),
             "what_changed": fields.get("What changed", ""),
             why: fields.get("Why it regressed" if regression else "Why it worked", "")}
    entry.update({key: _detail_value(fields[label]) for key, label in _PATTERN_DETAILS.items() if label in fields})
    return entry


def _append_pattern(work_dir: str, iteration: int, pattern: Dict[str, Any], *, regression=False):
    filename = "regression_patterns.md" if regression else "proven_patterns.md"
    title = "Verified failure lessons (regression records)" if regression else "Verified successful optimization experience"
    preamble = f"# {title}\n\n> Stage9 writes the analysis; the program saves it after verifying the performance change reached 5%; same-round retries update the original record.\n\n"
    tag = " 🔥fusion-related" if pattern.get("fusion_related") else ""
    delta = str(pattern.get("delta_pct", "?"))
    if not regression and not delta.startswith(("+", "-")):
        delta = "+" + delta
    record = (f"## iter{iteration}: speedup {pattern.get('speedup_before', '?')} → "
              f"{pattern.get('speedup_after', '?')} ({delta}%){tag}\n\n")
    why = "why_it_failed" if regression else "why_it_worked"
    record += _field_block("What changed", pattern.get("what_changed", "?"))
    record += _field_block("Why it regressed" if regression else "Why it worked", pattern.get(why, "?"))
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
    Format proven_patterns into text injectable into any stage's prompt.
    Uses: stage3 (coding reference), stage7 (analysis cross-check), stage8 (search reference), stage9 (summary citation).

    The injection notes tell the agent what this document is and how to use it.
    """
    patterns = load_proven_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\n📚 Verified successful optimization experience (knowledge/proven_patterns.md)",
        file_hint(work_dir, Path(work_dir) / "knowledge/proven_patterns.md", "Source of the success experience summaries below",
                  "First verify the experience's framework and chip, then the change, applicability conditions and improvement evidence; old records without environment notes must be verified before reuse").rstrip(),
        "This is success experience recorded automatically when performance improved greatly (>=5%) in past iterations. The content is written by the stage9 tech_lead and only saved after the orchestrator verifies the improvement.",
        "How to use:",
        "  - Writing code (stage3): continue effective directions and avoid reverting verified changes",
        "  - Analyzing performance (stage7): check against the success experience whether the current bottleneck already has a solution",
        "  - Searching for schemes (stage8): prioritize in-depth optimizations related to verified directions",
        "  - Tech Lead (stage9): cite the success experience as evidence for insights and suggest_next",
        "",
    ]
    for p in patterns:
        fusion_tag = " 🔥[fusion-related]" if p.get("fusion_related") else ""
        lines.append(
            f"  ✅ iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → "
            f"{p.get('speedup_after', '?')} (+{p.get('delta_pct', '?')}%){fusion_tag}"
        )
        lines.append(f"     What changed: {p.get('what_changed', '?')}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     Why it worked: {p.get('why_it_worked', '?')}")
        if p.get("applicability"):
            lines.append(f"     Applicability: {p['applicability']}")
        lines.append("")

    return "\n".join(lines)


def append_regression_pattern(work_dir: str, iteration: int, pattern: Dict[str, Any]):
    """Save a verified regression, updating the same iteration on retries."""
    _append_pattern(work_dir, iteration, pattern, regression=True)


def load_regression_patterns(work_dir: str) -> List[Dict[str, Any]]:
    """Read both legacy and detailed regression records."""
    return _load_patterns(work_dir, regression=True)


def format_regression_patterns_for_prompt(work_dir: str) -> str:
    """Format regression lessons into text injectable into any stage\'s prompt."""
    patterns = load_regression_patterns(work_dir)
    if not patterns:
        return ""

    lines = [
        "\n🚨 Verified failure lessons (knowledge/regression_patterns.md)",
        file_hint(work_dir, Path(work_dir) / "knowledge/regression_patterns.md", "Source of the regression lesson summaries below",
                  "First verify the experience's framework and chip, then look at the regressing change, root cause and affected cases; old records without environment notes must be verified before reuse").rstrip(),
        "These are failure lessons recorded automatically when performance regressed greatly (>=5%) in past iterations. Verify applicability conditions first; avoid repeating failures under the verified conditions, and re-verify when conditions change.",
        "How to use:",
        "  - Writing code (stage2/3): avoid repeating changes that already caused regressions under the same conditions",
        "  - Analyzing performance (stage7): check whether the current bottleneck matches the recorded failure conditions",
        "  - Searching for schemes (stage8): exclude failure modes proven under verified conditions; changed conditions require new evidence",
        "  - Tech Lead (stage9): cite failure lessons as evidence for rejecting insights",
        "",
    ]
    for p in patterns:
        fusion_tag = " 🔥[fusion-related]" if p.get("fusion_related") else ""
        lines.append(
            f"  ❌ iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → "
            f"{p.get('speedup_after', '?')} ({p.get('delta_pct', '?')}%){fusion_tag}"
        )
        lines.append(f"     What changed: {p.get('what_changed', '?')}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     Why it regressed: {p.get('why_it_failed', '?')}")
        if p.get("applicability"):
            lines.append(f"     Applicability: {p['applicability']}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Pitfall log (tech_lead_pitfalls): subordinate feedback → superior adjudication → retained knowledge
# ═══════════════════════════════════════════════════════════════

def append_pitfall(work_dir: str, iteration: int, pitfall: Dict[str, Any]):
    """Upsert this iteration's ruling by question path; retain older rulings."""
    path = Path(work_dir) / "knowledge/tech_lead_pitfalls.md"
    preamble = ("# Tech Lead decision pitfall log\n\n"
                "> Records subordinate feedback and adjudications; same-round retries of the same question update the original record.\n"
                "> ✅ misjudgment confirmed: follow the correct practice; ❌ rejected: follow the correct understanding.\n\n")
    verdict = pitfall.get("verdict", "confirmed")
    mark = "✅ misjudgment confirmed" if verdict == "confirmed" else "❌ rejected"
    topic = str(pitfall.get("topic", "?"))
    record = f"## iter{iteration} {mark}: {' '.join(topic.splitlines())}\n\n"
    record += _field_block("Topic", topic)
    for key, label in (("target_advice", "Involved advice"), ("feedback", "cannbot feedback"),
                       ("root_cause", "Misjudgment confirmation reason" if verdict == "confirmed" else "Rejection reason"),
                       ("correct_approach", "Correct practice" if verdict == "confirmed" else "Correct understanding")):
        record += _field_block(label, pitfall.get(key, "?"))
    if pitfall.get("question_path"):
        record += _field_block("Question file", pitfall["question_path"])
    if pitfall.get("decision_path"):
        record += _field_block("This round's decision file", pitfall["decision_path"])
    if pitfall.get("environment") is not None:
        record += _field_block("Framework and chip", pitfall["environment"], structured=True)

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
    match = re.match(r"## iter(\d+)\s+(✅ misjudgment confirmed|❌ rejected):\s*([^\n]*)", block)
    if not match:
        return None
    fields = _read_fields(block)
    entry = {"iter": int(match.group(1)), "verdict": "confirmed" if "misjudgment confirmed" in match.group(2) else "rejected",
             "topic": fields.get("Topic", match.group(3).strip()),
             "target_advice": fields.get("Involved advice", ""), "feedback": fields.get("cannbot feedback", ""),
             "root_cause": fields.get("Misjudgment confirmation reason", fields.get("Rejection reason", "")),
             "correct_approach": fields.get("Correct practice", fields.get("Correct understanding", ""))}
    if "Question file" in fields:
        entry["question_path"] = fields["Question file"]
    if "This round's decision file" in fields:
        entry["decision_path"] = fields["This round's decision file"]
    if "Framework and chip" in fields:
        entry["environment"] = _detail_value(fields["Framework and chip"])
    return entry


def load_pitfalls(work_dir: str) -> List[Dict[str, Any]]:
    """Read the pitfall log (parse the Markdown back into structured data)."""
    path = Path(work_dir) / "knowledge/tech_lead_pitfalls.md"
    if not path.exists():
        return []
    _, blocks = _record_blocks(path.read_text(encoding="utf-8-sig"))
    return [entry for block in blocks if (entry := _pitfall_entry(block)) is not None]


def format_pitfalls_for_prompt(work_dir: str) -> str:
    """Format the pitfall log into text injectable into stage2/3/7/9 prompts."""
    pitfalls = load_pitfalls(work_dir)
    if not pitfalls:
        return ""
    lines = [
        "\n📕 Tech Lead decision pitfall log (knowledge/tech_lead_pitfalls.md)",
        file_hint(work_dir, Path(work_dir) / "knowledge/tech_lead_pitfalls.md", "Source of the historical adjudication summaries below",
                  "First verify the experience's framework and chip, then distinguish confirmed/rejected; old records without environment notes must be verified first, then correct suggestions per the correct practice or understanding").rstrip(),
        "These are records of tech_lead decision mistakes or invalid cannbot feedback across rounds.",
        "  - ✅ misjudgment confirmed: tech_lead once suggested wrongly; avoid it when giving advice or writing code, and follow the 'correct practice'",
        "  - ❌ rejected: cannbot once wrongly raised a question; do not repeat it and understand it per the 'correct understanding'",
        "",
    ]
    for p in pitfalls:
        mark = "✅" if p["verdict"] == "confirmed" else "❌"
        lines.append(f"  {mark} iter{p['iter']}: {p['topic']}")
        lines.append(f"     {environment_summary(p.get('environment'))}")
        lines.append(f"     Involved advice: {p['target_advice']}")
        lines.append(f"     Correct {'practice' if p['verdict']=='confirmed' else 'understanding'}: {p['correct_approach']}")
        lines.append("")
    return "\n".join(lines)


def _format_action_task(item: Dict[str, Any]) -> List[str]:
    """Render a checked v2 task without replacing its executable details by prose."""
    pri = item.get("priority", "P2")
    marker = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(pri, "⚪")
    task_type = {"inspect": "inspect", "modify": "modify"}.get(item.get("task_type"), "unknown")
    title = f"### {marker} [{pri}] {item.get('task_id', '?')} · {task_type}: {item.get('action', '')}"
    if item.get("source") == "human":
        title += f" [human feedback {item.get('human_message_id', '?')}]"
    lines = [title, f"Reason: {item.get('reason', '')}"]
    if item.get("case_scope") == "operator":
        lines.append(f"Whole-operator/project task (not posing as one case's optimization): {item.get('operator_reason', '')}")
    else:
        lines.append("Target cases (complete IDs): " + ", ".join(item.get("target_cases", [])))
        for binding in item.get("case_bindings", []):
            lines.append(f"- case→implementation: {binding.get('case_id', '?')} → "
                         + ", ".join(binding.get("implementation_files", [])))
            for evidence in binding.get("route_evidence", []):
                lines.append(f"  Routing evidence: {evidence.get('file', '?')} · {evidence.get('location', '?')}"
                             f" — {evidence.get('explanation', '')}")
    lines.append("Per-file execution method (this is the actual task scope):")
    for change in item.get("changes", []):
        operation = {"inspect": "inspect, no modification", "modify": "modify", "create": "create"}.get(
            change.get("operation"), "unknown")
        lines.append(f"- [{operation}] {change.get('file', '?')} · {change.get('location', '?')}")
        lines.append(f"  Method: {change.get('method', '')}")
    lines.append("Acceptance points:")
    lines.extend(f"- {check}" for check in item.get("acceptance_checks", []))
    return lines


def format_for_prompt(work_dir: str, max_ledger: int = 10,
                      current_iteration: Optional[int] = None) -> str:
    """
    Format history into text injectable into the stage3 prompt.
    Every field carries usage notes so cannbot knows how to use it.
    """
    history = load_history(work_dir)
    ledger = history.get("ledger", [])
    current_entry = (next((entry for entry in reversed(ledger)
                           if entry.get("iter") == current_iteration), None)
                     if current_iteration is not None else ledger[-1] if ledger else None)
    current_label = f"iter{current_iteration}" if current_iteration is not None else "the last round"

    parts = []

    # ── suggest_next (read first, prioritized)──
    suggest = history.get("suggest_next", [])
    if current_iteration is not None:
        current_plan = current_entry.get("action_plan", {}) if current_entry else {}
        if isinstance(current_plan, dict) and current_plan.get("version") == 2:
            suggest = current_plan.get("tasks", [])
        elif not current_entry or current_entry is not ledger[-1]:
            suggest = []
            parts.append(f"Current {current_label} has no task list to show; another round's suggestions cannot be reused; Stage9 must adjudicate again.")
    if suggest:
        parts.append(
            "## ★ This round's modification directives (specified by tech_lead; execute by priority)\n"
            "P0 must be completed first, including the latest human guidance; follow the order and constraints given by the Tech Lead and do not promote items yourself based on fusion involvement.\n"
            "P1 comes next; P2 is a nice-to-have.\n"
            "suggest_next is the single list of this round's executable items. v2 tasks list target cases, routing evidence, per-file methods and acceptance points by task_id.\n"
            "Execute only the operations declared in changes; inspect grants no modification rights and does not restrict checking other relevant read-only evidence.\n"
            "direction is a direction overview; case_analysis.next_action and tracker next actions are research leads; neither additionally authorizes file modifications.\n"
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
                        line += f" [human feedback {item.get('human_message_id', '?')}]"
                    if reason:
                        line += f" (reason: {reason})"
                    parts.append(line)
                    inspect = item.get("inspect_files")
                    modify = item.get("modify_files")
                    if (isinstance(inspect, list) and isinstance(modify, list)
                            and all(isinstance(path, str) and path.strip() for path in inspect + modify)
                            and (inspect or modify)):
                        parts.append("  ⚠️ Legacy task kept for review only; it lacks v2's case mapping, per-file methods and acceptance; Stage9 must redo the task list before resuming execution.")
                        parts.append(f"  Inspect files (no modification rights): {', '.join(inspect) if inspect else 'none'}")
                        parts.append(f"  Modify files (still within this round's allowed scope): {', '.join(modify) if modify else 'none, inspect only'}")
                    else:
                        parts.append("  ⚠️ This legacy record lacks complete per-item file scope and was not validated under the new rules; Stage9 must complete it, and the modification scope must not be expanded based on suggestion prose.")
                else:
                    parts.append(f"- {item}\n  ⚠️ This legacy suggestion was not validated under the new rules and lacks per-item file scope; Stage9 must complete it.")
        elif isinstance(suggest, str) and suggest.strip():
            # legacy format compatibility (single string)
            parts.append(f"- 🔴 [P0] {suggest}\n  ⚠️ This legacy suggestion was not validated under the new rules and lacks per-item file scope; Stage9 must complete it.")

    # ── insights (knowledge base referenced with evidence and applicability conditions)──
    if history.get("insights"):
        parts.append(
            "## Historical experience knowledge base (insights)\n"
            "Each entry records the outcome of trying one optimization direction. Format: [direction] round | evidence | conclusion | status\n"
            "- ✅ verified: keep the effective change after checking applicability conditions\n"
            "- ❌ rejected: avoid repeating failures under the same conditions; when conditions or evidence change, re-verify per the latest plan\n"
            "- 🔄 to continue: promising; this round may go deeper\n"
        )
        for i, insight in enumerate(history["insights"], 1):
            parts.append(f"{i}. {insight}")

    # ── bottleneck_now (current bottleneck)──
    if history.get("bottleneck_now"):
        parts.append(
            f"\n## Current performance bottleneck\n"
            f"This is background for understanding the current task; the concrete fixes and permissions follow suggest_next.changes.\n\n"
            f"{history['bottleneck_now']}"
        )

    # ── worst_cases_tracker (slowest case tracking)──
    if history.get("worst_cases_tracker"):
        parts.append(
            "## Slowest case tracking\n"
            "Each case's historical speedup changes and causes. 'Hardware limitation' is a historical judgment whose conditions and evidence must be verified; it never justifies permanently skipping a case.\n"
            "Next actions are research leads only; whether to execute this round and which files can change follow the corresponding suggest_next entries.\n"
        )
        tracker = history["worst_cases_tracker"]
        if isinstance(tracker, dict):
            for case_id, info in tracker.items():
                if isinstance(info, dict):
                    iteration = info.get("iteration", "?")
                    parts.append(f"- {case_id} [iter{iteration}]")
                    for key, label in (("explanation", "Conclusion"), ("next_action", "Next action"), ("evidence", "Evidence")):
                        if info.get(key):
                            value = info[key]
                            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            parts.append(f"  {label}: {text}")
                    parts.append("  Full observations and historical data are in knowledge/history.json's worst_cases_tracker.")
                else:
                    parts.append(f"- {case_id}: {info}")
        elif isinstance(tracker, list):
            for item in tracker:
                parts.append(f"- {item}")

    # ── fusion_kernel_strategy (fusion operator strategy tracking)──
    fds = history.get("fusion_kernel_strategy", [])
    if fds:
        parts.append(
            "## Fusion operator strategy tracking (fusion_kernel_strategy)\n"
            "Each entry records one fusion scheme attempt. Format: iter/strategy/evidence/status\n"
            "- ✅ valid: the current scheme's fusion dataflow is correct; keep it\n"
            "- ❌ invalid: measured or correctness evidence shows the scheme has problems and needs improvement\n"
            "- 🔄 pending verification: proposed but its effect is unverified\n"
            "Review schemes against the current code, underperforming case trends and measured bottlenecks; multiple kernels or HBM intermediates do not by themselves mean invalid.\n"
        )
        for fd in fds:
            if isinstance(fd, dict):
                parts.append(
                    f"- iter{fd.get('iter', '?')}: [{fd.get('status', '?')}] {fd.get('direction') or fd.get('strategy') or '?'}\n"
                    f"  Evidence: {fd.get('evidence', 'none')}"
                )
            else:
                parts.append(f"- {fd}")

    # ── proven_patterns (verified success experience)──
    patterns = load_proven_patterns(work_dir)
    if patterns:
        parts.append(
            "## Verified successful optimization experience (knowledge/proven_patterns.md, relative to the working directory)\n"
            "These are success experiences recorded when performance improved greatly in past rounds. **These directions are proven effective; you may reference and continue them when modifying.**\n"
        )
        parts.append(file_hint(work_dir, Path(work_dir) / "knowledge/proven_patterns.md",
                               "Full record of success experience", "Verify framework and chip before reuse; the full historical changes and reasons are kept below, with per-case analysis and evidence looked up by their iter; old records without environment notes must be verified first"))
        for p in patterns:
            fusion_tag = " [fusion-related]" if p.get("fusion_related") else ""
            parts.append(
                f"- iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → {p.get('speedup_after', '?')} "
                f"(+{p.get('delta_pct', '?')}%){fusion_tag}\n"
                f"  {environment_summary(p.get('environment'))}\n"
                f"  What changed: {p.get('what_changed', '?')}\n"
                f"  Why it worked: {p.get('why_it_worked', '?')}"
            )
            if p.get("applicability"):
                parts.append(f"  Applicability: {p['applicability']}")

    # ── regression_patterns (verified failure lessons)──
    regressions = load_regression_patterns(work_dir)
    if regressions:
        parts.append(
            "## Verified failure lessons (knowledge/regression_patterns.md, relative to the working directory)\n"
            "These are failure lessons recorded when performance regressed greatly in past rounds. Avoid repeating failures under verified conditions; re-verify when conditions change.\n"
        )
        parts.append(file_hint(work_dir, Path(work_dir) / "knowledge/regression_patterns.md",
                               "Full record of regression lessons", "Verify framework and chip before reuse; the full historical changes and reasons are kept below, with per-case analysis and evidence looked up by their iter; old records without environment notes must be verified first"))
        for p in regressions:
            fusion_tag = " [fusion-related]" if p.get("fusion_related") else ""
            parts.append(
                f"- iter{p.get('iter', '?')}: speedup {p.get('speedup_before', '?')} → {p.get('speedup_after', '?')} "
                f"({p.get('delta_pct', '?')}%){fusion_tag}\n"
                f"  {environment_summary(p.get('environment'))}\n"
                f"  What changed: {p.get('what_changed', '?')}\n"
                f"  Why it regressed: {p.get('why_it_failed', '?')}"
            )
            if p.get("applicability"):
                parts.append(f"  Applicability: {p['applicability']}")

    # ── ledger (iteration tracking ledger)──
    if ledger:
        recent = ledger[-max_ledger:]
        if current_entry and all(entry is not current_entry for entry in recent):
            recent = recent + [current_entry]
        parts.append(
            f"## Iteration tracking ledger (last {len(recent)} rounds)\n"
            "verdict and speedup describe that round's evaluated implementation; a new record's direction is the next-step plan proposed after evaluation and must not be treated as an already-failed change.\n"
            "Old records did not distinguish direction timing; check back through design and evaluation evidence, and do not label a direction success or failure directly.\n"
            f"Old rounds' file scopes are for review only; only {current_label}'s adjudicated plan's modify_files/readonly_files take effect for this round's modifications.\n"
            "regression/no_change by themselves cannot veto the next-step plan, nor can they replace per-case trend judgment.\n"
        )
        for entry in recent:
            reason = entry.get('reason', '?')
            direction = entry.get('direction', '?')
            modify = entry.get('modify_files', [])
            readonly = entry.get('readonly_files', [])
            verdict = entry.get('verdict', '?')
            speedup_before = entry.get('avg_speedup_before')
            speedup_after = entry.get('avg_speedup_after')
            before_text = speedup_before if speedup_before is not None else 'not evaluated/missing'
            after_text = speedup_after if speedup_after is not None else 'not evaluated/missing'
            is_current = entry is current_entry and direction not in ('?', '', '(pending tech_lead)')

            parts.append(f"### iter{entry['iter']} ({reason}) evaluated result: {verdict or 'not evaluated/missing'}")
            if entry.get("evaluation_summary"):
                parts.append(f"This round's evaluation review: {entry['evaluation_summary']}")
                parts.append(f"Next-step plan proposed after evaluation (not yet judged by this entry\'s verdict): {direction}")
            else:
                parts.append(f"Legacy direction (executed changes/next-step plan were not distinguished then): {direction}")
            if entry.get("case_analysis"):
                parts.append("This round\'s per-case analysis is in this entry\'s ledger.case_analysis in knowledge/history.json.")
            parts.append(f"speedup: {before_text} → {after_text}")
            if entry.get("comparison_reason"):
                parts.append(f"Program comparison conclusion: {entry['comparison_reason']}")
            if modify:
                label = 'current plan may modify' if is_current else 'planned modification at the time (review only)'
                parts.append(f"{label}: {', '.join(modify)}")
            if readonly:
                label = 'current plan must not modify' if is_current else 'read-only at the time (review only)'
                visible = readonly if is_current else readonly[:5]
                parts.append(f"{label}: {', '.join(visible)}{'...' if not is_current and len(readonly) > 5 else ''}")
            parts.append("")

    # ── rounds trends ──
    rounds = history.get("rounds", [])
    if rounds:
        parts.append(
            f"## Performance trends ({len(rounds)} rounds)\n"
            "Each round's avg_speedup and slowest case. ↑=improvement; not improved=flat or down; ?=missing comparison basis."
            "A new baseline cannot be compared directly with previous rounds; numbers across different hardware, cases, baselines or timing protocols cannot infer up/down.\n"
        )
        for r in rounds:
            mark = "↑" if r.get("improved") is True else "not improved" if r.get("improved") is False else "?"
            if r.get("comparison_status") in {"new_baseline", "unavailable"}:
                mark = "new baseline, no up/down comparison" if r["comparison_status"] == "new_baseline" else "not comparable"
            avg = r.get("avg_speedup")
            worst = r.get("worst_case") or {}
            worst_speedup = worst.get("speedup")
            parts.append(
                f"- iter{r['iter']}: avg_speedup={avg if avg is not None else 'unknown'} {mark} "
                f"| worst={worst.get('case_id', '?')}({worst_speedup if worst_speedup is not None else 'unknown'})"
            )

    return (file_hint(work_dir, _history_path(work_dir), "Source of the cross-round experience and modification ledger below",
                      f"First read the cases, files, methods and acceptance by task ID, then verify the read-only scope of {current_label} in the ledger; historical evidence is for review only")
            + "\n".join(parts)) if parts else "(no history records)"


def get_latest_fix_plan(work_dir: str, current_iteration: Optional[int] = None) -> str:
    """
    Extract modify_files/readonly_files/direction from the last ledger entry
    and format them as stage3's file constraint directives.
    Returns an empty string if tech_lead has not filled them in (pending).
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
    current_label = f"iter{current_iteration}" if current_iteration is not None else "the last entry"
    direction = latest.get("direction", "")
    modify = latest.get("modify_files", [])
    readonly = latest.get("readonly_files", [])
    reason = latest.get("reason", "")

    action_plan = latest.get("action_plan", {})
    is_v2 = isinstance(action_plan, dict) and action_plan.get("version") == 2
    if direction == "(pending tech_lead)" or (not modify and not readonly and not is_v2):
        return ""

    lines = []
    lines.append("=== Tech Lead's modification directives for this round (must be followed strictly)===")
    lines.append(file_hint(work_dir, _history_path(work_dir), "Source of the file scope and modification direction below",
                           f"Read {current_label}'s adjudicated plan in the ledger and strictly follow modify_files/readonly_files").rstrip())
    lines.append(f"Problem type: {reason}")
    lines.append(f"Direction overview (adds no execution directives or modification permissions): {direction}")
    lines.append("Execute concrete items only via suggest_next's v2 task list; changes lists operation, location and method per file, with acceptance per acceptance_checks.")
    lines.append("The writable files are aggregated automatically from changes\' modify/create steps, not from a separate authorization filled in by the model; inspect is read-only and does not restrict checking other evidence.")
    if modify:
        lines.append(f"Files that may be modified (change only these):")
        for f in modify:
            lines.append(f"  - {f}")
    else:
        lines.append("Files that may be modified: none; this round is inspect-only or the program has confirmed it is finished.")
    if readonly:
        lines.append(f"Files that must not be modified (absolutely untouchable):")
        for f in readonly:
            lines.append(f"  - {f}")
    lines.append("⚠️ Modify strictly within the file scope above; files not in modify_files must never be touched.")
    if not is_v2:
        lines.append("⚠️ This legacy task was not v2-validated and is for review only; Stage9 must refill it before resuming execution, and fields must not be guessed.")
    lines.append("direction, case_analysis.next_action and old fix_plan never expand permissions by themselves. Case mappings have evidence, but still verify against actual routing; report conflicts first instead of overstepping.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# rounds protection: backup + restore after stage9
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
    """Called before stage9: back up rounds and fusion_kernel_strategy (program-protected fields)."""
    history = load_history(work_dir)
    _save_rounds_snapshot(work_dir, history.get("rounds", []))
    # back up fusion_kernel_strategy
    fd_path = _fusion_snapshot_path(work_dir)
    Path(os.path.dirname(fd_path)).mkdir(parents=True, exist_ok=True)
    with open(fd_path, "w", encoding="utf-8") as f:
        json.dump(history.get("fusion_kernel_strategy", []), f, ensure_ascii=False, indent=2)


def restore_program_fields(work_dir: str) -> Dict[str, Any]:
    """
    Called after stage9's tech_lead writes.
    Restores rounds from the backup (program hard data the model must not change),
    merges fusion_kernel_strategy (new entries appended by the model are kept, but old entries must not be deleted),
    and keeps tech_lead's insights/suggest/bottleneck/worst_cases_tracker/ledger.direction.
    """
    history = load_history(work_dir)

    # restore rounds
    snapshot_path = _rounds_snapshot_path(work_dir)
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            history["rounds"] = json.load(f)

    # merge fusion_kernel_strategy: old entries must be kept, new entries appended
    fd_snapshot_path = _fusion_snapshot_path(work_dir)
    if os.path.exists(fd_snapshot_path):
        with open(fd_snapshot_path, "r", encoding="utf-8") as f:
            old_fds = json.load(f)
        current_fds = history.get("fusion_kernel_strategy", [])
        # start from old_fds and append new entries from current_fds that are not already there
        merged = list(old_fds)
        for fd in current_fds:
            if fd not in merged:
                merged.append(fd)
        history["fusion_kernel_strategy"] = merged

    save_history(work_dir, history)
    return history

"""Small protocol helpers shared by Stage9, development and the human inbox.

The workflow, not a model, controls waiting, delivery and evaluation counters.
"""
import json
from pathlib import Path

from .handoff import atomic_write_json, atomic_write_text
from .prompt_files import file_hint


def evidence_sources(work_dir, iter_dirs):
    work = Path(work_dir)
    items = [
        (work / "task", "Operator requirements, cases and golden", "Verify interfaces, precision and coverage"),
        (work / "ANALYSIS.md", "Stage1 requirements analysis", "Verify implementation constraints"),
        (work / "device_info.json", "Current hardware", "Verify core counts, memory capacity and the programming model"),
        (work / "impl", "This round's actual implementation", "Compare the entry point, tiling and dataflow"),
        (work / "fusion", "Stage1.5 initial schemes and probabilities", "Probabilities are priors; judge together with current measurements"),
        (work / "selection", "Best implementation, code binding, window and case trends", "Verify that snapshot metrics match the implementation"),
        (work / "knowledge/history.json", "Complete cross-round ledger", "Compare attempted directions and results"),
        (work / "knowledge/proven_patterns.md", "Success experience", "Verify applicability conditions"),
        (work / "knowledge/regression_patterns.md", "Regression experience", "Avoid repeating proven problems"),
        (work / "knowledge/tech_lead_pitfalls.md", "Dispute adjudications", "Check historical misjudgments and correct practices"),
        (work / "develop", "Cross-round fusion selection rationale and self-tests", "Distinguish implementation choices, self-tests and formal evaluations"),
    ]
    for key, purpose in (("build", "This round's build log"), ("eval", "Precision, performance and raw profiling"),
                         ("profile", "Stage7 bottleneck conclusions"), ("search", "Stage8 search and suggestions")):
        items.append((Path(iter_dirs[key]), purpose, "Read the conclusions first, then check raw evidence per case"))
    return [{"path": str(path), "purpose": purpose, "read_hint": hint,
             "source_description": file_hint(work_dir, path, purpose, hint).strip()}
            for path, purpose, hint in items]


def validate_question(output, request_id):
    if not isinstance(output, dict) or output.get("request_id") != request_id:
        raise ValueError("the consultation output must match this request_id")
    for key in ("question", "difficulty", "current_scheme", "attempts", "evidence", "recommendation_reason"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise ValueError(f"the consultation is missing {key}")
    options = output.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 3:
        raise ValueError("the consultation must provide 2-3 concrete options")
    ids = set()
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("consultation options must be objects")
        for key in ("id", "title", "benefit", "cost", "risk"):
            if not isinstance(option.get(key), str) or not option[key].strip():
                raise ValueError(f"a consultation option is missing {key}")
        if option["id"] in ids:
            raise ValueError("consultation option IDs must be unique")
        ids.add(option["id"])
    if output.get("recommended_option") not in ids:
        raise ValueError("the recommended option must appear in the option list")


def render_question(output, request, selection_status, work_dir):
    validate_question(output, request["request_id"])
    lines = ["# Stage9 requests human judgment", "", output["question"], "",
             f"Current scheme: {output['current_scheme']}", f"Difficulty: {output['difficulty']}",
             f"Already tried: {output['attempts']}", "", f"Evidence: {output['evidence']}"]
    best = selection_status.get("best") or {}
    lines += ["", f"Stagnation trigger rounds recorded by the program: {request.get('trigger_iterations', [])}",
              f"Best avg_speedup: {best.get('avg_speedup', 'n/a')}; HAP: {(best.get('hap') or {}).get('performance_score', 'n/a')}",
              f"Window: {json.dumps(selection_status.get('window', {}), ensure_ascii=False)}", ""]
    for option in output["options"]:
        lines += [f"- {option['id']}: {option['title']}. Benefit: {option['benefit']}; cost: {option['cost']}; risk: {option['risk']}."]
    lines += ["", f"Recommended {output['recommended_option']}: {output['recommendation_reason']}", "",
              "Submit an option or feedback to this work directory via tools/human_review.py. Other directions may be proposed; use --kind question for a pure question.",
              "After notification wait 2 minutes; replying \"please wait\" adds 10 minutes to the original deadline only. On timeout continue per the recommended direction; it must not be treated as human consent.", ""]
    manifest = request.get("evidence_manifest_path")
    if manifest:
        lines.append(file_hint(work_dir, manifest, "Version manifest of all evidence at consultation time", "Find snapshots by purpose and reading hint; missing material is clearly recorded"))
    return "\n".join(lines) + "\n"


def human_prompt(work_dir, messages, bundle_path=None, *, ending_reason=None):
    if not messages and not bundle_path:
        return ""
    text = "\n=== Human feedback (handle item by item; substantive directions are P0) ===\n"
    if bundle_path:
        text += file_hint(work_dir, bundle_path, "Full human feedback bundle and consultation-time evidence", "Read the original question, all options and all verbatim messages first, then verify snapshot versions; timeout does not imply human consent")
    text += json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
    text += (
        "human_responses must output {message_id, kind, answer} for every message; kind reuses the submitted type direction/question/wait."
        "A substantive direction must not become a question; when it conflicts with correctness or hardware use kind=conflict, explain why and add alternative."
        "Every direction/conflict must have a P0 in suggest_next with source=human and human_message_id matching that message's number;"
        "action must state the goal or a feasible alternative of the P0 (human suggestion) explicitly. Answer pure questions first; wait is not a direction."
        "Do not disguise a timeout recommendation as human feedback.\n")
    if ending_reason:
        text += f"The program has decided to exit: {ending_reason}. Feedback still must be handled and human P0s preserved, but Stage3 will not run again; explain why anything was not executed and never break the exit conditions.\n"
    return text


def development_human_prompt(work_dir, messages, output_path):
    if not messages:
        return ""
    return (
        "\n=== This round's human P0 delivery ===\n" + json.dumps(messages, ensure_ascii=False, indent=2)
        + "\nFollow the corresponding human P0 from Stage9 in history; in the fusion scheme selection rationale document, explain by feedback ID what was actually implemented, which files were modified, or why it could not be implemented."
        + file_hint(work_dir, output_path, "Execution receipt for this round's human feedback (to be generated)",
                    "Write a JSON list with each item's message_id, status (implemented or not_implemented) and details (concrete changes or reasons). Passing self-tests cannot substitute for formal performance verification")
    )


def validate_execution_receipt(path, messages):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = {message["id"] for message in messages}
    if not isinstance(rows, list):
        raise ValueError("the human execution receipt must be a JSON list")
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("message_id") not in expected
                or row["message_id"] in seen or row.get("status") not in {"implemented", "not_implemented"}
                or not isinstance(row.get("details"), str) or not row["details"].strip()):
            raise ValueError("the human execution receipt lacks a valid ID, status or concrete implementation rationale")
        seen.add(row["message_id"])
    if seen != expected:
        raise ValueError("the human execution receipt must cover every piece of this round\'s feedback")
    return rows

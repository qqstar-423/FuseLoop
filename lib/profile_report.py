"""Publish one program-attributed profiling report per workflow iteration."""

from pathlib import Path

from lib.handoff import atomic_write_text


_SOURCE_START = "<!-- profile-report-source:start -->"
_SOURCE_END = "<!-- profile-report-source:end -->"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


def profile_report_path(work_dir, iteration):
    """Return the canonical report location without creating any files."""
    if type(iteration) is not int or iteration < 0:
        raise ValueError("profile report iteration must be a nonnegative integer")
    return Path(work_dir).resolve() / "profile" / f"iter{iteration}" / "bottleneck_analysis.md"


def _inside(path, boundary):
    resolved = path.resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f"profile report path escapes its directory: {path}")
    return resolved


def validate_profile_report_directory(work_dir, iteration):
    """Check paths and competing Markdown reports; the report may be absent.

    This check is read-only so callers can run it before committing a decision.
    Links are resolved before traversal; a link outside this iteration is never
    followed. Directory aliases inside the iteration are visited only once.
    """
    report = profile_report_path(work_dir, iteration)
    work = Path(work_dir).resolve()
    if not work.is_dir():
        raise ValueError(f"profile report work directory does not exist: {work}")
    profile = report.parent.parent
    profile_root = _inside(profile, work)
    if profile_root != profile:
        raise ValueError(f"profile report directory must not alias another directory: {profile}")
    iteration_root = _inside(report.parent, profile_root)
    if iteration_root != report.parent:
        raise ValueError(f"profile iteration directory must not alias another directory: {report.parent}")
    _inside(report, iteration_root)
    for directory in (profile, report.parent):
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"profile report parent is not a directory: {directory}")
    if report.is_symlink():
        raise ValueError(f"profile report must not be a symlink: {report}")
    if report.exists() and not report.is_file():
        raise ValueError(f"profile report is not a regular file: {report}")
    if not report.parent.exists():
        return report

    pending = [report.parent]
    visited = set()
    competitors = []
    while pending:
        directory = pending.pop()
        resolved = directory.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        for entry in sorted(directory.iterdir()):
            _inside(entry, iteration_root)
            if entry.is_dir():
                pending.append(entry)
            elif entry.suffix.lower() in _MARKDOWN_SUFFIXES and entry != report:
                competitors.append(entry.relative_to(report.parent).as_posix())
    if competitors:
        raise ValueError("profile iteration must contain only bottleneck_analysis.md; "
                         "additional Markdown reports: " + ", ".join(sorted(competitors)))
    return report


def _source_block(stage, iteration, decision_path):
    return (
        f"{_SOURCE_START}\n"
        f"> 分析来源：{stage}\n"
        f"> 轮次：{iteration}\n"
        f"> 原始 decision 路径（相对 work）：{decision_path}\n"
        "> 来源标识由工作流程序写入。\n"
        f"{_SOURCE_END}\n\n"
    )


def _without_source_block(content):
    if not content.startswith(_SOURCE_START + "\n"):
        return content
    source_block, separator, body = content.partition(_SOURCE_END + "\n\n")
    if not separator:
        raise ValueError("profile report has an incomplete program source block")
    if "> 分析来源：Stage9" in source_block.splitlines():
        raise ValueError("cannot attribute an existing Stage9 report to Stage7")
    return body


def stamp_stage7_report(work_dir, iteration):
    """Attribute an existing Stage7 report without rewriting its analysis."""
    report = validate_profile_report_directory(work_dir, iteration)
    if not report.is_file():
        raise ValueError(f"Stage7 profile report is missing: {report}")
    original = report.read_text(encoding="utf-8")
    body = _without_source_block(original)
    if not body.strip():
        raise ValueError(f"Stage7 profile report is empty: {report}")
    content = _source_block("Stage7", iteration, "不适用（Stage7 原始报告）") + body
    if content != original:
        atomic_write_text(str(report), content)
    return report


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile report requires nonempty {field}")
    return value


def _stage9_content(iteration, decision, relative_decision):
    """Render supplied validated findings; do not infer metrics or read profiling."""
    if (not isinstance(decision, dict) or type(decision.get("iteration")) is not int
            or decision["iteration"] != iteration):
        raise ValueError("profile report decision must belong to the current iteration")
    ledger = decision.get("ledger_entry")
    if not isinstance(ledger, dict):
        raise ValueError("profile report requires ledger_entry")
    summary = _text(ledger.get("evaluation_summary"), "ledger_entry.evaluation_summary")
    analyses = ledger.get("case_analysis", [])
    suggestions = decision.get("suggest_next", [])
    if not isinstance(analyses, list) or not isinstance(suggestions, list):
        raise ValueError("profile report case_analysis and suggest_next must be lists")
    lines = [
        _source_block("Stage9", iteration, relative_decision).rstrip("\n"), "",
        "# 性能瓶颈分析", "",
        "本报告由程序根据已校验的 Stage9 decision 生成。分析与证据表述来自该 decision；"
        "生成报告时不重新读取原始 profiling，也不补充未提供的指标。", "",
        "## 本轮评测回顾", "", summary, "",
        "## 逐 case 分析", "",
    ]
    for analysis in analyses:
        if not isinstance(analysis, dict):
            raise ValueError("profile report case_analysis entries must be objects")
        case_id = _text(analysis.get("case_id"), "case_analysis.case_id")
        lines.extend(["### " + case_id, ""])
        for key, label in (("observation", "现象"), ("explanation", "原因"),
                           ("evidence", "证据"), ("next_action", "下一步")):
            lines.extend([f"**{label}**", "", _text(analysis.get(key), f"case_analysis.{key}"), ""])
    if not analyses:
        lines.extend(["decision 未提供逐 case 分析。", ""])

    lines.extend([
        "## 后续计划（suggest_next）", "",
        "以下内容仅为计划摘要，不新增或扩大文件修改权限。执行范围、文件约束和完整任务细节"
        "以已校验的原始 decision 及工作流授权为准；逐 case 的下一步也不单独授权修改。", "",
    ])
    for index, task in enumerate(suggestions, 1):
        if not isinstance(task, dict):
            raise ValueError("profile report suggest_next entries must be objects")
        priority = _text(task.get("priority"), "suggest_next.priority")
        action = _text(task.get("action"), "suggest_next.action")
        reason = _text(task.get("reason"), "suggest_next.reason")
        task_id = task.get("task_id", str(index))
        lines.extend([f"### {priority} · {task_id}", "", f"**计划**：{action}", "",
                      f"**依据**：{reason}", ""])
        checks = task.get("acceptance_checks", [])
        if not isinstance(checks, list):
            raise ValueError("profile report acceptance_checks must be a list")
        if checks:
            lines.extend(["**验收**", ""])
            lines.extend("- " + _text(check, "suggest_next.acceptance_checks") for check in checks)
            lines.append("")
    if not suggestions:
        lines.extend(["decision 未提出后续任务。", ""])
    return "\n".join(lines)


def write_stage9_report(work_dir, iteration, decision, decision_path):
    """Publish a report only after the caller has validated its Stage9 decision.

    Rendering and all filesystem checks precede the atomic replacement, so a
    malformed input or conflicting report never destroys the previous report.
    The decision is neither modified nor independently committed here.
    """
    work = Path(work_dir).resolve()
    source = Path(decision_path)
    if not source.is_absolute():
        source = work / source
    source = _inside(source, work)
    if not source.is_file():
        raise ValueError(f"profile report decision source is not a file: {source}")
    relative_decision = source.relative_to(work).as_posix()
    content = _stage9_content(iteration, decision, relative_decision)
    report = validate_profile_report_directory(work, iteration)
    if not report.exists() or report.read_text(encoding="utf-8") != content:
        atomic_write_text(str(report), content)
    return report

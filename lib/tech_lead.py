"""Validate one Tech Lead decision and merge only its model-owned fields.

The orchestrator owns request identity, persistence and measured performance.
This module does no I/O and never mutates either of its input dictionaries.
"""

from copy import deepcopy
import ntpath
import posixpath

from lib.stage9_plan import normalize_decision_plan
from lib.performance_pattern import (PerformancePatternValidationError,
                                     validate_performance_pattern_details)


_OUTPUT_FIELDS = {
    "plan_version",
    "iteration", "request_id", "ledger_entry", "insights", "bottleneck_now",
    "suggest_next", "worst_cases_tracker", "fusion_kernel_strategy",
    "proven_pattern", "regression_pattern", "pitfall",
}
_LEDGER_FIELDS = {"direction", "modify_files", "readonly_files", "fix_plan",
                  "evaluation_summary", "case_analysis", "action_plan"}
_TRANSIENT_FIELDS = {"proven_pattern", "regression_pattern", "pitfall"}


def _nonempty_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _string_list(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    for item in value:
        _nonempty_string(item, field)


def _file_list(value, field):
    """Validate declared work-relative files without guessing paths from prose."""
    _string_list(value, field)
    normalized = []
    for item in value:
        path = item.replace("\\", "/")
        if (item != item.strip() or any(ord(char) < 32 for char in item)
                or ntpath.splitdrive(path)[0] or path.startswith("/")
                or path.endswith("/") or ".." in path.split("/")
                or any(char in path for char in "*?[]<>|:\"")):
            raise ValueError(f"{field}: {item!r} must be a concrete work-relative file; "
                             "absolute paths, parent traversal, directories and globs are not allowed")
        path = posixpath.normpath(path)
        if path in ("", ".", ".."):
            raise ValueError(f"{field}: {item!r} must identify a concrete file")
        if path not in normalized:
            normalized.append(path)
    return normalized


def _paths_overlap(first, second):
    # Prefix checks also stop a declared readonly directory from being bypassed
    # by naming one of its children, even when that directory lacks a slash.
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def validate_advice_file_scope(ledger, suggestions, *, required=False):
    """Return normalized copies; no I/O or prose inference.

    ``required=True`` also rejects pre-contract suggestions at resume boundaries.
    Otherwise wholly legacy decisions pass through until explicitly upgraded.
    """
    if not isinstance(ledger, dict):
        raise ValueError("ledger_entry must be an object")
    if not isinstance(suggestions, list) or any(not isinstance(item, dict) for item in suggestions):
        raise ValueError("suggest_next must be a list of objects")
    if not required and not any(
            "inspect_files" in item or "modify_files" in item for item in suggestions):
        return deepcopy(ledger), deepcopy(suggestions)
    if "fix_plan" in ledger:
        raise ValueError("ledger_entry.fix_plan is not accepted with structured file scope; "
                         "use ledger_entry file lists and suggest_next as the single current plan")
    ledger = deepcopy(ledger)
    for field in ("modify_files", "readonly_files"):
        ledger[field] = _file_list(ledger.get(field), f"ledger_entry.{field}")
    writable = set(ledger["modify_files"])
    readonly = set(ledger["readonly_files"])
    for path in writable:
        conflict = next((other for other in readonly if _paths_overlap(path, other)), None)
        if conflict is not None:
            raise ValueError(f"ledger_entry.modify_files {path!r} overlaps readonly_files {conflict!r}")
    suggestions = deepcopy(suggestions)
    for index, suggestion in enumerate(suggestions):
        label = f"suggest_next[{index}]"
        for field in ("inspect_files", "modify_files"):
            suggestion[field] = _file_list(suggestion.get(field), f"{label}.{field}")
        if not suggestion["inspect_files"] and not suggestion["modify_files"]:
            raise ValueError(f"{label} must identify at least one inspect_files or modify_files target")
        for path in suggestion["modify_files"]:
            conflict = next((other for other in readonly if _paths_overlap(path, other)), None)
            if conflict is not None:
                raise ValueError(f"{label}.modify_files {path!r} conflicts with readonly_files {conflict!r}; "
                                 "resolve the current plan before sending it to Stage3")
            if path not in writable:
                raise ValueError(f"{label}.modify_files {path!r} is not in ledger_entry.modify_files; "
                                 "declare the authorized file or revise this suggestion")
    return ledger, suggestions


def _validate_pattern(output, name, explanation):
    if name not in output:
        return
    value = output[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    errors = []
    for field in ("what_changed", explanation):
        try:
            _nonempty_string(value.get(field), f"{name}.{field}")
        except ValueError as exc:
            errors.append(str(exc))
    if type(value.get("fusion_related")) is not bool:
        errors.append(f"{name}.fusion_related must be a boolean")
    if errors:
        raise Stage9DecisionValidationError(errors)


def _validate_pitfall(output):
    if "pitfall" not in output:
        return
    value = output["pitfall"]
    if not isinstance(value, dict):
        raise ValueError("pitfall must be an object")
    errors = []
    if value.get("verdict") not in ("confirmed", "rejected"):
        errors.append("pitfall.verdict must be confirmed or rejected")
    for field in ("topic", "target_advice", "feedback", "root_cause", "correct_approach"):
        try:
            _nonempty_string(value.get(field), f"pitfall.{field}")
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise Stage9DecisionValidationError(errors)


class Stage9DecisionValidationError(ValueError):
    """Independent decision errors reported together for one repair."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class Stage9ConditionValidationError(Stage9DecisionValidationError):
    """Errors in the current request's scene and experience conditions."""


def validate_stage9_request_conditions(output, *, perf_diff, performance_case_ids, has_question):
    """Check the supplied scene's conditional fields before merging any history.

    An unrequested experience is rejected explicitly, never silently discarded
    or reclassified as a measured performance regression. Legacy standalone
    merge callers retain their original optional-field compatibility.
    """
    errors = []
    for required, field in ((perf_diff.get("has_improvement"), "proven_pattern"),
                            (perf_diff.get("has_regression"), "regression_pattern"),
                            (has_question, "pitfall")):
        if not required:
            if field in output:
                condition = ("this round has no development question.md awaiting adjudication" if field == "pitfall"
                             else "this round did not trigger a comparable performance up/down experience")
                errors.append(f"{field}: {condition}, this field must be omitted; "
                              "the related error reasons and fix rationale are kept in ledger_entry.evaluation_summary and suggest_next and must not be lost")
            continue
        if field not in output:
            errors.append(f"Stage9 is missing this round\'s required {field}; placeholder experience or re-issuing an old plan is forbidden")
            continue
        try:
            if field == "pitfall":
                _validate_pitfall(output)
            else:
                validate_performance_pattern_details(output[field], name=field)
        except PerformancePatternValidationError as exc:
            errors.extend(exc.errors)
        except ValueError as exc:
            errors.extend(getattr(exc, "errors", [str(exc)]))

    ledger = output.get("ledger_entry")
    analyses = ledger.get("case_analysis", []) if isinstance(ledger, dict) else None
    if performance_case_ids is not None and isinstance(analyses, list):
        if not performance_case_ids and analyses:
            errors.append("ledger_entry.case_analysis: this round is not a performance slow-case analysis task; it must be omitted or filled with []; "
                          "per-case causes of build, precision or evaluation errors go into ledger_entry.evaluation_summary and the corresponding suggest_next task")
        elif performance_case_ids:
            reviewed = [entry.get("case_id") for entry in analyses
                        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str)]
            expected, actual = set(performance_case_ids), set(reviewed)
            if expected != actual:
                errors.append(f"ledger_entry.case_analysis must cover the current worst cases: "
                              f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    if errors:
        raise Stage9ConditionValidationError(errors)


def _collect_review_errors(output, iteration, allow_empty_suggestions,
                           performance_case_ids, validate_experience_fields):
    """Check review fields independently of whether the task plan can normalize."""
    errors = []
    if type(output.get("iteration")) is not int or output["iteration"] != iteration:
        errors.append("decision iteration must match the current iteration")
    if output.get("suggest_next") == [] and not allow_empty_suggestions:
        errors.append("suggest_next must contain current P0/P1/P2 suggestions")
    ledger = output.get("ledger_entry")
    analyses = ledger.get("case_analysis", []) if isinstance(ledger, dict) else None
    if isinstance(analyses, list):
        seen = set()
        for index, entry in enumerate(analyses):
            case_id = entry.get("case_id") if isinstance(entry, dict) else None
            if not isinstance(case_id, str) or not case_id.strip():
                continue  # The schema reports malformed children separately.
            if case_id in seen:
                errors.append(f"ledger_entry.case_analysis[{index}] case={case_id}: "
                              "case_analysis must not repeat a case_id")
            seen.add(case_id)
        if performance_case_ids is not None and seen != set(performance_case_ids):
            expected = set(performance_case_ids)
            errors.append(f"ledger_entry.case_analysis must cover the current worst cases: "
                          f"missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}")
    fusion = output.get("fusion_kernel_strategy", [])
    if isinstance(fusion, dict):
        fusion = [fusion]
    if isinstance(fusion, list):
        for index, entry in enumerate(fusion):
            if not isinstance(entry, dict):
                continue
            field = f"fusion_kernel_strategy[{index}]"
            if type(entry.get("iter")) is not int or entry["iter"] != iteration:
                errors.append(f"{field}: fusion_kernel_strategy entries must belong to the current iteration")
            for name in ("direction", "strategy", "evidence", "status"):
                if name in entry:
                    try:
                        _nonempty_string(entry[name], f"{field}.{name}")
                    except ValueError as exc:
                        errors.append(str(exc))
    checks = []
    if validate_experience_fields:
        checks.extend((lambda: _validate_pattern(output, "proven_pattern", "why_it_worked"),
                       lambda: _validate_pattern(output, "regression_pattern", "why_it_failed"),
                       lambda: _validate_pitfall(output)))
    for check in checks:
        try:
            check()
        except ValueError as exc:
            errors.extend(getattr(exc, "errors", [str(exc)]))
    return errors


def merge_tech_lead_update(base_history: dict, output: dict, iteration: int, *,
                           reason: str = "", allow_empty_suggestions: bool = False,
                           require_evaluation_summary=False,
                           performance_case_ids=None, require_file_scope=False,
                           require_structured_plan=False, known_case_ids=None,
                           validate_experience_fields=True) -> dict:
    """Return an independently owned history after a valid current decision.

    Insights and current conclusions keep their existing replacement semantics;
    case tracking is updated by case ID, with current analysis archived per round.
    fusion attempts are appended without removing historical entries. Performance
    fields and old ledger entries always come from the caller's history snapshot.
    Optional experience/correction payloads are validated here but deliberately
    not stored: the caller consumes those payloads directly from this decision.

    ``ValueError`` rejects invalid decisions before any caller-visible mutation.
    Empty suggestions are accepted only when the caller has confirmed exit.
    New workflow decisions require explicit inspect/modify file lists; old
    standalone callers remain compatible until they opt in or supply scopes.
    The workflow disables duplicate experience checks only after running the
    complete request-condition validator, which rejects unrequested fields
    without also asking the model to fill them in.
    """
    if not isinstance(base_history, dict) or not isinstance(output, dict):
        raise ValueError("base_history and output must be objects")
    # A malformed task must not conceal independent review errors until
    # the next attempt. No partially normalized output is accepted or merged.
    errors = []
    normalized = None
    try:
        normalized = normalize_decision_plan(
            output, known_case_ids=known_case_ids, required=require_structured_plan)
    except ValueError as exc:
        errors.extend(getattr(exc, "errors", [str(exc)]))
    if require_structured_plan or output.get("plan_version") == 2:
        errors.extend(_collect_review_errors(output, iteration, allow_empty_suggestions,
                                             performance_case_ids,
                                             validate_experience_fields))
    if errors:
        raise Stage9DecisionValidationError(list(dict.fromkeys(errors)))
    output = normalized
    if type(iteration) is not int or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if type(output.get("iteration")) is not int or output["iteration"] != iteration:
        raise ValueError("decision iteration must match the current iteration")
    if not isinstance(reason, str) or type(allow_empty_suggestions) is not bool:
        raise ValueError("reason must be a string and allow_empty_suggestions a boolean")
    unexpected = set(output) - _OUTPUT_FIELDS
    if unexpected:
        raise ValueError(f"unsupported decision fields: {', '.join(sorted(map(str, unexpected)))}")

    ledger_update = output.get("ledger_entry")
    if not isinstance(ledger_update, dict):
        raise ValueError("ledger_entry must be an object")
    unexpected = set(ledger_update) - _LEDGER_FIELDS
    if unexpected:
        raise ValueError(f"unsupported ledger_entry fields: {', '.join(sorted(map(str, unexpected)))}")
    _nonempty_string(ledger_update.get("direction"), "ledger_entry.direction")
    for field in ("modify_files", "readonly_files"):
        _string_list(ledger_update.get(field), f"ledger_entry.{field}")
    # Treat slash variants and ./ aliases as the same path. Case is preserved
    # because workflow implementations may run on a case-sensitive filesystem.
    def path_key(path):
        return posixpath.normpath(path.replace("\\", "/"))

    writable = {path_key(path) for path in ledger_update["modify_files"]}
    readonly = {path_key(path) for path in ledger_update["readonly_files"]}
    if writable & readonly:
        raise ValueError("modify_files and readonly_files must not overlap")
    if "fix_plan" in ledger_update and not isinstance(ledger_update["fix_plan"], dict):
        raise ValueError("ledger_entry.fix_plan must be an object")
    if require_evaluation_summary or "evaluation_summary" in ledger_update:
        _nonempty_string(ledger_update.get("evaluation_summary"), "ledger_entry.evaluation_summary")
    # Performance scenes supply measured case IDs; error scenes supply [].
    # None keeps compatibility for callers validating old standalone decisions.
    case_analysis = ledger_update.get("case_analysis", [])
    if not isinstance(case_analysis, list):
        raise ValueError("ledger_entry.case_analysis must be a list")
    reviewed_ids = set()
    for analysis in case_analysis:
        if not isinstance(analysis, dict):
            raise ValueError("ledger_entry.case_analysis entries must be objects")
        case_id = analysis.get("case_id")
        _nonempty_string(case_id, "case_analysis.case_id")
        if case_id in reviewed_ids:
            raise ValueError("case_analysis must not repeat a case_id")
        reviewed_ids.add(case_id)
        for field in ("observation", "explanation", "evidence", "next_action"):
            _nonempty_string(analysis.get(field), f"case_analysis.{field}")
    if performance_case_ids is not None:
        expected_ids = set(performance_case_ids)
        if reviewed_ids != expected_ids:
            missing = sorted(expected_ids - reviewed_ids)
            unexpected_cases = sorted(reviewed_ids - expected_ids)
            raise ValueError(f"case_analysis must cover the current worst cases: missing={missing}, "
                             f"unexpected={unexpected_cases}")

    suggestions = output.get("suggest_next")
    if not isinstance(suggestions, list) or (not suggestions and not allow_empty_suggestions):
        raise ValueError("suggest_next must contain current P0/P1/P2 suggestions")
    for suggestion in suggestions:
        if not isinstance(suggestion, dict) or suggestion.get("priority") not in ("P0", "P1", "P2"):
            raise ValueError("suggest_next entries must have priority P0, P1 or P2")
        _nonempty_string(suggestion.get("action"), "suggest_next.action")
        _nonempty_string(suggestion.get("reason"), "suggest_next.reason")
    ledger_update, suggestions = validate_advice_file_scope(
        ledger_update, suggestions, required=require_file_scope)
    if "insights" in output:
        _string_list(output["insights"], "insights")
    if "bottleneck_now" in output and not isinstance(output["bottleneck_now"], str):
        raise ValueError("bottleneck_now must be a string")
    if "worst_cases_tracker" in output and not isinstance(output["worst_cases_tracker"], dict):
        raise ValueError("worst_cases_tracker must be an object")

    fusion_updates = output.get("fusion_kernel_strategy", [])
    if isinstance(fusion_updates, dict):
        fusion_updates = [fusion_updates]
    if not isinstance(fusion_updates, list):
        raise ValueError("fusion_kernel_strategy must be an object or a list of objects")
    for entry in fusion_updates:
        if (not isinstance(entry, dict) or type(entry.get("iter")) is not int
                or entry["iter"] != iteration):
            raise ValueError("fusion_kernel_strategy entries must belong to the current iteration")
        for field in ("direction", "strategy", "evidence", "status"):
            if field in entry:
                _nonempty_string(entry[field], f"fusion_kernel_strategy.{field}")

    if validate_experience_fields:
        _validate_pattern(output, "proven_pattern", "why_it_worked")
        _validate_pattern(output, "regression_pattern", "why_it_failed")
        _validate_pitfall(output)
    ledger = base_history.get("ledger", [])
    if not isinstance(ledger, list) or any(not isinstance(entry, dict) for entry in ledger):
        raise ValueError("base_history.ledger must be a list of objects")
    if not isinstance(base_history.get("fusion_kernel_strategy", []), list):
        raise ValueError("base_history.fusion_kernel_strategy must be a list")

    merged = deepcopy(base_history)
    if output.get("plan_version") == 2:
        merged["plan_version"] = 2
    for field in list(merged):
        if field in _TRANSIENT_FIELDS or (isinstance(field, str) and field.startswith("_pending_")):
            merged.pop(field)
    for field in ("insights", "bottleneck_now"):
        if field in output:
            merged[field] = deepcopy(output[field])
    merged["suggest_next"] = deepcopy(suggestions)
    tracker_updates = output.get("worst_cases_tracker", {})
    if tracker_updates or case_analysis:
        tracker = merged.get("worst_cases_tracker", {})
        # Older display-compatible list trackers have no reliable case keys.
        # Preserve them explicitly instead of dropping them during the upgrade.
        if not isinstance(tracker, dict):
            merged.setdefault("legacy_worst_cases_tracker", deepcopy(tracker))
            tracker = {}
        else:
            tracker = deepcopy(tracker)
        tracker.update(deepcopy(tracker_updates))
        for analysis in case_analysis:
            tracker[analysis["case_id"]] = {
                **deepcopy(analysis), "iteration": iteration,
            }
        merged["worst_cases_tracker"] = tracker

    merged_ledger = merged.setdefault("ledger", [])
    current = next((entry for entry in reversed(merged_ledger)
                    if type(entry.get("iter")) is int and entry["iter"] == iteration), None)
    if current is None:
        current = {"iter": iteration, "reason": "", "avg_speedup_before": None,
                   "avg_speedup_after": None, "delta": None, "verdict": None}
        merged_ledger.append(current)
    # A missing optional plan must not carry forward a previous invocation's
    # nested plan while the current top-level direction/files have changed.
    current.pop("fix_plan", None)
    current.pop("evaluation_summary", None)
    current.pop("case_analysis", None)
    current.pop("action_plan", None)
    current.update(deepcopy(ledger_update))
    if reason.strip():
        current["reason"] = reason.strip().splitlines()[0]

    attempts = merged.setdefault("fusion_kernel_strategy", [])
    for entry in fusion_updates:
        if entry not in attempts:
            attempts.append(deepcopy(entry))
    return merged

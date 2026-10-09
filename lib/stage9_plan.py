"""Program-owned Stage9 action-plan contract, without I/O or new dependencies.

Models submit one detailed task list. File permissions and the Stage3 handoff
are derived from that list, never independently authored duplicate plans.
Structural checks establish consistency of the declarations; they do not prove
that the model's case-to-code attribution or proposed optimization is correct.
"""

from copy import deepcopy
from bisect import bisect_left
import ntpath
import posixpath
import re

from lib.performance_pattern import (build_performance_pattern_schema,
                                     build_performance_pattern_template)


def _object(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


def _array(items, *, minimum=0, unique=False):
    result = {"type": "array", "items": items, "minItems": minimum}
    if unique:
        result["uniqueItems"] = True
    return result


_TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
_STRING = {"type": "string"}
_PATH = {**_TEXT, "description": "Concrete file relative to the work directory; "
         "no absolute paths, parent traversal, directories (even for inspect) or globs."}
_PATHS = _array(_PATH, unique=True)
_CASE_ANALYSIS = _object(
    {key: _TEXT for key in ("case_id", "observation", "explanation", "evidence", "next_action")},
    ("case_id", "observation", "explanation", "evidence", "next_action"))
_ROUTE_EVIDENCE = _object(
    {"file": _PATH, "location": _TEXT, "explanation": _TEXT},
    ("file", "location", "explanation"))
_BINDING = _object(
    {"case_id": _TEXT, "implementation_files": _array(_PATH, minimum=1, unique=True),
     "route_evidence": _array(_ROUTE_EVIDENCE, minimum=1)},
    ("case_id", "implementation_files", "route_evidence"))
_CHANGE = _object(
    {"file": _PATH, "operation": {"enum": ["inspect", "modify", "create"]},
     "location": _TEXT, "method": _TEXT},
    ("file", "operation", "location", "method"))
_TASK = _object(
    {"task_id": _TEXT, "priority": {"enum": ["P0", "P1", "P2"]},
     "task_type": {"enum": ["inspect", "modify"]}, "action": _TEXT, "reason": _TEXT,
     "case_scope": {"enum": ["cases", "operator"]},
     "target_cases": _array(_TEXT, unique=True), "operator_reason": _STRING,
     "case_bindings": _array(_BINDING), "changes": _array(_CHANGE, minimum=1),
     "acceptance_checks": _array(_TEXT, minimum=1)},
    ("task_id", "priority", "task_type", "action", "reason", "case_scope", "target_cases",
     "operator_reason", "case_bindings", "changes", "acceptance_checks"))
_LEDGER = _object(
    {"evaluation_summary": _TEXT, "direction": _TEXT, "readonly_files": _PATHS,
     "case_analysis": _array(_CASE_ANALYSIS)},
    ("evaluation_summary", "direction", "readonly_files"))
_PITFALL = {
    "type": "object",
    "properties": {"verdict": {"enum": ["confirmed", "rejected"]},
                   **{key: _TEXT for key in ("topic", "target_advice", "feedback",
                                            "root_cause", "correct_approach")}},
    "required": ["verdict", "topic", "target_advice", "feedback", "root_cause", "correct_approach"],
}
_DECISION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Stage9 decision and executable task plan, version 2",
    "description": "Author one task list. The program derives writable/inspect lists and "
                   "the history action_plan; never submit those derived fields.",
    **_object({
        "plan_version": {"type": "integer", "const": 2},
        "iteration": {"type": "integer", "minimum": 0}, "request_id": _TEXT,
        "ledger_entry": _LEDGER, "suggest_next": _array(_TASK),
        "insights": _array(_TEXT), "bottleneck_now": _STRING,
        "worst_cases_tracker": {"type": "object"},
        "fusion_kernel_strategy": {"anyOf": [
            {"type": "object"}, _array({"type": "object"})]},
        "proven_pattern": {"type": "object"}, "regression_pattern": {"type": "object"},
        "pitfall": {"type": "object"},
    }, ("plan_version", "iteration", "request_id", "ledger_entry", "suggest_next")),
}


def _required_performance_patterns(perf_diff):
    return [name for flag, name in (("has_improvement", "proven_pattern"),
                                    ("has_regression", "regression_pattern"))
            if (perf_diff or {}).get(flag)]


def _independent_schema_tree(value):
    """Copy every JSON container occurrence without preserving shared aliases.

    ``deepcopy`` preserves aliases such as the reused _TEXT dictionaries. A
    caller binding request_id.const must not also constrain every text field.
    """
    if isinstance(value, dict):
        return {key: _independent_schema_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_independent_schema_tree(child) for child in value]
    return value


def build_decision_schema(perf_diff=None, *, performance_case_ids=None, has_question=None):
    """Return the structural contract plus this request's conditional experience."""
    schema = _independent_schema_tree(_DECISION_SCHEMA)
    for name in _required_performance_patterns(perf_diff):
        schema["properties"][name] = build_performance_pattern_schema(name)
        schema["required"].append(name)
    if perf_diff is not None:
        for name in ("proven_pattern", "regression_pattern"):
            if name not in schema["required"]:
                schema["properties"].pop(name)
    if has_question is not None:
        if has_question:
            schema["properties"]["pitfall"] = deepcopy(_PITFALL)
            schema["required"].append("pitfall")
        else:
            schema["properties"].pop("pitfall")
    if performance_case_ids is not None:
        ledger = schema["properties"]["ledger_entry"]
        cases = ledger["properties"]["case_analysis"]
        case_ids = list(dict.fromkeys(performance_case_ids))
        cases["maxItems"] = len(case_ids)
        if case_ids:
            ledger["required"].append("case_analysis")
            cases["minItems"] = len(case_ids)
            # Replace the shared text node instead of attaching an enum to it.
            cases["items"]["properties"]["case_id"] = {**_TEXT, "enum": case_ids}
            cases["description"] = "One analysis per listed performance case; do not repeat case_id."
        else:
            cases["description"] = (
                "No performance-case analysis is requested. Omit or use []; "
                "put build/precision/error findings in evaluation_summary and suggest_next.")
    return _independent_schema_tree(schema)


class DecisionPlanValidationError(ValueError):
    """Independent plan errors, without producing a partially accepted plan."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _has_duplicate(values):
    """Keep scalar case/path lists linear while handling malformed JSON items."""
    seen, containers = set(), []
    for value in values:
        try:
            if value in seen:
                return True
            seen.add(value)
        except TypeError:
            if value in containers:
                return True
            containers.append(value)
    return False


def _collect_schema_errors(value, schema, field):
    """Check independent sibling fields; stop only an incorrectly typed subtree."""
    errors = []
    if "anyOf" in schema:
        if not any(not _collect_schema_errors(value, candidate, field)
                   for candidate in schema["anyOf"]):
            return [f"{field} has an invalid value type"]
    kind = schema.get("type")
    matches = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "boolean": lambda: type(value) is bool,
    }
    if kind is not None and not matches[kind]():
        return [f"{field} must be {kind}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{field} must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{field} must be one of {schema['enum']}")
    if kind == "object":
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            errors.append(f"{field} is missing required fields: {', '.join(missing)}")
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{field} must contain at least {schema['minProperties']} property/properties")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = set(value) - set(properties)
            if unexpected:
                errors.append(f"{field} has unsupported fields: "
                              f"{', '.join(sorted(map(str, unexpected)))}")
        for key, child in properties.items():
            if key in value:
                errors.extend(_collect_schema_errors(value[key], child, f"{field}.{key}"))
    elif kind == "array":
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{field} must contain at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{field} must contain at most {schema['maxItems']} item(s)")
        if schema.get("uniqueItems") and _has_duplicate(value):
            errors.append(f"{field} must not contain duplicate items")
        for index, item in enumerate(value):
            errors.extend(_collect_schema_errors(item, schema.get("items", {}), f"{field}[{index}]"))
    elif kind == "string":
        if (len(value) < schema.get("minLength", 0)
                or ("pattern" in schema and not re.search(schema["pattern"], value))):
            errors.append(f"{field} must be a nonempty string")
    elif kind == "integer" and "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{field} must be at least {schema['minimum']}")
    return errors


def _validate_schema(value, schema, field):
    """Strictly validate the same schema, reporting every independent error."""
    errors = _collect_schema_errors(value, schema, field)
    if errors:
        raise DecisionPlanValidationError(errors)


def _path(value, field):
    path = value.replace("\\", "/")
    if (value != value.strip() or any(ord(char) < 32 for char in value)
            or ntpath.splitdrive(path)[0] or path.startswith("/") or path.endswith("/")
            or ".." in path.split("/") or any(char in path for char in '*?[]<>|:"')):
        raise ValueError(f"{field}: {value!r} must be a concrete work-relative file; "
                         "absolute paths, parent traversal, directories and globs are not allowed")
    path = posixpath.normpath(path)
    if path in ("", ".", ".."):
        raise ValueError(f"{field}: {value!r} must identify a concrete file")
    return path


def _path_list(values, field):
    paths = [_path(value, f"{field}[{index}]") for index, value in enumerate(values)]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{field} must not contain duplicate file aliases")
    return paths


def _overlap(first, second):
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def _unique(values):
    return list(dict.fromkeys(values))


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def _diagnostic_path(value, field, errors):
    # Type and empty-text errors already come from the shared schema collector.
    if not _nonempty_text(value):
        return None
    try:
        return _path(value, field)
    except ValueError as exc:
        errors.append(str(exc))
        return None


def _diagnostic_paths(values, field, errors):
    if not isinstance(values, list):
        return [], False
    paths, complete = [], True
    for index, value in enumerate(values):
        path = _diagnostic_path(value, f"{field}[{index}]", errors)
        if path is None:
            complete = False
        else:
            paths.append(path)
    if len(set(paths)) != len(paths):
        # Raw duplicates are already diagnosed by the schema; normalized aliases
        # such as a.py and ./a.py need a separate path-level explanation.
        if not _has_duplicate(values):
            errors.append(f"{field} must not contain duplicate file aliases")
        # Duplicate entries are invalid, but the declared set is still known;
        # they must not hide an independent outside-scope or missing-operation error.
    return _unique(paths), complete


def _readonly_matches(path, readonly_set, readonly_sorted):
    """Find all prefix conflicts without scanning every protected file per write."""
    parts = path.split("/")
    result = [parent for index in range(1, len(parts) + 1)
              if (parent := "/".join(parts[:index])) in readonly_set]
    prefix = path + "/"
    index = bisect_left(readonly_sorted, prefix)
    while index < len(readonly_sorted) and readonly_sorted[index].startswith(prefix):
        result.append(readonly_sorted[index])
        index += 1
    return result


def collect_decision_plan_errors(output, known_case_ids=None, required=False):
    """Collect schema, path and task errors without accepting or changing a plan.

    Malformed subtrees do not hide independent sibling errors. Positive evidence
    of a conflict can be reported from valid entries in an incomplete list, but
    absence/membership claims wait until their participating lists are complete.
    No partial plan, permission list or action_plan is returned to the caller.
    """
    if not isinstance(output, dict):
        return ["decision must be an object"]
    if "plan_version" not in output and not required:
        return []
    errors = _collect_schema_errors(output, _DECISION_SCHEMA, "decision")
    if "plan_version" not in output:
        errors.insert(0, "decision.plan_version=2 is required; regenerate the structured Stage9 plan")
    ledger = output.get("ledger_entry")
    readonly, _ = _diagnostic_paths(
        ledger.get("readonly_files") if isinstance(ledger, dict) else None,
        "ledger_entry.readonly_files", errors)
    readonly_set, readonly_sorted = set(readonly), sorted(readonly)
    known = set(known_case_ids) if known_case_ids is not None else None
    tasks = output.get("suggest_next")
    if not isinstance(tasks, list):
        return _unique(errors)
    seen_task_ids = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        label = f"suggest_next[{index}]" + (f" ({task_id})" if _nonempty_text(task_id) else "")
        if _nonempty_text(task_id):
            if task_id in seen_task_ids:
                errors.append(f"{label}.task_id must be unique")
            seen_task_ids.add(task_id)
        cases, bindings = task.get("target_cases"), task.get("case_bindings")
        case_scope, task_type = task.get("case_scope"), task.get("task_type")
        operator_reason = task.get("operator_reason")
        cases_complete = isinstance(cases, list) and all(_nonempty_text(case) for case in cases)
        valid_cases = [case for case in cases if _nonempty_text(case)] if isinstance(cases, list) else []
        if case_scope == "operator":
            if ((isinstance(cases, list) and cases) or (isinstance(bindings, list) and bindings)
                    or isinstance(operator_reason, str) and not operator_reason.strip()):
                errors.append(f"{label}: operator scope requires operator_reason and empty "
                              "target_cases/case_bindings")
        elif case_scope == "cases":
            if (isinstance(cases, list) and not cases
                    or isinstance(operator_reason, str) and operator_reason):
                errors.append(f"{label}: cases scope requires target_cases and empty operator_reason")
            if known is not None and set(valid_cases) - known:
                errors.append(f"{label}.target_cases contains unknown case IDs: "
                              f"{sorted(set(valid_cases) - known)}")

        implementation_files, binding_ids, binding_records = set(), [], []
        implementation_complete = binding_ids_complete = isinstance(bindings, list)
        for binding_index, binding in enumerate(bindings if isinstance(bindings, list) else []):
            if not isinstance(binding, dict):
                implementation_complete = binding_ids_complete = False
                continue
            binding_label = f"{label}.case_bindings[{binding_index}]"
            case_id = binding.get("case_id")
            if _nonempty_text(case_id):
                binding_ids.append(case_id)
            else:
                binding_ids_complete = False
            files, complete = _diagnostic_paths(
                binding.get("implementation_files"), f"{binding_label}.implementation_files", errors)
            complete = complete and bool(files)
            implementation_complete = implementation_complete and complete
            implementation_files.update(files)
            binding_records.append((case_id, set(files), complete))
            route_evidence = binding.get("route_evidence")
            for evidence_index, evidence in enumerate(route_evidence if isinstance(route_evidence, list) else []):
                if isinstance(evidence, dict):
                    _diagnostic_path(evidence.get("file"),
                                     f"{binding_label}.route_evidence[{evidence_index}].file", errors)
        coverage_complete = cases_complete and binding_ids_complete
        coverage_matches = (coverage_complete and len(set(binding_ids)) == len(binding_ids)
                            and set(binding_ids) == set(cases))
        if case_scope == "cases" and coverage_complete and not coverage_matches:
            errors.append(f"{label}.case_bindings must cover target_cases exactly once")

        changes = task.get("changes")
        changes_complete = isinstance(changes, list)
        writes, inspections = [], []
        declared_writes = False
        for change_index, change in enumerate(changes if isinstance(changes, list) else []):
            if not isinstance(change, dict):
                changes_complete = False
                continue
            path = _diagnostic_path(change.get("file"), f"{label}.changes[{change_index}].file", errors)
            operation = change.get("operation")
            operation_valid = isinstance(operation, str) and operation in {"inspect", "modify", "create"}
            if operation_valid and operation in {"modify", "create"}:
                declared_writes = True
            if path is None or not operation_valid:
                changes_complete = False
                continue
            (inspections if operation == "inspect" else writes).append(path)
        writes, inspections = _unique(writes), _unique(inspections)
        if task_type == "inspect" and declared_writes:
            errors.append(f"{label}: inspect task cannot contain modify/create operations")
        if task_type == "modify" and changes_complete and not writes:
            errors.append(f"{label}: modify task requires at least one modify/create operation")
        for path in writes:
            for protected in _readonly_matches(path, readonly_set, readonly_sorted):
                errors.append(f"{label}.changes writes {path!r}, which conflicts with "
                              f"ledger_entry.readonly_files {protected!r}")
            if case_scope == "cases" and implementation_complete and path not in implementation_files:
                errors.append(f"{label}.changes writes {path!r} outside its "
                              "case_bindings.implementation_files")
        if (case_scope == "cases" and task_type in ("modify", "inspect")
                and coverage_matches and changes_complete):
            operations = set(writes if task_type == "modify" else inspections)
            for case_id, files, complete in binding_records:
                if complete and not operations.intersection(files):
                    errors.append(f"{label}: case {case_id!r} has no "
                                  f"{task_type} operation on its bound implementation_files")
    return _unique(errors)


def decision_file_targets_for_diagnostics(output):
    """Copy valid v2 structure and paths for independent file diagnostics only.

    This deliberately does not accept task semantics or grant file permissions.
    A missing implementation operation, conflicting scope or invalid case binding
    remains unchanged so the ordinary validator can reject it. The derived file
    lists only let the read-only filesystem checker report independent problems
    in the same repair request. Never use this result as an authorized handoff;
    no action_plan is generated, and normal acceptance checks remain mandatory.
    """
    _validate_schema(output, _DECISION_SCHEMA, "decision")
    result = deepcopy(output)
    ledger = result["ledger_entry"]
    ledger["readonly_files"] = _path_list(ledger["readonly_files"], "ledger_entry.readonly_files")
    writable = []
    for index, task in enumerate(result["suggest_next"]):
        label = f"suggest_next[{index}] ({task['task_id']})"
        for binding_index, binding in enumerate(task["case_bindings"]):
            binding_label = f"{label}.case_bindings[{binding_index}]"
            binding["implementation_files"] = _path_list(
                binding["implementation_files"], f"{binding_label}.implementation_files")
            for evidence_index, evidence in enumerate(binding["route_evidence"]):
                evidence["file"] = _path(evidence["file"],
                                         f"{binding_label}.route_evidence[{evidence_index}].file")
        for change_index, change in enumerate(task["changes"]):
            change["file"] = _path(change["file"], f"{label}.changes[{change_index}].file")
        task["modify_files"] = _unique(change["file"] for change in task["changes"]
                                       if change["operation"] in {"modify", "create"})
        task["inspect_files"] = _unique(change["file"] for change in task["changes"]
                                        if change["operation"] == "inspect")
        writable.extend(task["modify_files"])
    ledger["modify_files"] = _unique(writable)
    return result


def normalize_decision_plan(output, *, known_case_ids=None, required=False):
    """Validate one raw model decision; derive all redundant file permissions.

    Unversioned legacy data may pass only when the caller has not opted into the
    new contract. A supplied but unsupported version never bypasses validation.
    ``known_case_ids`` is the complete authoritative case set, not just the six
    slowest cases. It may be unavailable for legacy/early error scenes.
    """
    errors = collect_decision_plan_errors(output, known_case_ids=known_case_ids, required=required)
    if errors:
        raise DecisionPlanValidationError(errors)
    if "plan_version" not in output:
        return deepcopy(output)
    # Derive the same canonical handoff only after every semantic/schema/path
    # check passes. The diagnostic helper alone must never authorize a plan.
    result = decision_file_targets_for_diagnostics(output)
    raw_tasks = [{key: deepcopy(value) for key, value in task.items()
                  if key not in {"inspect_files", "modify_files"}}
                 for task in result["suggest_next"]]
    result["ledger_entry"]["action_plan"] = {"version": 2, "tasks": raw_tasks}
    return result


def validate_current_plan(ledger, suggestions, *, known_case_ids=None):
    """Verify a persisted handoff, including every derived compatibility view.

    Returns independent copies and never quietly repairs a changed handoff.
    The caller must route absent/old/inconsistent plans back through Stage9.
    """
    if not isinstance(ledger, dict):
        raise ValueError("current ledger must be an object")
    plan = ledger.get("action_plan")
    if not isinstance(plan, dict) or set(plan) != {"version", "tasks"}:
        raise ValueError("current ledger.action_plan version 2 and tasks are required")
    if type(plan["version"]) is not int or plan["version"] != 2:
        raise ValueError("current ledger.action_plan.version must be 2")
    authored = {key: deepcopy(ledger[key]) for key in _LEDGER["properties"] if key in ledger}
    normalized = normalize_decision_plan(
        {"plan_version": 2, "iteration": 0, "request_id": "resume-validation",
         "ledger_entry": authored, "suggest_next": deepcopy(plan["tasks"])},
        known_case_ids=known_case_ids, required=True)
    if ledger.get("modify_files") != normalized["ledger_entry"]["modify_files"]:
        raise ValueError("current ledger.modify_files differs from action_plan derived file scope")
    if plan != normalized["ledger_entry"]["action_plan"]:
        raise ValueError("current ledger.action_plan is not the canonical validated task plan")
    if suggestions != normalized["suggest_next"]:
        raise ValueError("current suggest_next differs from action_plan tasks and derived file scope")
    return deepcopy(ledger), deepcopy(suggestions)


def build_decision_template(iteration, request_id, performance_case_ids,
                            allow_empty_suggestions=False, *, perf_diff=None, has_question=False):
    """Return an intentionally incomplete model worksheet, not a valid decision.

    Empty text means the model must provide its own evidence and implementation
    details. The template contains no fabricated analysis or writable files.
    """
    case_ids = list(performance_case_ids or [])
    task_cases = case_ids[:1]
    task = {"task_id": "T1", "priority": "P1", "task_type": "modify", "action": "",
            "reason": "", "case_scope": "cases" if task_cases else "operator",
            "target_cases": task_cases, "operator_reason": "",
            "case_bindings": [{"case_id": case_id, "implementation_files": [""],
                               "route_evidence": [{"file": "", "location": "", "explanation": ""}]}
                              for case_id in task_cases],
            "changes": [{"file": "", "operation": "modify", "location": "", "method": ""}],
            "acceptance_checks": [""]}
    template = {"plan_version": 2, "iteration": iteration, "request_id": request_id,
                "ledger_entry": {"evaluation_summary": "", "direction": "", "readonly_files": [],
                                 "case_analysis": [dict(case_id=case_id, observation="", explanation="",
                                                        evidence="", next_action="") for case_id in case_ids]},
                "suggest_next": [] if allow_empty_suggestions else [task]}
    for name in _required_performance_patterns(perf_diff):
        template[name] = build_performance_pattern_template(name)
    if has_question:
        template["pitfall"] = {key: "" for key in _PITFALL["required"]}
    return template

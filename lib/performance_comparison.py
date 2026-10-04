"""Pure validation of archived measurements before single-round comparisons."""

from copy import deepcopy
import math


# Keep this schema aligned with orchestrator.comparison_context. Old reports must
# carry their own complete context; the active context never fills their gaps.
_CONTEXT_FIELDS = (
    "framework", "backend", "runtime_versions",
    "hardware", "evaluation_device_id", "hardware_config", "task_sha256",
    "metric", "baseline_protocol", "warmup", "repeat", "evaluator_sha256",
    "baseline_metadata_sha256",
)
_CONTEXT_LABELS = {
    "framework": "target framework",
    "backend": "runtime backend",
    "runtime_versions": "runtime versions",
    "hardware": "hardware info",
    "evaluation_device_id": "evaluation device",
    "hardware_config": "hardware configuration",
    "task_sha256": "task and case definitions",
    "metric": "timing protocol",
    "baseline_protocol": "baseline protocol",
    "warmup": "warmup count",
    "repeat": "repeat count",
    "evaluator_sha256": "evaluator code",
    "baseline_metadata_sha256": "baseline metadata",
}


def _positive_finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value > 0 and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _invalid(reason, field):
    return None, reason, [field]


def _validated_signature(perf):
    if not isinstance(perf, dict):
        return _invalid("the performance report is not an object", "report")
    context = perf.get("comparison_context")
    if not isinstance(context, dict) or not context:
        return _invalid("the archived comparison context is incomplete; an old report\'s protocol cannot be inferred", "comparison_context")
    for field in _CONTEXT_FIELDS:
        if field not in context:
            return _invalid(f"the archived comparison context is missing field {field}", f"comparison_context.{field}")
    if perf.get("comparison_context_stable") is not True:
        return _invalid("context stability proof is missing, or the context changed between measurements", "comparison_context_stable")
    if not isinstance(context["hardware"], dict) or not context["hardware"]:
        return _invalid("the archived hardware info must be a nonempty object", "comparison_context.hardware")
    if not isinstance(context["hardware_config"], dict):
        return _invalid("the archived hardware configuration must be an object", "comparison_context.hardware_config")
    for field in ("framework", "backend", "task_sha256", "metric", "baseline_protocol", "evaluator_sha256"):
        if not _nonempty_string(context[field]):
            return _invalid(f"the archived {_CONTEXT_LABELS[field]} must be a nonempty string", f"comparison_context.{field}")
    versions = context["runtime_versions"]
    if (not isinstance(versions, dict) or not versions
            or any(not _nonempty_string(key) or not _nonempty_string(value)
                   or value.lower() == "unknown" for key, value in versions.items())):
        return _invalid("the archived runtime versions must be a nonempty version object", "comparison_context.runtime_versions")
    metadata = context["baseline_metadata_sha256"]
    if metadata is not None and not _nonempty_string(metadata):
        return _invalid("the archived baseline metadata fingerprint must be a nonempty string or None", "comparison_context.baseline_metadata_sha256")
    device = context["evaluation_device_id"]
    if not ((type(device) is int and device >= 0) or _nonempty_string(device)):
        return _invalid("the archived evaluation device identifier is invalid", "comparison_context.evaluation_device_id")
    for field, minimum in (("warmup", 0), ("repeat", 1)):
        if type(context[field]) is not int or context[field] < minimum:
            return _invalid(f"the archived {_CONTEXT_LABELS[field]} must be an integer >= {minimum}", f"comparison_context.{field}")
    for field in ("_error", "score_error", "score_error_code"):
        if perf.get(field):
            return _invalid(f"the performance report contains an evaluation error {field}", field)
    if not _positive_finite(perf.get("avg_speedup")):
        return _invalid("avg_speedup must be a positive finite number", "avg_speedup")
    total = perf.get("total_cases")
    if type(total) is not int or total <= 0:
        return _invalid("total_cases must be a strictly positive integer", "total_cases")
    cases = perf.get("cases")
    if not isinstance(cases, list) or len(cases) != total:
        return _invalid("the case list is incomplete or inconsistent with total_cases", "cases")
    baselines = {}
    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            return _invalid(f"case {index + 1} is not an object", path)
        case_id = case.get("case_id")
        if not _nonempty_string(case_id):
            return _invalid(f"case {index + 1} lacks a nonempty case_id", f"{path}.case_id")
        if case_id in baselines:
            return _invalid(f"duplicate case_id: {case_id}", f"{path}.case_id")
        if case.get("status") != "success":
            return _invalid(f"case {case_id} did not complete successfully", f"{path}.status")
        for field in ("speedup", "elapsed_us"):
            if not _positive_finite(case.get(field)):
                return _invalid(f"case {case_id}\'s {field} must be a positive finite number", f"{path}.{field}")
        baseline_field = "baseline_perf_us" if "baseline_perf_us" in case else "baseline_us"
        baseline = case.get(baseline_field)
        if not _positive_finite(baseline):
            return _invalid(f"case {case_id} lacks a positive finite actual baseline timing", f"{path}.{baseline_field}")
        if "baseline_perf_us" in case and "baseline_us" in case:
            if not _positive_finite(case["baseline_us"]) or baseline != case["baseline_us"]:
                return _invalid(f"case {case_id}\'s two baseline timing fields disagree", f"{path}.baseline_us")
        baselines[case_id] = baseline
    return {
        "comparison_context": deepcopy(context),
        "case_baselines": {case_id: baselines[case_id] for case_id in sorted(baselines)},
    }, "the performance report is complete and contains a stable archived comparison context", []


def performance_signature(perf):
    """Return (signature, reason); an invalid or legacy report has no signature.

    The signature retains the entire archived context and actual case baselines.
    Reported timings/speedups are validated but never recalculated or included as
    identity fields, so measurements can improve within the same context.
    """
    signature, reason, _ = _validated_signature(perf)
    return signature, reason


def _context_mismatches(left, right, prefix):
    fields = sorted(set(left) | set(right))
    changed = [field for field in fields
               if field not in left or field not in right or left[field] != right[field]]
    return [f"{prefix}.{field}" for field in changed], [
        _CONTEXT_LABELS.get(field, str(field)) for field in changed
    ]


def compare_performance(previous, current, expected_context=None):
    """Check measurement comparability without I/O or relabeling old reports.

    ``mismatch_fields`` contains context paths, ``case_ids``, individual
    ``case_baselines.<case_id>`` paths, or invalid report fields prefixed with
    ``previous.``/``current.``. Active-context mismatches use ``expected_context``.
    """
    signatures = []
    for name, label, report in (("previous", "previous report", previous), ("current", "current report", current)):
        signature, reason, fields = _validated_signature(report)
        if signature is None:
            return {"comparable": False, "reason": f"{label} is not comparable: {reason}",
                    "mismatch_fields": [f"{name}.{field}" for field in fields]}
        signatures.append(signature)
    old, new = signatures
    if expected_context is not None:
        if not isinstance(expected_context, dict):
            return {"comparable": False, "reason": "the current comparison context must be an object",
                    "mismatch_fields": ["expected_context"]}
        fields, labels = _context_mismatches(new["comparison_context"], expected_context, "expected_context")
        if fields:
            return {"comparable": False, "reason": "the current round's archive disagrees with the current comparison context: " + "; ".join(labels),
                    "mismatch_fields": fields}
    fields, labels = _context_mismatches(old["comparison_context"], new["comparison_context"], "comparison_context")
    old_baselines, new_baselines = old["case_baselines"], new["case_baselines"]
    if set(old_baselines) != set(new_baselines):
        fields.append("case_ids")
        labels.append("case set")
    changed_baselines = [case_id for case_id in sorted(set(old_baselines) & set(new_baselines))
                         if old_baselines[case_id] != new_baselines[case_id]]
    if changed_baselines:
        fields.extend(f"case_baselines.{case_id}" for case_id in changed_baselines)
        labels.append("actual baseline timing (" + ", ".join(changed_baselines) + ")")
    if fields:
        return {"comparable": False, "reason": "the two rounds' measurement protocol changed: " + "; ".join(labels), "mismatch_fields": fields}
    return {"comparable": True, "reason": "the two rounds agree on hardware, task, cases, baselines and evaluation protocol", "mismatch_fields": []}

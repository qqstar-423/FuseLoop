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
    "framework": "目标框架",
    "backend": "运行后端",
    "runtime_versions": "运行时版本",
    "hardware": "硬件信息",
    "evaluation_device_id": "评测设备",
    "hardware_config": "硬件配置",
    "task_sha256": "任务及 case 定义",
    "metric": "计时口径",
    "baseline_protocol": "基准协议",
    "warmup": "预热次数",
    "repeat": "重复次数",
    "evaluator_sha256": "评测器代码",
    "baseline_metadata_sha256": "基准元数据",
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
        return _invalid("性能报告不是对象", "report")
    context = perf.get("comparison_context")
    if not isinstance(context, dict) or not context:
        return _invalid("缺少完整的存档比较上下文，不能推断旧报告口径", "comparison_context")
    for field in _CONTEXT_FIELDS:
        if field not in context:
            return _invalid(f"存档比较上下文缺少字段 {field}", f"comparison_context.{field}")
    if perf.get("comparison_context_stable") is not True:
        return _invalid("缺少上下文稳定证明，或测量前后上下文发生变化", "comparison_context_stable")
    if not isinstance(context["hardware"], dict) or not context["hardware"]:
        return _invalid("存档硬件信息必须是非空对象", "comparison_context.hardware")
    if not isinstance(context["hardware_config"], dict):
        return _invalid("存档硬件配置必须是对象", "comparison_context.hardware_config")
    for field in ("framework", "backend", "task_sha256", "metric", "baseline_protocol", "evaluator_sha256"):
        if not _nonempty_string(context[field]):
            return _invalid(f"存档{_CONTEXT_LABELS[field]}必须是非空字符串", f"comparison_context.{field}")
    versions = context["runtime_versions"]
    if (not isinstance(versions, dict) or not versions
            or any(not _nonempty_string(key) or not _nonempty_string(value)
                   or value.lower() == "unknown" for key, value in versions.items())):
        return _invalid("存档运行时版本必须是非空版本对象", "comparison_context.runtime_versions")
    metadata = context["baseline_metadata_sha256"]
    if metadata is not None and not _nonempty_string(metadata):
        return _invalid("存档基准元数据指纹必须是非空字符串或 None", "comparison_context.baseline_metadata_sha256")
    device = context["evaluation_device_id"]
    if not ((type(device) is int and device >= 0) or _nonempty_string(device)):
        return _invalid("存档评测设备标识无效", "comparison_context.evaluation_device_id")
    for field, minimum in (("warmup", 0), ("repeat", 1)):
        if type(context[field]) is not int or context[field] < minimum:
            return _invalid(f"存档{_CONTEXT_LABELS[field]}必须是大于等于 {minimum} 的整数", f"comparison_context.{field}")
    for field in ("_error", "score_error", "score_error_code"):
        if perf.get(field):
            return _invalid(f"性能报告包含评测错误 {field}", field)
    if not _positive_finite(perf.get("avg_speedup")):
        return _invalid("平均加速比必须是正且有限的数值", "avg_speedup")
    total = perf.get("total_cases")
    if type(total) is not int or total <= 0:
        return _invalid("total_cases 必须是严格正整数", "total_cases")
    cases = perf.get("cases")
    if not isinstance(cases, list) or len(cases) != total:
        return _invalid("case 列表不完整或与 total_cases 不一致", "cases")
    baselines = {}
    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            return _invalid(f"第 {index + 1} 个 case 不是对象", path)
        case_id = case.get("case_id")
        if not _nonempty_string(case_id):
            return _invalid(f"第 {index + 1} 个 case 缺少非空 case_id", f"{path}.case_id")
        if case_id in baselines:
            return _invalid(f"case_id 重复：{case_id}", f"{path}.case_id")
        if case.get("status") != "success":
            return _invalid(f"case {case_id} 未成功完成", f"{path}.status")
        for field in ("speedup", "elapsed_us"):
            if not _positive_finite(case.get(field)):
                return _invalid(f"case {case_id} 的 {field} 必须是正且有限的数值", f"{path}.{field}")
        baseline_field = "baseline_perf_us" if "baseline_perf_us" in case else "baseline_us"
        baseline = case.get(baseline_field)
        if not _positive_finite(baseline):
            return _invalid(f"case {case_id} 缺少正且有限的实际基准耗时", f"{path}.{baseline_field}")
        if "baseline_perf_us" in case and "baseline_us" in case:
            if not _positive_finite(case["baseline_us"]) or baseline != case["baseline_us"]:
                return _invalid(f"case {case_id} 的两种基准耗时字段不一致", f"{path}.baseline_us")
        baselines[case_id] = baseline
    return {
        "comparison_context": deepcopy(context),
        "case_baselines": {case_id: baselines[case_id] for case_id in sorted(baselines)},
    }, "性能报告完整且包含稳定的存档比较上下文", []


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
    for name, label, report in (("previous", "上一轮", previous), ("current", "当前轮", current)):
        signature, reason, fields = _validated_signature(report)
        if signature is None:
            return {"comparable": False, "reason": f"{label}报告不可比：{reason}",
                    "mismatch_fields": [f"{name}.{field}" for field in fields]}
        signatures.append(signature)
    old, new = signatures
    if expected_context is not None:
        if not isinstance(expected_context, dict):
            return {"comparable": False, "reason": "当前比较上下文必须是对象",
                    "mismatch_fields": ["expected_context"]}
        fields, labels = _context_mismatches(new["comparison_context"], expected_context, "expected_context")
        if fields:
            return {"comparable": False, "reason": "当前轮存档与当前比较上下文不一致：" + "、".join(labels),
                    "mismatch_fields": fields}
    fields, labels = _context_mismatches(old["comparison_context"], new["comparison_context"], "comparison_context")
    old_baselines, new_baselines = old["case_baselines"], new["case_baselines"]
    if set(old_baselines) != set(new_baselines):
        fields.append("case_ids")
        labels.append("case 集合")
    changed_baselines = [case_id for case_id in sorted(set(old_baselines) & set(new_baselines))
                         if old_baselines[case_id] != new_baselines[case_id]]
    if changed_baselines:
        fields.extend(f"case_baselines.{case_id}" for case_id in changed_baselines)
        labels.append("实际基准耗时（" + "、".join(changed_baselines) + "）")
    if fields:
        return {"comparable": False, "reason": "两轮测量口径变化：" + "、".join(labels), "mismatch_fields": fields}
    return {"comparable": True, "reason": "两轮硬件、任务、case、基准及评测协议一致", "mismatch_fields": []}

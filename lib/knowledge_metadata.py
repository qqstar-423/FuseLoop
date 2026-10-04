"""Program-owned environment labels and concise knowledge-write audit logs."""

import json
from pathlib import Path

from .framework_target import FRAMEWORK, BACKEND


def _read_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def build_knowledge_environment(work_dir, *, comparison_context=None, performance_report=None):
    """Label a new record without attributing the live chip to old measurements.

    New reviews target Triton Ascend; this is not a runtime version probe.
    Performance records use only the report actually used for comparison.
    Other reviews use the supplied workflow context, then device_info.json.
    """
    device_path = Path(work_dir) / "device_info.json"
    if performance_report is not None:
        context = _read_object(performance_report).get("comparison_context")
        context = context if isinstance(context, dict) else {}
        hardware = context.get("hardware")
        source_kind, source_path = "performance_report", str(Path(performance_report).resolve())
    elif isinstance(comparison_context, dict) and isinstance(comparison_context.get("hardware"), dict):
        context, hardware = comparison_context, comparison_context["hardware"]
        source_kind = "workflow_context"
        source_path = str(device_path.resolve()) if device_path.is_file() else None
    else:
        context, hardware = {}, _read_object(device_path)
        source_kind = "device_info" if hardware else "unavailable"
        source_path = str(device_path.resolve()) if device_path.is_file() else None
    hardware = hardware if isinstance(hardware, dict) else {}

    def context_label(key):
        value = context.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else "unknown"

    if performance_report is not None:
        framework, backend = context_label("framework"), context_label("backend")
        framework_source = "performance_report"
    else:
        framework, backend = FRAMEWORK, BACKEND
        framework_source = "workflow_target"

    def label(key):
        value = hardware.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else "unknown"

    device = context.get("evaluation_device_id")
    if not (type(device) is int and device >= 0 or isinstance(device, str) and device.strip()):
        device = None
    return {
        "framework": framework, "backend": backend, "framework_source": framework_source,
        "runtime_versions": context.get("runtime_versions", hardware.get("runtime_versions", {})),
        "toolchain": hardware.get("toolchain", {}),
        "chip_model": label("chip_model"), "soc_version": label("soc_version"),
        "npu_arch": label("npu_arch"), "programming_model": label("programming_model"),
        "device_id": device, "source_kind": source_kind, "source_path": source_path,
    }


def environment_summary(environment):
    """Keep historical unknowns explicit; never fill them from today's device."""
    if not isinstance(environment, dict) or not environment:
        return "framework and chip: not recorded (old experience requires checking the original evidence)"

    def value(key):
        item = environment.get(key)
        if not isinstance(item, str) or not item.strip() or item.strip().lower() == "unknown":
            return "not recorded"
        return " ".join(item.splitlines())

    return (f"framework={value('framework')}; backend={value('backend')}; chip={value('chip_model')}; "
            f"SoC={value('soc_version')}; programming model={value('programming_model')}")


def log_knowledge_write(log, state_log, *, label, path, iteration, environment,
                        decision_path, detail=""):
    """Log only after a successful write, once per distinct logger."""
    output = Path(path).resolve()
    message = ("[knowledge accumulation] %s saved; iter=%s; %s; %s; output directory=%s; file=%s; "
               "original decision=%s; environment source=%s (%s)")
    args = (label, iteration, detail, environment_summary(environment), output.parent,
            output, decision_path, environment.get("source_path"), environment.get("source_kind"))
    for logger in (log, state_log if state_log is not log else None):
        if logger is not None:
            logger.info(message, *args)

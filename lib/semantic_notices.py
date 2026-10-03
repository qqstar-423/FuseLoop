"""Durable, once-per-evaluation notices for the two semantic window triggers.

These counts are lifetime audit counts, independent of the human-consultation
counter, which resets after a consultation or performance recovery.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from lib.handoff import atomic_write_json


def _trigger_scene(iteration: int, status: dict) -> str | None:
    if (status.get("eligible") is not True
            or status.get("latest_iteration") != iteration
            or not isinstance(status.get("group_id"), str) or not status["group_id"]):
        return None
    passed, under = status.get("should_exit") is True, status.get("review_fusion") is True
    if passed == under:
        return None
    scene = "passed" if passed else "underperforming"
    window = status.get("window") or {}
    if (window.get("status") != scene or window.get("end_iteration") != iteration
            or window.get("enough_samples") is not True or window.get("stagnated") is not True):
        return None
    numeric = ("start_best_avg_speedup", "end_best_avg_speedup",
               "cumulative_improvement", "threshold")
    if any(isinstance(window.get(key), bool) or not isinstance(window.get(key), (float, int))
           or not math.isfinite(window[key]) for key in numeric):
        return None
    required = window.get("required_improvements")
    if (isinstance(required, bool) or not isinstance(required, int) or required < 1
            or window["start_best_avg_speedup"] <= 0 or window["end_best_avg_speedup"] <= 0
            or window["cumulative_improvement"] >= window["threshold"]):
        return None
    return scene


def log_semantic_trigger(work_dir, iteration, status, log, state_log, *, human_counter=None):
    """Persist and prominently log a fresh trigger; retries return the same count.

Called by Stage6 and by Stage9 recovery. Invalid/old results do not create a file.
One iteration in one comparison group can count once for each scene, even when
the same evaluation is loaded repeatedly. Both files receive the same message.
"""
    scene = _trigger_scene(iteration, status)
    if scene is None:
        return None
    path = Path(work_dir).resolve() / "selection" / "semantic_events.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(state, dict) or not isinstance(state.get("events"), list)
                or not isinstance(state.get("counts"), dict)):
            raise ValueError(f"Invalid semantic trigger event archive: {path}")
    else:
        state = {"schema_version": 1, "counts": {"passed": 0, "underperforming": 0}, "events": []}
    event_key = f"{status['group_id']}:{iteration}:{scene}"
    previous = next((item for item in state["events"] if item["event_key"] == event_key), None)
    if previous:
        return {**deepcopy(previous), "reused": True}
    count = state["counts"].get(scene, 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"Invalid semantic trigger count: {path}")
    count += 1
    window = deepcopy(status["window"])
    event = {"event_key": event_key, "scene": scene, "iteration": iteration,
             "group_id": status["group_id"], "entry_count": count, "window": window,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "state_path": str(path)}
    human_count = human_counter.get("count") if isinstance(human_counter, dict) else None
    if scene == "underperforming" and isinstance(human_count, int) and not isinstance(human_count, bool):
        event["human_consultation_count"] = human_count
    state["counts"][scene] = count
    state["events"].append(event)
    atomic_write_json(str(path), state)

    scene_number = 1 if scene == "passed" else 2
    condition = ("每个 case 的 speedup 都 >= 1" if scene == "passed" else
                 "并非每个 case 的 speedup 都 >= 1（至少一个 case < 1）")
    config_key = "passed_window" if scene == "passed" else "underperforming_window"
    window_symbol = "x" if scene == "passed" else "y"
    required = window["required_improvements"]
    action = ("达到语义退出条件，交付最佳已验证达标实现。" if scene == "passed" else
              "进入 Stage7→8→9 审查慢 case 趋势和融合方案；不强制换方案。")
    lines = [f"===== 场景{scene_number}停滞触发：第{count}次进入本场景 =====",
             f"iter{iteration}；场景{scene_number}：{condition}。",
             f"连续 {window_symbol}={required} 次有效性能迭代，最佳 avg_speedup 累计提升不足 {window['threshold']:.0%}；"
             f"配置 workflow.semantic_exit.{config_key}={required}。",
             f"窗口为同一场景的 1 个基线样本 + {required} 次有效性能迭代（共 {required + 1} 个样本）；无效轮不计数。",
             f"窗口 iter{window['start_iteration']}→iter{window['end_iteration']}；"
             f"最佳 avg_speedup {window['start_best_avg_speedup']:.6g}→{window['end_best_avg_speedup']:.6g}；"
             f"累计提升={window['cumulative_improvement']:.2%} < {window['threshold']:.2%}。",
             action,
             "本场景进入次数为终身累计；与人工咨询计数独立，断点恢复不重复计数。"]
    if "human_consultation_count" in event:
        lines.append(f"当前人工咨询触发累计={human_count}/3。")
    lines.extend([f"事件={event_key}；可追溯记录={path}", "===== 语义窗口触发记录结束 ====="])
    notice = "\n".join(lines)
    log.warning(notice)
    if state_log is not None and state_log is not log:
        state_log.warning(notice)
    return {**deepcopy(event), "reused": False}

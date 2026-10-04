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
    condition = ("every case's speedup is >= 1" if scene == "passed" else
                 "not every case's speedup is >= 1 (at least one case < 1)")
    config_key = "passed_window" if scene == "passed" else "underperforming_window"
    window_symbol = "x" if scene == "passed" else "y"
    required = window["required_improvements"]
    action = ("Semantic exit condition met; deliver the best verified passing implementation." if scene == "passed" else
              "Enter Stage7→8→9 to review slow case trends and the fusion scheme; no forced scheme replacement.")
    lines = [f"===== Scenario {scene_number} stagnation trigger: entry {count} into this scene =====",
             f"iter{iteration}; scenario {scene_number}: {condition}.",
             f"{window_symbol}={required} consecutive valid performance iterations with the best avg_speedup's cumulative improvement under {window['threshold']:.0%}; "
             f"configured workflow.semantic_exit.{config_key}={required}.",
             f"The window is 1 baseline sample plus {required} valid performance iterations within the same scene ({required + 1} samples in total); invalid rounds do not count.",
             f"window iter{window['start_iteration']}→iter{window['end_iteration']}; "
             f"best avg_speedup {window['start_best_avg_speedup']:.6g}→{window['end_best_avg_speedup']:.6g}; "
             f"cumulative improvement={window['cumulative_improvement']:.2%} < {window['threshold']:.2%}.",
             action,
             "Entries into this scene are counted over the workflow\'s lifetime; independent of the human consultation counter, and checkpoint resume does not double count."]
    if "human_consultation_count" in event:
        lines.append(f"Current human consultation trigger count={human_count}/3.")
    lines.extend([f"event={event_key}; auditable record={path}", "===== end of semantic window trigger record ====="])
    notice = "\n".join(lines)
    log.warning(notice)
    if state_log is not None and state_log is not log:
        state_log.warning(notice)
    return {**deepcopy(event), "reused": False}

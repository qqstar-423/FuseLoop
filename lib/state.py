import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .handoff import atomic_write_json


@dataclass
class State:
    schema_version: int = 1
    op_name: str = ""
    work_dir: str = ""
    current_stage: str = "N1_phase1"
    iteration: int = 0
    max_iterations: int = 20
    last_eval: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    stopped_by: Optional[str] = None
    stage9_context: Dict[str, Any] = field(default_factory=dict)
    human_review_config: Dict[str, Any] = field(default_factory=dict)

    _path: str = field(default="", repr=False)

    @classmethod
    def load_or_create(cls, work_dir: str, op_name: str = "", max_iterations: int = 20) -> "State":
        state_path = os.path.join(work_dir, ".state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version", 1) != 1 or "fusion" in data or "workflow" in data:
                raise ValueError(
                    "This checkpoint belongs to the withdrawn experimental routing workflow. "
                    "Use a new work directory; the existing checkpoint has not been modified."
                )
            st = cls(
                schema_version=data.get("schema_version", 1),
                op_name=data.get("op_name", op_name),
                work_dir=data.get("work_dir", work_dir),
                current_stage=data.get("current_stage", "N1_phase1"),
                iteration=data.get("iteration", 0),
                max_iterations=data.get("max_iterations", max_iterations),
                last_eval=data.get("last_eval", {}),
                history=data.get("history", []),
                stopped_by=data.get("stopped_by"),
                stage9_context=data.get("stage9_context", {}),
                human_review_config=data.get("human_review_config", {}),
            )
            st._path = state_path
            return st
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        st = cls(op_name=op_name, work_dir=work_dir, max_iterations=max_iterations)
        st._path = state_path
        st.flush()
        return st

    def flush(self):
        data = {
            "schema_version": self.schema_version,
            "op_name": self.op_name,
            "work_dir": self.work_dir,
            "current_stage": self.current_stage,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "last_eval": self.last_eval,
            "history": self.history,
            "stopped_by": self.stopped_by,
            "stage9_context": self.stage9_context,
            "human_review_config": self.human_review_config,
        }
        atomic_write_json(self._path, data)

    def update_last_eval(self, eval_result: Dict[str, Any]):
        self.last_eval = {
            "precision_pass": eval_result.get("precision_overall", False),
            "perf_pass": eval_result.get("perf_pass", False),
            "perf_speedup": eval_result.get("perf_speedup", 0.0),
        }
        # A resumed Stage6 re-evaluates the same round. Replace that round's
        # previous result (including duplicates in older checkpoints), rather
        # than turning it into an additional iteration.
        self.history = [
            item for item in self.history
            if item.get("iteration") != self.iteration
        ]
        self.history.append({
            "iteration": self.iteration,
            "precision_pass": eval_result.get("precision_overall", False),
            "perf_pass": eval_result.get("perf_pass", False),
            "speedup": eval_result.get("perf_speedup", 0.0),
            "timestamp": datetime.now().isoformat(),
        })
        self.flush()

    def get_previous_eval(self, iteration: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return the latest recorded evaluation strictly before a round.

        Select by iteration number, not list position: legacy checkpoints may
        contain duplicates or have been resumed out of order. For duplicates
        of the selected round, the last written entry wins. Zero/failed
        evaluations remain eligible, matching the existing comparison policy.
        """
        current_iteration = self.iteration if iteration is None else iteration
        previous = None
        for item in self.history:
            recorded_iteration = item.get("iteration")
            if type(recorded_iteration) is not int or recorded_iteration >= current_iteration:
                continue
            if previous is None or recorded_iteration >= previous["iteration"]:
                previous = item
        return dict(previous) if previous is not None else None

    def get_recent_history(self, n: int = 5) -> str:
        recent = self.history[-n:] if len(self.history) > n else self.history
        lines = []
        for h in recent:
            lines.append(
                f"- 迭代{h['iteration']}: 精度={'PASS' if h.get('precision_pass') else 'FAIL'}, "
                f"性能加速比={h.get('speedup', 'N/A')}"
            )
        return "\n".join(lines) if lines else "(无历史记录)"

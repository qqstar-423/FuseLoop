"""Evaluation comparisons stay stable when Stage6 resumes an existing round."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lib.state import State


class StateHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = State.load_or_create(self.temp.name)

    def evaluate(self, iteration, speedup):
        self.state.iteration = iteration
        self.state.update_last_eval({
            "precision_overall": True,
            "perf_pass": speedup >= 1,
            "perf_speedup": speedup,
        })

    def test_repeated_stage6_resume_keeps_previous_round_as_baseline(self):
        self.evaluate(1, 2.0)
        self.evaluate(2, 1.8)
        for _ in range(3):
            self.state = State.load_or_create(self.temp.name)
            self.evaluate(2, 1.8)
            self.assertEqual(self.state.get_previous_eval()["speedup"], 2.0)
            self.assertEqual([row["iteration"] for row in self.state.history], [1, 2])
        persisted = State.load_or_create(self.temp.name)
        self.assertEqual(persisted.last_eval["perf_speedup"], 1.8)
        self.assertEqual(len(persisted.history), 2)

    def test_upsert_removes_all_legacy_current_round_duplicates_only(self):
        older = {"iteration": 1, "speedup": 2.0, "extra": "preserve"}
        future = {"iteration": 4, "speedup": 2.4}
        self.state.history = [
            older, {"iteration": 2, "speedup": 1.7},
            future, {"iteration": 2, "speedup": 1.8},
        ]
        self.evaluate(2, 1.9)
        self.assertEqual(self.state.history[:2], [older, future])
        self.assertEqual(sum(row["iteration"] == 2 for row in self.state.history), 1)
        self.assertEqual(self.state.history[-1]["speedup"], 1.9)
        self.assertEqual(self.state.get_previous_eval()["speedup"], 2.0)

    def test_previous_round_uses_iteration_order_and_latest_duplicate(self):
        self.state.iteration = 5
        self.state.history = [
            {"iteration": 4, "speedup": 2.2},
            {"iteration": 8, "speedup": 8.0},
            {"iteration": 5, "speedup": 1.8},
            {"iteration": 4, "speedup": 2.0},
            {"iteration": 2, "speedup": 1.0},
        ]
        before = deepcopy(self.state.history)
        self.assertEqual(self.state.get_previous_eval(), {"iteration": 4, "speedup": 2.0})
        self.assertEqual(self.state.get_previous_eval(4), {"iteration": 2, "speedup": 1.0})
        self.assertEqual(self.state.history, before)

    def test_previous_zero_or_failed_evaluation_is_not_skipped(self):
        self.state.iteration = 3
        failed = {"iteration": 2, "speedup": 0.0, "precision_pass": False, "perf_pass": False}
        self.state.history = [{"iteration": 1, "speedup": 2.0}, failed]
        self.assertEqual(self.state.get_previous_eval(), failed)

    def test_no_previous_round_returns_none_with_current_or_future_entries(self):
        self.state.iteration = 1
        self.assertIsNone(self.state.get_previous_eval())
        self.state.history = [{"iteration": 1, "speedup": 1.1}, {"iteration": 2, "speedup": 1.2}]
        self.assertIsNone(self.state.get_previous_eval())

    def test_returned_previous_evaluation_cannot_change_stored_metric(self):
        self.state.iteration = 2
        self.state.history = [{"iteration": 1, "speedup": 2.0}]
        previous = self.state.get_previous_eval()
        previous["speedup"] = 99.0
        self.assertEqual(self.state.history[0]["speedup"], 2.0)

    def test_legacy_consultation_checkpoint_resumes_autonomous_review(self):
        context = {
            "iteration": 4, "scene": "stagnation", "fail_reason": "perf_optimize",
            "perf_diff": {"has_regression": True, "avg_speedup_before": 1.2},
            "semantic_event": {"event_key": "group:4:underperforming", "entry_count": 3},
        }
        path = Path(self.state._path)
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.update(iteration=4, current_stage="iter4_stage9",
                      human_review_config={"consultation_enabled": True},
                      stage9_context=dict(context, phase="consulting", message_ids=["old-message"],
                                          bundle_path="human_review/old/bundle.json", consultation_id="old-request"))
        path.write_text(json.dumps(legacy), encoding="utf-8")
        restored = State.load_or_create(self.temp.name)
        self.assertEqual(restored.iteration, 4)
        self.assertEqual(restored.current_stage, "iter4_stage9")
        self.assertEqual(restored.stage9_context, dict(context, phase="reviewing"))
        self.assertFalse(hasattr(restored, "human_review_config"))
        restored.flush()
        persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("human_review_config", persisted)
        self.assertEqual(persisted["stage9_context"], dict(context, phase="reviewing"))
        self.assertFalse((Path(self.temp.name) / "human_review").exists())

    def test_legacy_interaction_fields_do_not_discard_pending_development(self):
        context = {
            "iteration": 3, "scene": "all_passed", "fail_reason": "perf_pass_optimize",
            "perf_diff": {"has_improvement": True}, "phase": "developing",
            "development_reason": "perf_pass_optimize",
            "decision_path": "knowledge/stage9/iter3/accepted/decision.json",
        }
        path = Path(self.state._path)
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.update(iteration=3, current_stage="iter3_stage3",
                      human_review_config={"proactive_enabled": True},
                      stage9_context=dict(context, message_ids=["old-message"],
                                          bundle_path="old-bundle", consultation_id="old-request"))
        path.write_text(json.dumps(legacy), encoding="utf-8")
        restored = State.load_or_create(self.temp.name)
        self.assertEqual(restored.current_stage, "iter3_stage3")
        self.assertEqual(restored.stage9_context, context)
        restored.flush()
        self.assertEqual(State.load_or_create(self.temp.name).stage9_context, context)


if __name__ == "__main__":
    unittest.main()

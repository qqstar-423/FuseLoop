"""Audit counts are durable, scene-specific, and independent of human resets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from lib.semantic_notices import log_semantic_trigger


class SemanticNoticesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.log, self.state_log = Mock(), Mock()

    def status(self, iteration=4, *, passed=False, group="hardware-task"):
        return {"eligible": True, "latest_iteration": iteration, "group_id": group,
                "should_exit": passed, "review_fusion": not passed,
                "window": {"status": "passed" if passed else "underperforming",
                           "required_improvements": 2 if passed else 3,
                           "valid_samples": 4, "completed_improvements": 2 if passed else 3,
                           "enough_samples": True, "stagnated": True,
                           "start_iteration": 1, "end_iteration": iteration,
                           "start_best_avg_speedup": 2.0, "end_best_avg_speedup": 2.04,
                           "cumulative_improvement": 0.02, "threshold": 0.05}}

    def emit(self, iteration=4, status=None, **kwargs):
        return log_semantic_trigger(self.work, iteration, status or self.status(iteration),
                                    self.log, self.state_log, **kwargs)

    def test_both_scenes_log_actual_config_and_window_with_visible_banner(self):
        for passed in (False, True):
            event = self.emit(status=self.status(passed=passed))
            self.assertEqual(event["entry_count"], 1)
            self.assertFalse(event["reused"])
            notice = self.log.warning.call_args.args[0]
            self.assertIn("===== Scenario", notice)
            self.assertIn("entry 1 into this scene", notice)
            config_key, symbol, required = ("passed_window", "x", 2) if passed else ("underperforming_window", "y", 3)
            self.assertIn(f"workflow.semantic_exit.{config_key}={required}", notice)
            self.assertIn(f"{symbol}={required} consecutive valid performance iterations", notice)
            self.assertIn(f"1 baseline sample plus {required} valid performance iterations within the same scene ({required + 1} samples in total)", notice)
            self.assertIn("invalid rounds do not count", notice)
            self.assertIn("independent of the human consultation counter, and checkpoint resume does not double count", notice)
            self.assertIn("iter1→iter4", notice)
            self.assertIn("avg_speedup 2→2.04", notice)
            self.assertIn("2.00% < 5.00%", notice)
            self.assertIn("scenario 1: every case's speedup is >= 1" if passed else
                          "scenario 2: not every case's speedup is >= 1 (at least one case < 1)", notice)
            self.state_log.warning.assert_called_with(notice)
            self.assertTrue(Path(event["state_path"]).is_file())

    def test_duplicate_same_group_iteration_survives_restart_without_relogging(self):
        first = self.emit()
        second = log_semantic_trigger(str(self.work), 4, deepcopy(self.status()),
                                      self.log, self.state_log)
        self.assertTrue(second["reused"])
        self.assertEqual(second["entry_count"], first["entry_count"])
        self.log.warning.assert_called_once()
        self.state_log.warning.assert_called_once()
        data = json.loads(Path(first["state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(data["events"]), 1)

    def test_lifetime_counts_are_per_scene_across_groups_and_human_reset(self):
        first = self.emit(human_counter={"count": 3})
        self.assertIn("Current human consultation trigger count=3/3", self.log.warning.call_args.args[0])
        second = self.emit(5, human_counter={"count": 1})
        self.assertIn("entry 2 into this scene", self.log.warning.call_args.args[0])
        self.assertIn("Current human consultation trigger count=1/3", self.log.warning.call_args.args[0])
        third = self.emit(5, status=self.status(5, group="new-hardware"))
        passed = self.emit(6, status=self.status(6, passed=True))
        self.assertEqual([first["entry_count"], second["entry_count"], third["entry_count"]], [1, 2, 3])
        self.assertEqual(second["human_consultation_count"], 1)
        self.assertEqual(passed["entry_count"], 1)

    def test_invalid_stale_nontrigger_or_inconsistent_window_does_not_count(self):
        variants = [dict(eligible=False), dict(latest_iteration=3), dict(group_id=""),
                    dict(should_exit=False, review_fusion=False), dict(should_exit=True, review_fusion=True)]
        for change in variants:
            status = self.status()
            status.update(change)
            self.assertIsNone(self.emit(status=status))
        for change in ({"end_iteration": 3}, {"status": "passed"}, {"enough_samples": False},
                       {"stagnated": False}, {"cumulative_improvement": .05},
                       {"cumulative_improvement": float("nan")}, {"required_improvements": 0}):
            status = self.status()
            status["window"].update(change)
            self.assertIsNone(self.emit(status=status))
        self.log.warning.assert_not_called()
        self.assertFalse((self.work / "selection").exists())

    def test_a_nontrigger_gap_does_not_reset_lifetime_count(self):
        self.emit()
        normal = self.status(5)
        normal.update(review_fusion=False)
        self.assertIsNone(self.emit(5, status=normal))
        self.assertEqual(self.emit(6)["entry_count"], 2)

    def test_same_logger_is_not_written_twice_and_optional_state_logger_supported(self):
        log_semantic_trigger(self.work, 4, self.status(), self.log, self.log)
        self.log.warning.assert_called_once()
        log_semantic_trigger(self.work, 5, self.status(5), self.log, None)
        self.assertEqual(self.log.warning.call_count, 2)

    def test_corrupt_existing_archive_is_not_silently_reset(self):
        self.emit()
        path = self.work / "selection" / "semantic_events.json"
        path.write_text('{"events": [], "counts": {"underperforming": -1}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.emit(5)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["counts"]["underperforming"], -1)


if __name__ == "__main__":
    unittest.main()

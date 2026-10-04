"""Real archive validation and interruption recovery for selective code restore."""
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

from lib.fusion_evidence import load_evidence
from lib.regression_restore import (prepare_regression_action, apply_regression_action,
                                    load_regression_action, regression_action_prompt)
import test_semantic_exit as semantic_tests


class RegressionRestoreTests(unittest.TestCase):
    def setUp(self):
        self.fx = semantic_tests.SemanticExitTests(methodName="runTest")
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.work = self.fx.work
        self.code = self.work / "impl/nested/kernel.py"
        self.original = self.code.read_text(encoding="utf-8")
        self.fx.record(1, 2.0)
        self.code.write_text("# regressed\n", encoding="utf-8")
        self.status = self.fx.record(2, 1.5)
        self.log = logging.getLogger("regression-offline")
        self.diff = {"has_regression": True, "prev_iter": 1, "curr_iter": 2,
                     "prev_avg_speedup": 2.0, "curr_avg_speedup": 1.5, "delta_pct": -25.0}
        self.decision = self.work / "knowledge/stage9/iter2/test/decision.json"
        self.fx.write(self.decision, {"iteration": 2, "request_id": "test", "regression_pattern": {"what_changed": "tile"}})
        self.fx.write(self.decision.parent / "commit.json", {"committed": True, "iteration": 2, "request_id": "test"})
        (self.work / "knowledge/regression_patterns.md").write_text("iter2: more padding", encoding="utf-8")

    def prepare(self, status=None):
        return prepare_regression_action(self.work, 2, self.diff, status or self.status, self.log, None)

    def apply(self, **kwargs):
        return apply_regression_action(self.work, 2, self.decision, self.log, None, **kwargs)

    def test_restore_invalidates_old_binding_and_keeps_failed_code_and_measured_results(self):
        self.fx.write(self.work / "selection/current_implementation.json", {"eligible": True, "original": "binding"})
        before = (self.work / "selection/state.json").read_bytes()
        self.prepare()
        record = self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), self.original)
        self.assertEqual((Path(record["failed_implementation_dir"]) / "nested/kernel.py").read_text(encoding="utf-8"), "# regressed\n")
        self.assertEqual((self.work / "selection/state.json").read_bytes(), before)
        self.assertFalse(load_evidence(self.work)["eligible"])
        saved = json.loads((Path(record["record_path"]).parent / "failed_binding.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["original"], "binding")
        prompt = regression_action_prompt(self.work, 2)
        self.assertIn("The working impl has been restored to the best baseline", prompt)
        self.assertIn("belongs to the pre-restore code", prompt)
        self.assertIn("iteration template", prompt)

    def test_resume_after_either_directory_rename_finishes_once(self):
        # Each subtest uses fresh files; the failure occurs after the actual rename.
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                other = RegressionRestoreTests(methodName="runTest")
                other.setUp()
                try:
                    other.prepare()
                    original_rename = Path.rename
                    moves = []
                    def rename(path, target):
                        result = original_rename(path, target)
                        if path.name == "impl" or path.name.startswith("prepared-"):
                            moves.append(str(path))
                            if len(moves) == fail_at:
                                raise RuntimeError("synthetic rename interruption")
                        return result
                    with patch.object(Path, "rename", rename), self.assertRaisesRegex(RuntimeError, "interruption"):
                        other.apply()
                    self.assertEqual(load_regression_action(other.work, 2)["status"], "prepared")
                    result = other.apply()
                    self.assertEqual(result["status"], "restored")
                    self.assertEqual(other.code.read_text(encoding="utf-8"), other.original)
                    other.code.write_text("# partial development\n", encoding="utf-8")
                    other.apply()
                    self.assertEqual(other.code.read_text(encoding="utf-8"), "# partial development\n")
                finally:
                    other.doCleanups()

    def test_corrupt_best_after_plan_never_replaces_live_source(self):
        self.prepare()
        archived = Path(self.status["best"]["implementation_dir"]) / "nested/kernel.py"
        archived.write_text("# corrupted archive", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot is invalid"):
            self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), "# regressed\n")

    def test_uncommitted_stage9_and_changed_context_refuse_restore(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.apply(comparison_context={"different": "hardware"})
        self.fx.write(self.decision.parent / "commit.json", {"committed": False, "iteration": 2})
        with self.assertRaisesRegex(ValueError, "Stage9 must first commit"):
            self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), "# regressed\n")

    def test_new_source_change_between_review_and_restore_is_not_overwritten(self):
        self.prepare()
        self.code.write_text("# unsaved development", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite development changes"):
            self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), "# unsaved development")

    def test_invalid_current_evaluation_cannot_restore_historical_passed_best(self):
        self.status.update(eligible=False, reason="invalid self-test")
        record = self.prepare()
        self.assertEqual(record["action"], "keep_current")
        self.assertIsNone(record["all_cases_pass"])
        self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), "# regressed\n")

    def test_path_escape_is_rejected_before_any_directory_operation(self):
        record = self.prepare()
        record["failed_implementation_dir"] = str(self.work.parent / "outside-impl")
        self.fx.write(Path(record["record_path"]), record)
        with self.assertRaisesRegex(ValueError, "inside the working directory"):
            self.apply()
        self.assertEqual(self.code.read_text(encoding="utf-8"), "# regressed\n")


if __name__ == "__main__":
    unittest.main()

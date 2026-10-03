"""Offline tests for measured best snapshots and semantic-exit decisions."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import shutil

from lib.fusion_evidence import implementation_hash
from lib.regression_restore import apply_regression_action, prepare_regression_action
from lib.semantic_exit import format_selection_for_prompt, load_selection_status, record_evaluation


class SemanticExitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        (self.work / "impl" / "nested").mkdir(parents=True)
        (self.work / "impl" / "nested" / "kernel.py").write_text("# measured implementation\n", encoding="utf-8")
        self.context = {"hardware": {"chip": "test-chip"}, "task_sha256": "same-task",
                        "metric": "kernel_details", "baseline_protocol": "native-v1"}
        self.config = {"workflow": {"semantic_exit": {"passed_window": 3, "underperforming_window": 3}}}
        self.evidence_path = self.work / "self_test.md"
        self.evidence_path.write_text("Executed all cases and changed-data continuous calls.\n", encoding="utf-8")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def inputs(self, iteration, average, speeds=None, *, baseline=100):
        speeds = speeds or [average, average]
        cases = [{"case_id": f"case_{index}", "status": "success", "speedup": speed,
                  "elapsed_us": baseline / speed, "baseline_perf_us": baseline,
                  "perf_score": 0.4 + index / 100, "t_hw_us": 3.25 + index,
                  "op_times": [{"name": "kernel", "time_us": baseline / speed}]}
                 for index, speed in enumerate(speeds)]
        source = self.work / "eval" / f"iter{iteration}" / "source.json"
        operator = {"total_cases": len(cases), "passed_cases": len(cases), "failed_cases": 0,
                    "avg_speedup": average, "performance_score": 12.75, "cases": cases}
        self.write(source, {"operators": [operator]})
        perf = {**deepcopy(operator), "source_json": str(source), "perf_pass": all(s >= 1 for s in speeds)}
        precision = {"precision_overall": True, "total_cases": len(cases),
                     "passed_cases": len(cases), "failed_cases": 0}
        evidence = {"eligible": True, "implementation_plan": "Use tiled matmul and sigmoid epilogue.",
                    "fusion_scheme": {"id": "F2", "kernel_count": 2, "shape_dispatch": True},
                    "impl_sha256": implementation_hash(str(self.work / "impl")),
                    "evidence_paths": {"self_test_report": str(self.evidence_path)}}
        return {"perf": perf, "precision": precision, "evidence": evidence,
                "comparison_context": self.context, "config": self.config}

    def record(self, iteration, average, speeds=None, **overrides):
        inputs = self.inputs(iteration, average, speeds)
        inputs.update(overrides)
        return record_evaluation(self.work, iteration, **inputs)

    def test_before_performance_no_selection_directory(self):
        status = load_selection_status(self.work)
        self.assertIsNone(status["best"])
        self.assertFalse(status["should_exit"])
        self.assertFalse((self.work / "selection").exists())

    def test_passed_has_priority_over_faster_underperforming(self):
        first = self.record(0, 3.0, [5.2, 0.8])
        self.assertEqual(first["best"]["selection_label"], "best_available")
        second = self.record(1, 2.0)
        self.assertEqual(second["best"]["iteration"], 1)
        third = self.record(2, 4.0, [7.2, 0.8])
        self.assertEqual(third["best"]["iteration"], 1)
        self.assertTrue(third["best"]["all_cases_pass"])

    def test_best_is_largest_average_within_same_status(self):
        self.record(0, 2.0)
        self.record(1, 2.3)
        last = self.record(2, 2.1)
        self.assertEqual(last["best"]["iteration"], 1)
        self.assertEqual(last["best"]["avg_speedup"], 2.3)

    def test_passed_requires_baseline_and_three_improvements(self):
        for index, average in enumerate([2.0, 2.01, 2.02]):
            result = self.record(index, average)
            self.assertFalse(result["should_exit"])
        result = self.record(3, 2.04)
        self.assertTrue(result["should_exit"])
        self.assertFalse(result["review_fusion"])
        self.assertEqual(result["window"]["valid_samples"], 4)
        self.assertAlmostEqual(result["window"]["cumulative_improvement"], 0.02)

    def test_exact_five_percent_does_not_exit(self):
        for index, average in enumerate([2.0, 2.01, 2.03, 2.1]):
            result = self.record(index, average)
        self.assertFalse(result["should_exit"])
        self.assertEqual(result["window"]["cumulative_improvement"], 0.05)

    def test_underperforming_stagnation_requests_review_only(self):
        for index, average in enumerate([2.0, 2.01, 2.02, 2.04]):
            result = self.record(index, average, [average * 2 - .8, .8])
        self.assertFalse(result["should_exit"])
        self.assertTrue(result["review_fusion"])
        self.assertFalse(result["best"]["all_cases_pass"])

    def test_regression_then_recovery_is_not_new_gain(self):
        for index, average in enumerate([2.0, 1.5, 1.7, 2.0]):
            result = self.record(index, average)
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["window"]["cumulative_improvement"], 0)

    def test_window_uses_historical_best_even_before_its_start(self):
        for index, average in enumerate([2.0, 2.2, 1.5, 1.7, 2.0, 2.2]):
            result = self.record(index, average)
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["window"]["start_iteration"], 2)
        self.assertEqual(result["window"]["start_best_avg_speedup"], 2.2)

    def test_status_transition_resets_window(self):
        self.record(0, 2.0)
        self.record(1, 2.01)
        self.record(2, 2.02)
        under = self.record(3, 2.03, [3.26, .8])
        self.assertEqual(under["window"]["valid_samples"], 1)
        passed = self.record(4, 2.03)
        self.assertEqual(passed["window"]["valid_samples"], 1)
        self.assertFalse(passed["should_exit"])

    def test_invalid_precision_does_not_advance_window(self):
        self.record(0, 2.0)
        inputs = self.inputs(1, 9.0)
        inputs["precision"]["precision_overall"] = False
        invalid = record_evaluation(self.work, 1, **inputs)
        self.assertFalse(invalid["eligible"])
        self.assertEqual(invalid["best"]["iteration"], 0)
        self.assertFalse(invalid["should_exit"])
        self.assertFalse(load_selection_status(self.work)["should_exit"])
        self.record(2, 2.01)
        self.record(3, 2.02)
        valid = self.record(4, 2.03)
        self.assertTrue(valid["should_exit"])
        self.assertEqual(valid["window"]["valid_samples"], 4)
        self.assertEqual([point["iteration"] for point in valid["case_trends"][0]["history"]], [0, 2, 3, 4])

    def test_bad_or_missing_evidence_never_creates_best(self):
        for fault in ("failed_selftest", "missing_file", "changed_code"):
            with self.subTest(fault=fault):
                inputs = self.inputs(0, 2.0)
                if fault == "failed_selftest":
                    inputs["evidence"]["eligible"] = False
                    inputs["evidence"]["reason"] = "Continuous calls failed"
                elif fault == "missing_file":
                    inputs["evidence"]["evidence_paths"]["self_test_report"] = str(self.work / "missing.md")
                else:
                    inputs["evidence"]["impl_sha256"] = "old hash"
                status = record_evaluation(self.work, 0, **inputs)
                self.assertFalse(status["eligible"])
                self.assertFalse((self.work / "selection" / "best.json").exists())

    def test_missing_or_incomplete_report_rejected(self):
        for fault in ("missing", "incomplete", "failed_case", "bad_average", "mismatch"):
            with self.subTest(fault=fault):
                inputs = self.inputs(0, 2.0)
                if fault == "missing":
                    Path(inputs["perf"]["source_json"]).unlink()
                elif fault == "incomplete":
                    inputs["perf"]["cases"].pop()
                elif fault == "failed_case":
                    inputs["perf"]["cases"][0]["status"] = "failed"
                elif fault == "bad_average":
                    inputs["perf"]["avg_speedup"] = float("nan")
                else:
                    inputs["perf"]["cases"][0]["speedup"] = 9
                self.assertFalse(record_evaluation(self.work, 0, **inputs)["eligible"])

    def test_snapshot_has_independent_code_reports_and_evidence(self):
        result = self.record(0, 2.0)
        best = result["best"]
        archived_code = Path(best["implementation_dir"]) / "nested" / "kernel.py"
        archived_report = Path(best["performance_report"])
        archived_evidence = Path(best["evidence_paths"]["self_test_report"])
        expected_report = archived_report.read_bytes()
        (self.work / "impl" / "nested" / "kernel.py").write_text("# changed later\n", encoding="utf-8")
        Path(best["original_performance_report"]).write_text("{}", encoding="utf-8")
        self.evidence_path.write_text("new evidence", encoding="utf-8")
        self.assertEqual(archived_code.read_text(encoding="utf-8"), "# measured implementation\n")
        self.assertEqual(archived_report.read_bytes(), expected_report)
        self.assertIn("continuous", archived_evidence.read_text(encoding="utf-8"))
        self.assertTrue(load_selection_status(self.work)["eligible"])
        self.assertEqual(best["hap"]["cases"][0]["perf_score"], .4)
        self.assertEqual(best["hap"]["cases"][0]["t_hw_us"], 3.25)
        self.assertEqual(best["hap"]["performance_score"], 12.75)
        self.assertNotIn("avg_hap", best["hap"])
        self.assertEqual(best["fusion_scheme"]["kernel_count"], 2)

    def test_resume_is_idempotent(self):
        inputs = self.inputs(0, 2.0)
        first = record_evaluation(self.work, 0, **inputs)
        second = record_evaluation(self.work, 0, **inputs)
        self.assertEqual(first["best"]["record_id"], second["best"]["record_id"])
        self.assertEqual(second["window"]["valid_samples"], 1)
        self.assertEqual(len(list((self.work / "selection" / "records").glob("iter*"))), 1)

    def test_repeated_iteration_counts_once_and_preserves_old_archive(self):
        first = self.record(0, 2.0)
        second = self.record(0, 1.9)
        self.assertEqual(second["window"]["valid_samples"], 1)
        self.assertEqual(second["best"]["avg_speedup"], 2.0)
        self.assertTrue(Path(first["best"]["manifest_path"]).is_file())
        self.assertEqual(len(list((self.work / "selection" / "records").glob("iter*"))), 2)

    def test_repeated_iteration_old_best_still_anchors_the_window(self):
        self.config["workflow"]["semantic_exit"]["passed_window"] = 1
        self.record(0, 2.0)
        self.record(0, 1.5)
        result = self.record(1, 1.6)
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["window"]["cumulative_improvement"], 0)

    def test_different_contexts_and_case_sets_are_isolated(self):
        self.record(0, 9.0)
        for index, field in enumerate(("hardware", "task_sha256", "metric", "baseline_protocol"), 1):
            context = {**self.context, field: f"different-{field}"}
            result = self.record(index, 2.0, comparison_context=context)
            self.assertEqual(result["best"]["iteration"], index)
            self.assertEqual(result["window"]["valid_samples"], 1)
        result = self.record(5, 2.0, [2.0, 2.0, 2.0])
        self.assertEqual(result["window"]["valid_samples"], 1)
        self.assertEqual(result["best"]["iteration"], 5)
        mismatch = load_selection_status(self.work, {**self.context, "task_sha256": "unseen-task"})
        self.assertIsNone(mismatch["best"])

    def test_measured_performance_fluctuation_keeps_the_same_baseline_group(self):
        first = self.record(0, 2.0)
        second = self.record(1, 2.01)
        self.assertEqual(first["group_id"], second["group_id"])
        self.assertEqual(second["window"]["valid_samples"], 2)
        self.assertEqual(first["current"]["case_results"][0]["baseline_perf_us"], 100)
        self.assertEqual(second["current"]["case_results"][0]["baseline_perf_us"], 100)
        self.assertNotEqual(first["current"]["case_results"][0]["elapsed_us"],
                            second["current"]["case_results"][0]["elapsed_us"])

    def test_changed_actual_baselines_isolate_best_and_window(self):
        old_best = self.record(0, 5.0)
        new_baseline = record_evaluation(self.work, 1, **self.inputs(1, 2.0, baseline=20))
        regression = record_evaluation(self.work, 2, **self.inputs(2, 1.8, baseline=20))
        self.assertNotEqual(old_best["group_id"], new_baseline["group_id"])
        self.assertEqual(new_baseline["window"]["valid_samples"], 1)
        self.assertEqual(regression["group_id"], new_baseline["group_id"])
        self.assertEqual(regression["best"]["iteration"], 1)
        self.assertEqual(regression["window"]["valid_samples"], 2)
        self.assertFalse(regression["should_exit"])
        self.assertEqual([point["iteration"] for point in regression["case_trends"][0]["history"]], [1, 2])
        loaded = load_selection_status(self.work, self.context)
        self.assertEqual(loaded["best"]["iteration"], 1)
        index = json.loads((self.work / "selection" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(index["groups"][old_best["group_id"]]["case_baselines"],
                         {"case_0": 100, "case_1": 100})
        self.assertEqual(index["groups"][new_baseline["group_id"]]["case_baselines"],
                         {"case_0": 20, "case_1": 20})

    def test_case_order_and_equivalent_numeric_baselines_keep_the_same_group(self):
        first = self.record(0, 2.0)
        inputs = self.inputs(1, 2.01, baseline=100.0)
        inputs["perf"]["cases"].reverse()
        for case in inputs["perf"]["cases"]:
            case["baseline_us"] = case.pop("baseline_perf_us")
        second = record_evaluation(self.work, 1, **inputs)
        self.assertTrue(second["eligible"])
        self.assertEqual(first["group_id"], second["group_id"])
        self.assertEqual(second["window"]["valid_samples"], 2)

    def test_legacy_group_without_archived_baselines_is_not_loaded(self):
        result = self.record(0, 2.0)
        state_path = self.work / "selection" / "state.json"
        index = json.loads(state_path.read_text(encoding="utf-8"))
        index["groups"][result["group_id"]].pop("case_baselines")
        self.write(state_path, index)
        original = state_path.read_bytes()
        for context in (None, self.context):
            with self.subTest(context=context):
                loaded = load_selection_status(self.work, context)
                self.assertFalse(loaded["eligible"])
                self.assertIsNone(loaded["best"])
                self.assertFalse(loaded["should_exit"])
                self.assertIn("no archived case baselines", loaded["reason"])
        self.assertEqual(state_path.read_bytes(), original)

    def test_pending_restore_from_legacy_group_is_rejected(self):
        self.record(0, 2.5)
        selection = self.record(1, 2.0)
        logger = Mock()
        comparison = {"has_regression": True, "prev_iter": 0,
                      "prev_avg_speedup": 2.5, "curr_avg_speedup": 2.0, "delta_pct": -20}
        action = prepare_regression_action(self.work, 1, comparison, selection, logger, logger)
        self.assertEqual(action["status"], "planned")
        state_path = self.work / "selection" / "state.json"
        index = json.loads(state_path.read_text(encoding="utf-8"))
        index["groups"][selection["group_id"]].pop("case_baselines")
        self.write(state_path, index)
        decision_path = self.work / "knowledge" / "stage9" / "decision.json"
        self.write(decision_path, {"iteration": 1, "request_id": "legacy-review",
                                   "regression_pattern": {"what_changed": "old measured change"}})
        self.write(decision_path.parent / "commit.json", {
            "committed": True, "iteration": 1, "request_id": "legacy-review",
        })
        Path(action["knowledge_path"]).write_text("Archived regression lesson", encoding="utf-8")
        before = implementation_hash(self.work / "impl")
        with self.assertRaisesRegex(ValueError, "最佳或本轮评测快照失效"):
            apply_regression_action(self.work, 1, decision_path, logger, logger,
                                    comparison_context=self.context)
        self.assertEqual(implementation_hash(self.work / "impl"), before)
        self.assertFalse(Path(action["failed_implementation_dir"]).exists())
        pending = json.loads(Path(action["record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(pending["status"], "planned")

    def test_every_case_has_trend_including_slow_case_improvement(self):
        self.record(0, 2.684, [3.0] * 8 + [.683])
        result = self.record(1, 2.724, [3.0] * 8 + [.977])
        self.assertEqual(len(result["case_trends"]), 9)
        slow = next(c for c in result["case_trends"] if c["case_id"] == "case_8")
        self.assertAlmostEqual(slow["change_from_previous"], .294)
        self.assertAlmostEqual(slow["gap_to_one"], .023)
        self.assertEqual([p["speedup"] for p in slow["history"]], [.683, .977])
        prompt = format_selection_for_prompt(self.work, self.context)
        self.assertIn("case_8", prompt)
        self.assertIn("0.683", prompt)
        self.assertIn("never mandatory fusion replacement", prompt)

    def test_custom_windows_survive_read_only_resume(self):
        self.config["workflow"]["semantic_exit"] = {"passed_window": 1, "underperforming_window": 2}
        self.record(0, 2.0)
        result = self.record(1, 2.01)
        self.assertTrue(result["should_exit"])
        loaded = load_selection_status(self.work, self.context)
        self.assertTrue(loaded["should_exit"])
        self.assertEqual(loaded["latest_iteration"], 1)

    def test_tampered_snapshot_cannot_cause_exit(self):
        result = self.record(0, 2.0)
        Path(result["best"]["performance_report"]).write_text("{}", encoding="utf-8")
        loaded = load_selection_status(self.work)
        self.assertFalse(loaded["eligible"])
        self.assertFalse(loaded["should_exit"])
        self.assertIn("changed", loaded["reason"])

    def test_invalid_window_configuration_rejected(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.record(0, 2.0, config={"passed_window": value})

    def test_truthy_precision_string_is_not_a_pass(self):
        inputs = self.inputs(0, 2.0)
        inputs["precision"]["precision_overall"] = "false"
        self.assertFalse(record_evaluation(self.work, 0, **inputs)["eligible"])

    def test_invalid_or_mismatched_baseline_is_rejected(self):
        for value in (0, -1, float("inf"), 99):
            with self.subTest(value=value):
                inputs = self.inputs(0, 2.0)
                inputs["perf"]["cases"][0]["baseline_perf_us"] = value
                self.assertFalse(record_evaluation(self.work, 0, **inputs)["eligible"])

    def test_missing_optional_report_links_are_not_forwarded(self):
        inputs = self.inputs(0, 2.0)
        inputs["perf"]["source_md"] = str(self.work / "missing.md")
        inputs["perf"]["source_csv_dir"] = str(self.work / "mutable_csvs")
        inputs["perf"]["cases"][0]["kernel_csv"] = str(self.work / "missing.csv")
        result = record_evaluation(self.work, 0, **inputs)
        archived = json.loads(Path(result["best"]["perf_result_path"]).read_text(encoding="utf-8"))
        self.assertEqual(archived["source_md"], "")
        self.assertEqual(archived["source_csv_dir"], "")
        self.assertEqual(archived["cases"][0]["kernel_csv"], "")

    def test_report_changed_during_copy_is_not_published(self):
        inputs = self.inputs(0, 2.0)
        source = Path(inputs["perf"]["source_json"])
        original_copy = shutil.copy2

        def mutate_report(src, dst, *args, **kwargs):
            if Path(src) == source:
                source.write_text("{}", encoding="utf-8")
            return original_copy(src, dst, *args, **kwargs)

        with patch("lib.semantic_exit.shutil.copy2", side_effect=mutate_report):
            result = record_evaluation(self.work, 0, **inputs)
        self.assertFalse(result["eligible"])
        self.assertIn("changed", result["reason"])
        self.assertFalse((self.work / "selection" / "best.json").exists())
        self.assertEqual(list((self.work / "selection" / "records").iterdir()), [])

    def test_corrupt_index_is_preserved_and_never_exits(self):
        self.record(0, 2.0)
        state = self.work / "selection" / "state.json"
        state.write_text("{broken", encoding="utf-8")
        result = self.record(1, 2.01)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["should_exit"])
        self.assertEqual(state.read_text(encoding="utf-8"), "{broken")


if __name__ == "__main__":
    unittest.main()

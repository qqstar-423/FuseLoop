"""Offline routing must not turn measurement-context changes into optimization wins."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
from lib.history_manager import load_history
from lib.regression_restore import load_regression_action
import test_fusion_routing as fusion_routing_tests
import test_semantic_routing as semantic_routing_tests


class PerformanceComparisonRoutingTests(unittest.TestCase):
    def setUp(self):
        # Compose the transport fixture without inheriting its unrelated tests.
        # Report parsing, routing, checkpoint recovery and history remain real.
        self.routing = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.work = self.routing.work
        self.routing.config["workflow"]["semantic_exit"].update(
            passed_window=8, underperforming_window=8)

    def read_json(self, relative):
        return json.loads((self.work / relative).read_text(encoding="utf-8"))

    def request(self, iteration):
        prompt = next(prompt for current, stage, prompt in reversed(self.routing.seen)
                      if current == iteration and stage == "stage9")
        output, _ = fusion_routing_tests.stage9_request(prompt)
        return json.loads((output.parent / "request.json").read_text(encoding="utf-8"))

    def change_after_first_review(self, change):
        def callback(output, payload, iteration, _prompt):
            if iteration == 1:
                change()
            self.routing.write_json(output, payload)
            return True
        self.routing.stage9_callback = callback

    def assert_no_patterns_or_rollback(self, iteration):
        for filename, marker in (("proven_patterns.md", "LOCAL_TILE_SUCCESS"),
                                 ("regression_patterns.md", "LOCAL_TILE_REGRESSION")):
            path = self.work / "knowledge" / filename
            self.assertNotIn(marker, path.read_text(encoding="utf-8") if path.exists() else "")
        self.assertIsNone(load_regression_action(self.work, iteration))

    def assert_new_baseline(self, iteration, average):
        comparison = self.request(iteration)["perf_diff"]
        self.assertFalse(comparison["comparable"])
        self.assertFalse(comparison["has_improvement"])
        self.assertFalse(comparison["has_regression"])
        self.assertEqual(comparison["comparison_status"], "new_baseline")
        self.assertTrue(comparison["comparison_reason"])
        self.assertFalse(comparison.get("diff_text"))
        self.assertFalse(comparison.get("regression_text"))
        history = load_history(str(self.work))
        row = next(row for row in history["rounds"] if row["iter"] == iteration)
        entry = next(row for row in history["ledger"] if row["iter"] == iteration)
        self.assertIsNone(row["improved"])
        self.assertIsNone(entry["avg_speedup_before"])
        self.assertIsNone(entry["delta"])
        self.assertEqual(entry["avg_speedup_after"], average)
        self.assertEqual(entry["verdict"], "baseline")
        for record in (row, entry):
            self.assertEqual(record["comparison_status"], "new_baseline")
            self.assertTrue(record["comparison_reason"])

    def test_device_change_does_not_record_apparent_improvement(self):
        self.routing.perfs = {1: (1.5,), 2: (3.0,)}
        device_path = self.work / "device_info.json"
        original_device = device_path.read_bytes()
        self.change_after_first_review(
            lambda: self.routing.config.update(hardware={"device_id": 1}))

        self.routing.run_workflow(max_iterations=2)

        first = self.read_json("eval/iter1/perf_result.json")
        second = self.read_json("eval/iter2/perf_result.json")
        self.assertIs(first["comparison_context_stable"], True)
        self.assertIs(second["comparison_context_stable"], True)
        self.assertEqual(first["comparison_context"]["evaluation_device_id"], 0)
        self.assertEqual(second["comparison_context"]["evaluation_device_id"], 1)
        self.assertEqual(first["comparison_context"]["hardware"],
                         second["comparison_context"]["hardware"])
        stored_device = json.loads(device_path.read_text(encoding="utf-8"))
        for key, value in json.loads(original_device).items():
            self.assertEqual(stored_device[key], value)
        self.assertEqual(stored_device["framework"], "Triton")
        self.assertEqual(stored_device["driver_backend"], "npu")
        self.assert_new_baseline(2, 3.0)
        self.assert_no_patterns_or_rollback(2)
        self.assertEqual(self.routing.fixture.events.count("stage6"), 2)

    def test_metadata_change_does_not_restore_apparent_regression(self):
        self.routing.perfs = {1: (3.0,), 2: (1.5,)}
        baseline = "metadata/synthetic_baseline.json"
        self.routing.write_json(baseline, {"synthetic_op_1": {"baseline_us": 100}})
        self.change_after_first_review(lambda: self.routing.write_json(
            baseline, {"synthetic_op_1": {"baseline_us": 50}}))

        self.routing.run_workflow(max_iterations=2)

        first = self.read_json("eval/iter1/perf_result.json")["comparison_context"]
        second = self.read_json("eval/iter2/perf_result.json")["comparison_context"]
        self.assertNotEqual(first["baseline_metadata_sha256"], second["baseline_metadata_sha256"])
        self.assertEqual(first["task_sha256"], second["task_sha256"])
        self.assertEqual(first["evaluator_sha256"], second["evaluator_sha256"])
        self.assert_new_baseline(2, 1.5)
        self.assert_no_patterns_or_rollback(2)
        self.assertEqual((self.work / "impl/synthetic_op.py").read_text(encoding="utf-8"),
                         "# implementation revision 2\n")

    def test_timer_metric_change_is_not_a_regression_on_underperforming_route(self):
        self.routing.perfs = {1: (0.8, 3.2), 2: (0.4, 1.6)}
        original_context = orchestrator.comparison_context

        def context_for_iteration(*args, **kwargs):
            context = original_context(*args, **kwargs)
            if self.routing.fixture.read_state()["iteration"] >= 2:
                context["metric"] = "elapsed_time"
            return context

        with patch.object(orchestrator, "comparison_context", side_effect=context_for_iteration):
            self.routing.run_workflow(max_iterations=2)

        self.assertEqual(self.read_json("eval/iter1/perf_result.json")
                         ["comparison_context"]["metric"], "kernel_details")
        self.assertEqual(self.read_json("eval/iter2/perf_result.json")
                         ["comparison_context"]["metric"], "elapsed_time")
        self.assert_new_baseline(2, 1.0)
        self.assert_no_patterns_or_rollback(2)
        self.assertEqual([stage for iteration, stage, _ in self.routing.seen if iteration == 2],
                         ["stage4", "stage5", "stage7", "stage8", "stage9", "stage3", "stage10"])

    def test_context_change_during_measurement_excludes_result_from_best_and_window(self):
        self.routing.perfs = {1: (1.5,), 2: (3.0,)}
        baseline = "metadata/synthetic_baseline.json"
        self.routing.write_json(baseline, {"synthetic_op_1": {"baseline_us": 100}})
        original_performance = self.routing.performance

        def performance_with_context_change(*args, **kwargs):
            result = original_performance(*args, **kwargs)
            if self.routing.fixture.read_state()["iteration"] == 2:
                # Change a real context input after Stage6 starts and before its
                # evaluator returns, while preserving the implementation itself.
                self.routing.write_json(baseline, {"synthetic_op_1": {"baseline_us": 200}})
            return result

        self.routing.fixture.performance = performance_with_context_change
        self.routing.run_workflow(max_iterations=2)

        first = self.read_json("eval/iter1/perf_result.json")
        second = self.read_json("eval/iter2/perf_result.json")
        self.assertIs(first["comparison_context_stable"], True)
        self.assertIs(second["comparison_context_stable"], False)
        # Both stamps describe the start of each measurement. Their equality
        # cannot make an evaluation with an unstable context comparable.
        self.assertEqual(first["comparison_context"], second["comparison_context"])
        self.assertEqual(second["avg_speedup"], 3.0)
        comparison = self.request(2)["perf_diff"]
        self.assertFalse(comparison["comparable"])
        self.assertFalse(comparison["has_improvement"])
        self.assertFalse(comparison["has_regression"])
        self.assertEqual(comparison["comparison_status"], "unavailable")
        self.assertTrue(comparison["comparison_reason"])
        status = self.routing.selection_status()
        self.assertFalse(status["eligible"])
        self.assertFalse(status["should_exit"])
        self.assertEqual(status["best"]["iteration"], 1)
        self.assertEqual(status["best"]["avg_speedup"], 1.5)
        self.assertEqual(status["window"]["valid_samples"], 1)
        self.assertEqual(status["window"]["end_iteration"], 1)
        history = load_history(str(self.work))
        row = next(row for row in history["rounds"] if row["iter"] == 2)
        entry = next(row for row in history["ledger"] if row["iter"] == 2)
        self.assertIsNone(row["improved"])
        self.assertIsNone(entry["avg_speedup_before"])
        self.assertIsNone(entry["delta"])
        self.assertEqual(entry["comparison_status"], "unavailable")
        self.assert_no_patterns_or_rollback(2)
        self.assertEqual(self.routing.fixture.events.count("stage6"), 2)

    def test_legacy_report_is_preserved_and_next_same_context_gain_records_once(self):
        self.routing.perfs = {1: (3.0,), 2: (1.5,), 3: (1.8,)}
        legacy_bytes = []

        def make_first_report_legacy():
            report = self.read_json("eval/iter1/perf_result.json")
            report.pop("comparison_context", None)
            report.pop("comparison_context_stable", None)
            self.routing.write_json("eval/iter1/perf_result.json", report)
            legacy_bytes.append((self.work / "eval/iter1/perf_result.json").read_bytes())

        self.change_after_first_review(make_first_report_legacy)
        self.routing.run_workflow(max_iterations=3)

        self.assertEqual((self.work / "eval/iter1/perf_result.json").read_bytes(), legacy_bytes[0])
        self.assert_new_baseline(2, 1.5)
        self.assertIsNone(load_regression_action(self.work, 2))
        comparison = self.request(3)["perf_diff"]
        self.assertTrue(comparison["comparable"])
        self.assertTrue(comparison["has_improvement"])
        self.assertFalse(comparison["has_regression"])
        self.assertEqual(comparison["comparison_status"], "comparable")
        self.assertEqual(comparison["prev_iter"], 2)
        self.assertEqual(comparison["delta_pct"], 20.0)
        proven = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(proven.count("LOCAL_TILE_SUCCESS"), 1)
        regression = self.work / "knowledge/regression_patterns.md"
        self.assertNotIn("LOCAL_TILE_REGRESSION",
                         regression.read_text(encoding="utf-8") if regression.exists() else "")
        entry = next(row for row in load_history(str(self.work))["ledger"] if row["iter"] == 3)
        self.assertEqual(entry["avg_speedup_before"], 1.5)
        self.assertEqual(entry["delta"], 0.3)
        self.assertEqual(entry["verdict"], "big_win")

    def test_returning_context_does_not_skip_the_latest_different_context_report(self):
        self.routing.perfs = {1: (1.5,), 2: (2.0,), 3: (3.0,)}
        original_context = orchestrator.comparison_context

        def context_for_iteration(*args, **kwargs):
            context = original_context(*args, **kwargs)
            if self.routing.fixture.read_state()["iteration"] == 2:
                context["metric"] = "elapsed_time"
            return context

        with patch.object(orchestrator, "comparison_context", side_effect=context_for_iteration):
            self.routing.run_workflow(max_iterations=3)

        self.assertEqual(self.read_json("eval/iter1/perf_result.json")["comparison_context"],
                         self.read_json("eval/iter3/perf_result.json")["comparison_context"])
        self.assert_new_baseline(2, 2.0)
        self.assert_new_baseline(3, 3.0)
        self.assert_no_patterns_or_rollback(3)

    def test_actual_baseline_change_restarts_best_group_and_restores_matching_snapshot(self):
        self.routing.perfs = {1: (5.0,), 2: (2.0,), 3: (1.8,)}
        original_performance = self.routing.performance
        original_agent = self.routing.fixture.agent
        second_selection = []
        restored_implementations = []

        def performance_with_actual_baseline(*args, **kwargs):
            success, output, report_path = original_performance(*args, **kwargs)
            iteration = self.routing.fixture.read_state()["iteration"]
            report = self.read_json(report_path)
            baseline_us = 100 if iteration == 1 else 20
            for case in report["operators"][0]["cases"]:
                case["baseline_perf_us"] = baseline_us
                case["elapsed_us"] = baseline_us / case["speedup"]
            # Rewrite the evaluator's raw report before parsing, so archived
            # source evidence and perf_result contain the same measurements.
            self.routing.write_json(report_path, report)
            return success, output, report_path

        def capture_second_selection(output, payload, iteration, _prompt):
            if iteration == 2:
                second_selection.append(copy.deepcopy(self.routing.selection_status()))
            self.routing.write_json(output, payload)
            return True

        def capture_restoration(agent, role, work, prompt, **kwargs):
            if (fusion_routing_tests.ROLES[Path(role).name] == "stage3"
                    and self.routing.fixture.read_state()["iteration"] == 3):
                restored_implementations.append(
                    (self.work / "impl/synthetic_op.py").read_text(encoding="utf-8"))
            return original_agent(agent, role, work, prompt, **kwargs)

        self.routing.fixture.performance = performance_with_actual_baseline
        self.routing.fixture.agent = capture_restoration
        self.routing.stage9_callback = capture_second_selection
        self.routing.run_workflow(max_iterations=3)

        contexts = [self.read_json(f"eval/iter{iteration}/perf_result.json")["comparison_context"]
                    for iteration in (1, 2, 3)]
        self.assertEqual(contexts[0], contexts[1])
        self.assertEqual(contexts[1], contexts[2])
        self.assert_new_baseline(2, 2.0)
        self.assertIsNone(load_regression_action(self.work, 2))
        self.assertEqual(len(second_selection), 1)
        second = second_selection[0]
        self.assertTrue(second["eligible"])
        self.assertEqual(second["best"]["iteration"], 2)
        self.assertEqual(second["best"]["avg_speedup"], 2.0)
        self.assertEqual(second["window"]["valid_samples"], 1)
        self.assertEqual(second["window"]["start_iteration"], 2)
        comparison = self.request(3)["perf_diff"]
        self.assertTrue(comparison["comparable"])
        self.assertTrue(comparison["has_regression"])
        self.assertFalse(comparison["has_improvement"])
        self.assertEqual(comparison["prev_iter"], 2)
        self.assertEqual(comparison["delta_pct"], -10.0)
        regression = (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8")
        self.assertEqual(regression.count("LOCAL_TILE_REGRESSION"), 1)
        self.assertIn("-10.0", regression)
        action = load_regression_action(self.work, 3)
        self.assertEqual(action["status"], "restored")
        self.assertEqual(action["source_iteration"], 2)
        self.assertEqual(action["best_record"]["iteration"], 2)
        self.assertEqual(action["best_implementation_dir"], second["best"]["implementation_dir"])
        # Iter2 evaluated the implementation delivered by Stage3 in iter1.
        # Observe the actual restored bytes before Stage3 edits them again.
        self.assertEqual(restored_implementations, ["# implementation revision 1\n"])
        self.assertEqual(self.routing.selection_status()["window"]["valid_samples"], 2)

    def assert_resume_rechecks_cached_diff(self, *, has_regression, underperforming):
        speeds = (0.8, 2.0) if underperforming else (1.6,)
        self.routing.perfs = {1: speeds, 2: speeds}
        self.routing.interrupt_stage9_iteration = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.routing.run_workflow(max_iterations=2)

        state = self.routing.fixture.read_state()
        self.assertEqual(state["current_stage"], "iter2_stage9")
        stale = copy.deepcopy(state["stage9_context"]["perf_diff"])
        stale.update(
            comparable=True, comparison_status="comparable", comparison_reason="old cached result",
            has_improvement=not has_regression, has_regression=has_regression,
            prev_iter=1, curr_iter=2, prev_avg_speedup=1.0, curr_avg_speedup=2.0,
            delta_pct=-50.0 if has_regression else 100.0,
            diff_text="" if has_regression else "STALE_COMPARISON_GAIN_REQUIREMENT",
            regression_text="STALE_COMPARISON_LOSS_REQUIREMENT" if has_regression else "")
        state["stage9_context"]["perf_diff"] = stale
        self.routing.write_json(".state.json", state)
        self.routing.config["hardware"] = {"device_id": 1}
        self.routing.interrupt_stage9_iteration = None

        self.routing.run_workflow()

        comparison = self.request(2)["perf_diff"]
        self.assertFalse(comparison["comparable"])
        self.assertFalse(comparison["has_improvement"])
        self.assertFalse(comparison["has_regression"])
        self.assertEqual(comparison["comparison_status"], "unavailable")
        self.assertTrue(comparison["comparison_reason"])
        prompt = self.routing.fixture.prompts["stage9"]
        self.assertNotIn("STALE_COMPARISON_GAIN_REQUIREMENT", prompt)
        self.assertNotIn("STALE_COMPARISON_LOSS_REQUIREMENT", prompt)
        self.assertFalse(comparison.get("diff_text"))
        self.assertFalse(comparison.get("regression_text"))
        self.assert_no_patterns_or_rollback(2)
        self.assertEqual(self.routing.fixture.events.count("stage6"), 2)
        self.assertEqual(self.routing.fixture.read_state()["stopped_by"], "max_iterations")

    def test_resume_passing_stage9_ignores_cached_improvement_after_device_change(self):
        self.assert_resume_rechecks_cached_diff(has_regression=False, underperforming=False)

    def test_resume_underperforming_stage9_ignores_cached_regression_after_device_change(self):
        self.assert_resume_rechecks_cached_diff(has_regression=True, underperforming=True)


if __name__ == "__main__":
    unittest.main()

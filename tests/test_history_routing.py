"""History uses average speedups and survives interrupted/failed evaluations."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import orchestrator
from lib.history_manager import load_history, save_history
from lib.state import State
import test_semantic_routing as semantic_routing_tests


class HistoryPreviousAverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.state = State.load_or_create(str(self.work))
        self.state.iteration = 3

    def report(self, iteration, payload):
        path = self.work / "eval" / f"iter{iteration}" / "perf_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def rounds(self, rows):
        history = load_history(str(self.work))
        history["rounds"] = rows
        save_history(str(self.work), history)

    def previous(self, log=None):
        return orchestrator._history_previous_average(str(self.work), self.state, log)

    def test_report_average_wins_over_old_state_overall_score_and_history(self):
        self.state.history = [{"iteration": 2, "speedup": 90.8189}]
        self.rounds([{"iter": 2, "avg_speedup": 3.9}])
        self.report(2, {"avg_speedup": 3.9199, "overall_score": 90.8189})
        self.assertEqual(self.previous(), 3.9199)

    def test_missing_report_uses_history_average_not_old_state_score(self):
        self.state.history = [{"iteration": 2, "speedup": 90.8189}]
        self.rounds([{"iter": 2, "avg_speedup": 3.9199}])
        self.assertEqual(self.previous(), 3.9199)

    def test_malformed_report_uses_last_history_entry_for_previous_round(self):
        self.state.history = [{"iteration": 2, "speedup": 99.0}]
        self.rounds([{"iter": 2, "avg_speedup": 3.8}, {"iter": 2, "avg_speedup": 3.9199}])
        self.report(2, {})
        path = self.work / "eval/iter2/perf_result.json"
        path.write_text("{interrupted-write", encoding="utf-8")
        self.assertEqual(self.previous(), 3.9199)

    def test_previous_round_selection_ignores_current_future_and_list_order(self):
        self.state.iteration = 4
        self.state.history = [
            {"iteration": 3, "speedup": 80.0}, {"iteration": 4, "speedup": 90.0},
            {"iteration": 6, "speedup": 100.0}, {"iteration": 1, "speedup": 70.0},
        ]
        self.rounds([
            {"iter": 6, "avg_speedup": 6.0}, {"iter": 3, "avg_speedup": 1.8},
            {"iter": 4, "avg_speedup": 4.0}, {"iter": 1, "avg_speedup": 2.0},
        ])
        self.assertEqual(self.previous(), 1.8)

    def test_zero_previous_round_is_not_skipped_for_earlier_positive_average(self):
        self.rounds([{"iter": 1, "avg_speedup": 2.0}, {"iter": 2, "avg_speedup": 0.0}])
        self.report(1, {"avg_speedup": 2.0})
        self.report(2, {"avg_speedup": 0.0, "score_error_code": "execution_failed"})
        self.assertEqual(self.previous(), 0.0)

    def test_unknown_previous_average_is_none_not_score_or_earlier_result(self):
        self.state.history = [{"iteration": 2, "speedup": 90.8189}]
        self.rounds([{"iter": 1, "avg_speedup": 2.0}])
        log = Mock()
        self.assertIsNone(self.previous(log))
        log.warning.assert_called_once()

    def test_first_evaluation_uses_zero_and_does_not_use_current_or_future(self):
        self.state.iteration = 1
        self.state.history = [{"iteration": 1, "speedup": 90.0}, {"iteration": 4, "speedup": 99.0}]
        self.rounds([{"iter": 1, "avg_speedup": 2.0}, {"iter": 4, "avg_speedup": 4.0}])
        self.assertEqual(self.previous(), 0.0)

    def test_perf_diff_invalid_current_averages_do_not_signal_gain_or_loss(self):
        self.report(1, {"avg_speedup": 2.0})
        for invalid in (None, float("nan"), float("inf"), True, "2.0"):
            with self.subTest(average=invalid):
                self.report(2, {"avg_speedup": invalid})
                diff = orchestrator._compute_perf_diff(str(self.work), 2)
                self.assertFalse(diff["has_improvement"])
                self.assertFalse(diff["has_regression"])
                self.assertNotIn("delta_pct", diff)

    def test_perf_diff_invalid_previous_averages_do_not_signal_gain_or_loss(self):
        self.report(2, {"avg_speedup": 2.0})
        for invalid in (None, float("nan"), float("inf"), True, "2.0"):
            with self.subTest(average=invalid):
                self.report(1, {"avg_speedup": invalid})
                diff = orchestrator._compute_perf_diff(str(self.work), 2)
                self.assertFalse(diff["has_improvement"])
                self.assertFalse(diff["has_regression"])
                self.assertNotIn("delta_pct", diff)

    def test_perf_diff_does_not_classify_incomplete_legacy_case_metrics(self):
        self.report(1, {"avg_speedup": 2.0, "cases": [
            {"case_id": "valid", "speedup": 2.0},
            {"case_id": "null_current", "speedup": 2.0},
            {"case_id": "null_previous", "speedup": None},
            {"case_id": "nan_current", "speedup": 2.0},
            {"case_id": "nan_previous", "speedup": float("nan")},
        ]})
        self.report(2, {"avg_speedup": 1.8, "cases": [
            {"case_id": "valid", "speedup": 1.8},
            {"case_id": "null_current", "speedup": None},
            {"case_id": "null_previous", "speedup": 2.0},
            {"case_id": "nan_current", "speedup": float("nan")},
            {"case_id": "nan_previous", "speedup": 2.0},
        ]})
        diff = orchestrator._compute_perf_diff(str(self.work), 2)
        self.assertFalse(diff["has_regression"])
        self.assertFalse(diff["has_improvement"])
        self.assertFalse(diff["comparable"])
        self.assertNotIn("case_diffs", diff)


class HistoryRoutingTests(unittest.TestCase):
    def setUp(self):
        # Reuse transports, not the parent TestCase's tests. Real routing,
        # report parsing, history merging and checkpoint reads still execute.
        self.routing = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        self.routing.setUp()
        self.addCleanup(self.routing.doCleanups)
        self.work = self.routing.work

    def test_resume_after_stage6_history_write_compares_against_previous_round(self):
        self.routing.perfs = {1: (2.0,), 2: (1.8,)}
        self.routing.config["workflow"]["semantic_exit"]["passed_window"] = 8
        original_append = orchestrator.append_round

        def interrupt_after_write(work, iteration, *args, **kwargs):
            result = original_append(work, iteration, *args, **kwargs)
            if iteration == 2:
                raise RuntimeError("synthetic interruption after Stage6 history write")
            return result

        with patch.object(orchestrator, "append_round", side_effect=interrupt_after_write):
            with self.assertRaisesRegex(RuntimeError, "after Stage6 history write"):
                self.routing.run_workflow(max_iterations=2)
        self.assertEqual(self.routing.fixture.read_state()["current_stage"], "iter2_stage6")
        before = next(row for row in load_history(str(self.work))["ledger"] if row["iter"] == 2)
        self.routing.run_workflow()
        after = next(row for row in load_history(str(self.work))["ledger"] if row["iter"] == 2)
        for row in (before, after):
            self.assertEqual(row["avg_speedup_before"], 2.0)
            self.assertEqual(row["avg_speedup_after"], 1.8)
            self.assertEqual(row["delta"], -0.2)
            self.assertEqual(row["verdict"], "regression")
        state = self.routing.fixture.read_state()
        self.assertEqual(sum(row["iteration"] == 2 for row in state["history"]), 1)
        self.assertEqual(self.routing.fixture.events.count("stage6"), 3)

    def test_null_case_speedup_report_reaches_stage9_evaluation_error(self):
        self.routing.perfs = {1: (0.0,)}

        def failed_report(*_args, **_kwargs):
            self.routing.fixture.events.append("stage6")
            report = self.work / "cannbench_null_case.json"
            self.routing.write_json(report, {"operators": [{
                "score": 0, "avg_speedup": 0, "performance_score": 0,
                "total_cases": 1, "passed_cases": 0, "failed_cases": 1,
                "score_error_code": "execution_failed", "score_error": "kernel execution failed",
                "cases": [{"case_id": "synthetic_op_1", "status": "failed",
                           "speedup": None, "elapsed_us": None, "baseline_perf_us": 100}],
            }]})
            return True, "", str(report)

        self.routing.fixture.performance = failed_report
        self.routing.run_workflow(max_iterations=1)
        history = load_history(str(self.work))
        self.assertIsNone(history["rounds"][0]["worst_case"]["speedup"])
        self.assertTrue(history["ledger"][0]["reason"].startswith("score_zero: execution_failed"))
        self.assertEqual([stage for _, stage, _ in self.routing.seen if stage in {"stage7", "stage8", "stage9"}],
                         ["stage9"])
        request_files = list((self.work / "knowledge/stage9/iter1").glob("*/request.json"))
        self.assertEqual(len(request_files), 1)
        request = json.loads(request_files[0].read_text(encoding="utf-8"))
        self.assertEqual(request["scene"], "evaluation_error")

    def test_null_case_with_positive_average_and_error_survives_diff_and_routes_to_stage9(self):
        self.routing.perfs = {1: (2.0,), 2: (1.9,)}

        def performance(*args, **kwargs):
            iteration = self.routing.fixture.read_state()["iteration"]
            if iteration == 1:
                return self.routing.performance(*args, **kwargs)
            self.routing.fixture.events.append("stage6")
            report = self.work / "cannbench_partially_failed.json"
            self.routing.write_json(report, {"operators": [{
                "score": 0, "avg_speedup": 1.9, "performance_score": 0,
                "total_cases": 1, "passed_cases": 0, "failed_cases": 1,
                "score_error_code": "eval_error", "score_error": "partial performance report",
                "cases": [{"case_id": "synthetic_op_1", "status": "failed",
                           "speedup": None, "elapsed_us": None, "baseline_perf_us": 100}],
            }]})
            return True, "", str(report)

        self.routing.fixture.performance = performance
        self.routing.run_workflow(max_iterations=2)
        history = load_history(str(self.work))
        current = next(row for row in history["rounds"] if row["iter"] == 2)
        self.assertEqual(current["avg_speedup"], 1.9)
        self.assertIsNone(current["worst_case"]["speedup"])
        self.assertEqual([stage for iteration, stage, _ in self.routing.seen
                          if iteration == 2 and stage in {"stage7", "stage8", "stage9"}], ["stage9"])
        request_path = next((self.work / "knowledge/stage9/iter2").glob("*/request.json"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(request["scene"], "evaluation_error")
        self.assertEqual(self.routing.selection_status()["best"]["iteration"], 1)
        self.assertFalse((self.work / "knowledge/regression_patterns.md").exists())


if __name__ == "__main__":
    unittest.main()

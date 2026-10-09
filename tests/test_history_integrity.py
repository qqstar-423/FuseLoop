import json
from pathlib import Path
import tempfile
import unittest

from lib.bench_parser import parse_cannbench_report_to_perf_result
from lib.history_manager import (
    append_round, format_for_prompt, get_latest_fix_plan, load_history, save_history,
)


class HistoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def parse_report(self, cases, **metrics):
        report = self.work / "report.json"
        report.write_text(json.dumps({"operators": [{
            "total_cases": len(cases), "avg_speedup": 2.123456,
            "score": 86.126, "performance_score": 12.346,
            "cases": cases, **metrics,
        }]}), encoding="utf-8")
        return parse_cannbench_report_to_perf_result(str(report))

    def test_real_parser_null_and_mixed_case_metrics_remain_unknown(self):
        values = [None, "not measured", False, float("nan"), -1, 0.8, 1.4]
        cases = [{"case_id": f"case_{i}", "speedup": speedup,
                  "elapsed_us": 10, "status": "success"}
                 for i, speedup in enumerate(values)]
        perf = self.parse_report(cases)
        self.assertFalse(perf["perf_pass"])
        history = append_round(self.work, 2, perf, 2.0)
        row = history["rounds"][0]
        self.assertEqual(row["worst_case"], {"case_id": "case_0", "speedup": None})
        self.assertEqual([c["speedup"] for c in row["worst_6_speedups"]],
                         [None, None, None, None, None, 0.8])
        # The saved history must be standard JSON: no NaN/Infinity or fake zero.
        json.dumps(load_history(self.work), allow_nan=False)
        self.assertEqual(load_history(self.work), history)
        snapshot = json.loads((self.work / "knowledge/_rounds_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot, history["rounds"])

    def test_real_parser_normal_sort_and_rounding_are_unchanged(self):
        cases = [{"case_id": f"case_{i}", "speedup": speedup,
                  "elapsed_us": 10, "status": "success"}
                 for i, speedup in enumerate([3.2, 0.987654, 2.4, 1.8, 1.1, 1.5, 2.1])]
        history = append_round(self.work, 3, self.parse_report(cases), 2.0)
        row, ledger = history["rounds"][0], history["ledger"][0]
        self.assertEqual([c["speedup"] for c in row["worst_6_speedups"]],
                         [0.9877, 1.1, 1.5, 1.8, 2.1, 2.4])
        self.assertEqual(row["avg_speedup"], 2.1235)
        self.assertEqual(row["overall_score"], 86.13)
        self.assertEqual(row["performance_score"], 12.35)
        self.assertTrue(row["improved"])
        self.assertEqual(ledger["delta"], 0.1235)
        self.assertEqual(ledger["verdict"], "big_win")

    def test_missing_and_nonfinite_aggregates_do_not_become_scores(self):
        for value in (None, "bad", False, float("nan"), float("inf"), -1):
            with self.subTest(value=value):
                history = append_round(self.work, 2, {"avg_speedup": value}, 2.0)
                row, ledger = history["rounds"][0], history["ledger"][0]
                self.assertIsNone(row["avg_speedup"])
                self.assertIsNone(row["overall_score"])
                self.assertIsNone(row["performance_score"])
                self.assertIsNone(row["improved"])
                self.assertIsNone(row["worst_case"]["speedup"])
                self.assertIsNone(ledger["avg_speedup_after"])
                self.assertIsNone(ledger["delta"])
                self.assertIsNone(ledger["verdict"])
                json.dumps(history, allow_nan=False)

    def test_invalid_previous_average_cannot_report_improvement(self):
        for previous in (None, "bad", False, float("inf")):
            with self.subTest(previous=previous):
                history = append_round(self.work, 2, {"avg_speedup": 2.0}, previous)
                self.assertIsNone(history["rounds"][0]["improved"])
                self.assertIsNone(history["ledger"][0]["avg_speedup_before"])
                self.assertIsNone(history["ledger"][0]["delta"])
                self.assertIsNone(history["ledger"][0]["verdict"])

    def test_reported_zero_keeps_existing_result_classification(self):
        baseline = append_round(self.work, 1, {"avg_speedup": 0.0}, 0.0)
        self.assertEqual(baseline["rounds"][0]["avg_speedup"], 0.0)
        self.assertEqual(baseline["ledger"][0]["verdict"], "baseline")
        history = append_round(self.work, 2, {"avg_speedup": 0.0}, 2.0)
        self.assertFalse(history["rounds"][-1]["improved"])
        self.assertEqual(history["ledger"][-1]["verdict"], "regression")

    def test_fallback_worst_cases_handles_null(self):
        history = append_round(self.work, 1, {"avg_speedup": 1.0, "cases": None,
            "worst_6_cases": [{"case_id": "case_1", "speedup": None}]})
        self.assertEqual(history["rounds"][0]["worst_case"]["case_id"], "case_1")
        self.assertIsNone(history["rounds"][0]["worst_case"]["speedup"])

    def test_old_file_scope_cannot_override_current_plan(self):
        history = load_history(self.work)
        history["ledger"] = [
            {"iter": 1, "reason": "perf_optimize", "direction": "adjust the main block",
             "modify_files": ["main.py"], "readonly_files": ["helper.py"],
             "verdict": "regression", "avg_speedup_before": 2.0, "avg_speedup_after": 1.8},
            {"iter": 2, "reason": "perf_optimize", "direction": "shrink the tail block",
             "evaluation_summary": "the main-block change did not clearly improve the average; the tail block is still slow",
             "modify_files": ["helper.py"], "readonly_files": ["main.py"],
             "verdict": "no_change", "avg_speedup_before": 1.8, "avg_speedup_after": 1.8},
        ]
        save_history(self.work, history)
        prompt = format_for_prompt(self.work)
        self.assertIn("read-only at the time (review only): helper.py", prompt)
        self.assertIn("current plan may modify: helper.py", prompt)
        self.assertNotIn("current plan must not modify: helper.py", prompt)
        self.assertIn("This round's evaluation review: the main-block change did not clearly improve the average; the tail block is still slow", prompt)
        self.assertIn("Next-step plan proposed after evaluation (not yet judged by this entry's verdict): shrink the tail block", prompt)
        self.assertIn("Legacy direction (executed changes/next-step plan were not distinguished then): adjust the main block", prompt)
        self.assertIn("must not be treated as an already-failed change", prompt)
        plan = get_latest_fix_plan(self.work)
        self.assertIn("Files that may be modified (change only these):\n  - helper.py", plan)
        self.assertIn("Files that must not be modified (absolutely untouchable):\n  - main.py", plan)

    def test_pending_latest_row_does_not_reactivate_old_scope(self):
        history = load_history(self.work)
        history["ledger"] = [
            {"iter": 1, "direction": "old plan", "modify_files": ["old.py"], "readonly_files": []},
            {"iter": 2, "direction": "(pending tech_lead)", "modify_files": [], "readonly_files": []},
        ]
        save_history(self.work, history)
        self.assertNotIn("current plan may modify", format_for_prompt(self.work))
        self.assertEqual(get_latest_fix_plan(self.work), "")

    def test_historical_labels_do_not_overrule_latest_review_p0(self):
        history = load_history(self.work)
        history.update({
            "suggest_next": [{"priority": "P0", "action": "recheck tail-block conditions", "reason": "conditions changed"}],
            "insights": ["❌ rejected: old shape tiling failed"],
            "worst_cases_tracker": {"case_1": "hardware limitation, to be verified"},
            "fusion_kernel_strategy": [{"iter": 1, "strategy": "partial fusion", "status": "pending verification"}],
        })
        save_history(self.work, history)
        prompt = format_for_prompt(self.work)
        self.assertIn("when conditions or evidence change, re-verify per the latest plan", prompt)
        self.assertIn("never justifies permanently skipping", prompt)
        self.assertIn("partial fusion", prompt)
        self.assertNotIn("fusion direction >", prompt)
        self.assertNotIn("do not try this direction again", prompt)
        self.assertNotIn("do not waste time", prompt)

    def test_unknown_history_metrics_render_without_false_downtrend(self):
        append_round(self.work, 1, {"cases": [{"case_id": "case_1", "speedup": None}]}, None)
        prompt = format_for_prompt(self.work)
        self.assertIn("avg_speedup=unknown ?", prompt)
        self.assertIn("worst=case_1(unknown)", prompt)
        self.assertNotIn("None", prompt)

    def test_structured_case_tracker_shows_source_iteration_and_readable_conclusion(self):
        history = load_history(self.work)
        history["worst_cases_tracker"] = {
            "case_1": {"iteration": 3, "observation": "0.8x → 0.9x", "explanation": "tail-block utilization is insufficient",
                       "next_action": "try a smaller tail block", "evidence": "profile/iter3/bottleneck_analysis.md"},
            "case_2": {"iteration": 1, "explanation": "historical memory-access bottleneck"},
        }
        save_history(self.work, history)
        prompt = format_for_prompt(self.work)
        self.assertIn("case_1 [iter3]\n  Conclusion: tail-block utilization is insufficient", prompt)
        self.assertIn("Next action: try a smaller tail block", prompt)
        self.assertIn("Evidence: profile/iter3/bottleneck_analysis.md", prompt)
        self.assertIn("case_2 [iter1]", prompt)
        self.assertNotIn("{'iteration':", prompt)


if __name__ == "__main__":
    unittest.main()

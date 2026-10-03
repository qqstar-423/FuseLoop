"""Offline checks for cann-bench strategy selection and report consumption."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from lib.bench_parser import run_perf_eval, parse_cannbench_report_to_perf_result


class CannBenchReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report_path = Path(self.temp.name) / "op_eval.json"
        self.cases = [
            {"case_id": f"level3/op_{index}", "status": "success",
             "baseline_perf_us": 24.65, "elapsed_us": 7.2, "speedup": 3.42,
             "t_hw_us": 2.46, "perf_score": 0.824,
             "op_times": {"device_kernels": {"K1": 4.76, "K2": 2.43}}}
            for index in range(1, 9)
        ]
        self.operator = {
            "score": 91.2, "compilation_score": 20.0, "compile_runtime_score": 20.0,
            "function_score": 30.0, "performance_score": 41.2, "avg_speedup": 3.42,
            "total_cases": 8, "passed_cases": 8, "failed_cases": 0,
            "score_error_code": None, "score_error": None, "cases": self.cases,
        }

    def parse(self):
        self.report_path.write_text(json.dumps({"operators": [self.operator]}), encoding="utf-8")
        before = self.report_path.read_bytes()
        result = parse_cannbench_report_to_perf_result(str(self.report_path))
        self.assertEqual(self.report_path.read_bytes(), before)
        return result

    def test_tool_invocation_selects_kernel_details_and_only_runs_once(self):
        with patch("lib.bench_parser._build_eval_env", return_value={"PYTHONPATH": ""}), \
             patch("lib.bench_parser.subprocess.run") as launch, \
             patch("lib.bench_parser._find_latest_report", return_value=str(self.report_path)):
            launch.return_value = subprocess.CompletedProcess([], 0, "tool output", "")
            success, output, path = run_perf_eval("task/op", "tool/src", device_id=1)
        self.assertTrue(success)
        self.assertEqual(path, str(self.report_path))
        self.assertEqual(output, "tool output")
        launch.assert_called_once()
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--perf-metric-strategy") + 1], "kernel_details")
        self.assertEqual(command[command.index("--device-id") + 1], "1")
        self.assertEqual(command[command.index("--warmup") + 1], "2")
        self.assertEqual(command[command.index("--repeat") + 1], "3")

    def test_all_original_metrics_are_preserved_without_recalculation(self):
        # Reported rounding intentionally differs from calculating B/C or HAP
        # afresh. The workflow must preserve the tool's reported numbers.
        result = self.parse()
        self.assertTrue(result["perf_pass"])
        for name in ("compilation_score", "compile_runtime_score", "function_score",
                     "performance_score", "avg_speedup", "total_cases", "passed_cases", "failed_cases"):
            self.assertEqual(result[name], self.operator[name])
        self.assertEqual(result["overall_score"], self.operator["score"])
        self.assertEqual(len(result["cases"]), 8)
        self.assertEqual(len(result["worst_6_cases"]), 6)
        for original, parsed in zip(self.cases, result["cases"]):
            for field, value in original.items():
                self.assertEqual(parsed[field], value)
            self.assertEqual(parsed["baseline_us"], original["baseline_perf_us"])
        self.assertNotIn("measurement_provenance", result)

    def test_single_kernel_and_two_kernel_reports_use_their_reported_speedup(self):
        self.cases[0]["op_times"] = {"device_kernels": {"single_kernel": 7.2}}
        result = self.parse()
        self.assertTrue(result["perf_pass"])
        self.assertEqual(result["cases"][0]["speedup"], result["cases"][1]["speedup"])

    def test_every_case_must_pass_and_exactly_one_is_allowed(self):
        self.cases[0]["speedup"] = 1.0
        self.assertTrue(self.parse()["perf_pass"])
        self.cases[0]["speedup"] = 0.999
        self.assertFalse(self.parse()["perf_pass"])

    def test_failed_missing_invalid_and_tool_rejected_cases_cannot_pass(self):
        original = copy.deepcopy(self.operator)
        mutations = [
            lambda op: op["cases"][0].update(status="failed"),
            lambda op: op["cases"].pop(),
            lambda op: op.update(cases=[], total_cases=0),
            lambda op: op["cases"][0].update(elapsed_us=0),
            lambda op: op["cases"][0].update(speedup=float("inf")),
            lambda op: op["cases"][0].update(speedup=float("nan")),
            lambda op: op.update(score_error_code="no_npu_kernel_detected", score=0),
            lambda op: op.update(score_error="cpu fallback", score=0),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.operator = copy.deepcopy(original)
                mutate(self.operator)
                result = self.parse()
                self.assertFalse(result["perf_pass"])
                self.assertEqual(result["overall_score"], self.operator["score"])
                self.assertEqual(result["score_error_code"], self.operator["score_error_code"])


if __name__ == "__main__":
    unittest.main()

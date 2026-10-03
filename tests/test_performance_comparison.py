"""Offline checks for strict, archived single-round comparison evidence."""

from copy import deepcopy
import unittest

from lib.performance_comparison import compare_performance, performance_signature


class PerformanceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "framework": "Triton", "backend": "triton-ascend",
            "runtime_versions": {"triton": "synthetic", "triton_ascend": "synthetic", "torch": "synthetic", "torch_npu": "synthetic"},
            "hardware": {"chip": "test-chip", "driver": "1.0"},
            "evaluation_device_id": 0,
            "hardware_config": {"device_id": 0},
            "task_sha256": "task-and-cases",
            "metric": "kernel_details",
            "baseline_protocol": "cann-bench-native/kernel-details-v1",
            "warmup": 2,
            "repeat": 3,
            "evaluator_sha256": "evaluator-code",
            "baseline_metadata_sha256": "baseline-assets",
        }
        self.report = {
            "comparison_context": deepcopy(self.context),
            "comparison_context_stable": True,
            "avg_speedup": 2.0,
            "total_cases": 2,
            "cases": [
                {"case_id": "op_2", "status": "success", "speedup": 2.0,
                 "elapsed_us": 20.0, "baseline_perf_us": 40.0},
                {"case_id": "op_1", "status": "success", "speedup": 2.0,
                 "elapsed_us": 10.0, "baseline_perf_us": 20.0},
            ],
            "_error": None,
            "score_error": None,
            "score_error_code": None,
        }

    def test_same_context_accepts_changed_timings_without_recalculation(self):
        previous = deepcopy(self.report)
        current = deepcopy(self.report)
        current["avg_speedup"] = 2.3
        current["cases"][0].update(speedup=2.45, elapsed_us=16.3)
        current["cases"][1].update(speedup=2.15, elapsed_us=9.3)
        before = deepcopy((previous, current))
        result = compare_performance(previous, current, deepcopy(self.context))
        self.assertTrue(result["comparable"])
        self.assertEqual(result["mismatch_fields"], [])
        self.assertEqual((previous, current), before)

    def test_case_order_does_not_affect_signature(self):
        reordered = deepcopy(self.report)
        reordered["cases"].reverse()
        original_signature, _ = performance_signature(self.report)
        reordered_signature, _ = performance_signature(reordered)
        self.assertEqual(original_signature, reordered_signature)
        self.assertEqual(list(original_signature["case_baselines"]), ["op_1", "op_2"])
        self.assertTrue(compare_performance(self.report, reordered)["comparable"])

    def test_signature_keeps_full_context_without_retaining_mutable_references(self):
        self.report["comparison_context"]["extra_protocol"] = {"version": 1}
        signature, _ = performance_signature(self.report)
        self.assertEqual(signature["comparison_context"], self.report["comparison_context"])
        self.report["comparison_context"]["hardware"]["chip"] = "changed"
        self.report["comparison_context"]["extra_protocol"]["version"] = 2
        self.assertEqual(signature["comparison_context"]["hardware"]["chip"], "test-chip")
        self.assertEqual(signature["comparison_context"]["extra_protocol"]["version"], 1)

    def test_trace_view_cannot_compare_with_kernel_details(self):
        previous = deepcopy(self.report)
        previous["comparison_context"].update(metric="trace_view", baseline_protocol="trace-view-v1")
        result = compare_performance(previous, self.report)
        self.assertFalse(result["comparable"])
        self.assertIn("计时口径", result["reason"])
        self.assertIn("comparison_context.metric", result["mismatch_fields"])
        self.assertIn("comparison_context.baseline_protocol", result["mismatch_fields"])

    def test_each_context_dimension_changes_comparability(self):
        changes = {
            "framework": "OtherFramework", "backend": "other-backend",
            "runtime_versions": {"triton": "changed"},
            "hardware": {"chip": "other-chip", "driver": "1.0"},
            "evaluation_device_id": 1,
            "hardware_config": {"device_id": 0, "clock": "fixed"},
            "task_sha256": "new-task-or-case-definitions",
            "metric": "trace_view",
            "baseline_protocol": "other-protocol",
            "warmup": 5,
            "repeat": 6,
            "evaluator_sha256": "new-evaluator",
            "baseline_metadata_sha256": "new-baseline-assets",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                current = deepcopy(self.report)
                current["comparison_context"][field] = value
                result = compare_performance(self.report, current)
                self.assertFalse(result["comparable"])
                self.assertEqual(result["mismatch_fields"], [f"comparison_context.{field}"])

    def test_unknown_context_fields_are_also_compared(self):
        current = deepcopy(self.report)
        current["comparison_context"]["future_protocol"] = "v2"
        result = compare_performance(self.report, current)
        self.assertFalse(result["comparable"])
        self.assertEqual(result["mismatch_fields"], ["comparison_context.future_protocol"])

    def test_none_metadata_is_valid_but_differs_from_a_fingerprint(self):
        current = deepcopy(self.report)
        current["comparison_context"]["baseline_metadata_sha256"] = None
        self.assertTrue(compare_performance(current, deepcopy(current))["comparable"])
        result = compare_performance(self.report, current)
        self.assertFalse(result["comparable"])
        self.assertIn("comparison_context.baseline_metadata_sha256", result["mismatch_fields"])

    def test_actual_baseline_change_is_rejected_even_with_same_metadata(self):
        current = deepcopy(self.report)
        current["cases"][0]["baseline_perf_us"] = 40.1
        result = compare_performance(self.report, current)
        self.assertFalse(result["comparable"])
        self.assertIn("实际基准耗时", result["reason"])
        self.assertEqual(result["mismatch_fields"], ["case_baselines.op_2"])

    def test_case_set_change_is_rejected_even_with_same_count(self):
        current = deepcopy(self.report)
        current["cases"][0]["case_id"] = "op_3"
        result = compare_performance(self.report, current)
        self.assertFalse(result["comparable"])
        self.assertIn("case_ids", result["mismatch_fields"])

    def test_legacy_reports_never_get_labels_from_expected_context(self):
        previous = deepcopy(self.report)
        previous.pop("comparison_context")
        previous.pop("comparison_context_stable")
        previous["metric"] = "trace_view"
        before = deepcopy(previous)
        result = compare_performance(previous, self.report, self.context)
        self.assertFalse(result["comparable"])
        self.assertIn("缺少", result["reason"])
        self.assertIn("previous.comparison_context", result["mismatch_fields"])
        self.assertEqual(previous, before)

    def test_missing_context_and_each_required_field_are_rejected(self):
        for field in self.context:
            with self.subTest(field=field):
                current = deepcopy(self.report)
                current["comparison_context"].pop(field)
                signature, reason = performance_signature(current)
                self.assertIsNone(signature)
                self.assertIn(field, reason)
                self.assertFalse(compare_performance(self.report, current)["comparable"])
        for value in (None, {}, [], "context"):
            with self.subTest(value=value):
                current = deepcopy(self.report)
                current["comparison_context"] = value
                self.assertIsNone(performance_signature(current)[0])

    def test_context_stability_must_be_explicitly_true(self):
        for value in (None, False, 1, "true"):
            with self.subTest(value=value):
                current = deepcopy(self.report)
                current["comparison_context_stable"] = value
                self.assertFalse(compare_performance(self.report, current)["comparable"])
        current.pop("comparison_context_stable")
        self.assertIsNone(performance_signature(current)[0])

    def test_invalid_context_values_are_rejected(self):
        invalid = {
            "hardware": ({}, [], None),
            "hardware_config": (None, []),
            "task_sha256": (None, "", "  ", False),
            "metric": (None, ""),
            "baseline_protocol": (None, ""),
            "evaluator_sha256": (None, ""),
            "baseline_metadata_sha256": ("", 0, False),
            "evaluation_device_id": (None, False, -1, ""),
            "warmup": (None, False, -1, 2.0),
            "repeat": (None, False, 0, 3.0),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    current = deepcopy(self.report)
                    current["comparison_context"][field] = value
                    self.assertIsNone(performance_signature(current)[0])

    def test_total_cases_is_strictly_a_positive_integer(self):
        for total in (None, 0, -1, True, 2.0, "2", 3):
            with self.subTest(total=total):
                current = deepcopy(self.report)
                current["total_cases"] = total
                self.assertIsNone(performance_signature(current)[0])

    def test_case_list_must_be_complete(self):
        for cases in (None, {}, [], self.report["cases"][:1], [None, self.report["cases"][1]]):
            with self.subTest(cases=cases):
                current = deepcopy(self.report)
                current["cases"] = cases
                self.assertIsNone(performance_signature(current)[0])

    def test_case_ids_must_be_nonempty_unique_strings(self):
        for case_id in (None, "", " ", 1, "op_1"):
            with self.subTest(case_id=case_id):
                current = deepcopy(self.report)
                current["cases"][0]["case_id"] = case_id
                self.assertIsNone(performance_signature(current)[0])

    def test_cases_must_be_successful_with_positive_finite_metrics(self):
        for status in (None, "failed", "error", True):
            with self.subTest(status=status):
                current = deepcopy(self.report)
                current["cases"][0]["status"] = status
                self.assertIsNone(performance_signature(current)[0])
        for field in ("speedup", "elapsed_us", "baseline_perf_us"):
            for value in (None, 0, -1, True, "2", float("inf"), float("nan"), 10 ** 1000):
                with self.subTest(field=field, value=value):
                    current = deepcopy(self.report)
                    current["cases"][0][field] = value
                    self.assertIsNone(performance_signature(current)[0])
            current = deepcopy(self.report)
            current["cases"][0].pop(field)
            self.assertIsNone(performance_signature(current)[0])

    def test_legacy_baseline_alias_is_supported_without_masking_bad_canonical_values(self):
        current = deepcopy(self.report)
        for case in current["cases"]:
            case["baseline_us"] = case.pop("baseline_perf_us")
        self.assertTrue(compare_performance(self.report, current)["comparable"])
        current["cases"][0]["baseline_perf_us"] = None
        self.assertIsNone(performance_signature(current)[0])

    def test_conflicting_baseline_aliases_are_rejected(self):
        current = deepcopy(self.report)
        current["cases"][0]["baseline_us"] = 40
        self.assertTrue(compare_performance(self.report, current)["comparable"])
        current["cases"][0]["baseline_us"] = 50
        self.assertIsNone(performance_signature(current)[0])

    def test_invalid_average_and_tool_errors_are_rejected(self):
        for average in (None, 0, -1, True, "2", float("inf"), float("nan")):
            with self.subTest(average=average):
                current = deepcopy(self.report)
                current["avg_speedup"] = average
                self.assertIsNone(performance_signature(current)[0])
        for field in ("_error", "score_error", "score_error_code"):
            with self.subTest(field=field):
                current = deepcopy(self.report)
                current[field] = "measurement rejected"
                self.assertIsNone(performance_signature(current)[0])

    def test_underperforming_but_complete_reports_remain_comparable(self):
        current = deepcopy(self.report)
        current["perf_pass"] = False
        current["avg_speedup"] = 0.8
        current["cases"][0]["speedup"] = 0.7
        current["cases"][1]["speedup"] = 0.9
        self.assertTrue(compare_performance(self.report, current)["comparable"])

    def test_expected_context_must_match_current_archived_context(self):
        expected = deepcopy(self.context)
        expected["task_sha256"] = "task-changed-after-evaluation"
        before = deepcopy(self.report)
        result = compare_performance(self.report, self.report, expected)
        self.assertFalse(result["comparable"])
        self.assertIn("当前轮存档", result["reason"])
        self.assertEqual(result["mismatch_fields"], ["expected_context.task_sha256"])
        self.assertEqual(self.report, before)
        self.assertFalse(compare_performance(self.report, self.report, {})["comparable"])
        self.assertFalse(compare_performance(self.report, self.report, "bad-context")["comparable"])

    def test_nonobject_reports_fail_closed(self):
        for value in (None, [], 1, "report"):
            with self.subTest(value=value):
                self.assertIsNone(performance_signature(value)[0])
                self.assertFalse(compare_performance(value, self.report)["comparable"])
                result = compare_performance(self.report, value)
                self.assertFalse(result["comparable"])
                self.assertEqual(result["mismatch_fields"], ["current.report"])


if __name__ == "__main__":
    unittest.main()

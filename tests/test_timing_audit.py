"""Offline synthetic CSV tests; no NPU or cann-bench dependency."""

import csv
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from lib.timing_audit import audit_kernel_csv


class TimingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "kernel_details.csv"

    def write(self, rows, fields=None):
        fields = fields or ["Name", "Type", "Start Time(us)", "Duration(us)", "Device_id", "Stream ID", "Task ID"]
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return self.path

    def row(self, name, start, duration, kind="AI_VECTOR_CORE", device="0", stream="1"):
        return {"Name": name, "Type": kind, "Start Time(us)": str(start),
                "Duration(us)": str(duration), "Device_id": device,
                "Stream ID": stream, "Task ID": "1"}

    def groups(self, patterns, gap=0):
        rows = []
        for repeat, pattern in enumerate(patterns):
            start = Decimal(repeat * 1000)
            rows.append(self.row("clean", start, 1, kind="_Z19CannBenchCacheCleanIDhEv"))
            start += 1
            for name, duration in pattern:
                rows.append(self.row(name, start, duration))
                start += Decimal(str(duration)) + Decimal(str(gap))
        return rows

    def audit(self, rows, expected=3, fields=None):
        result = audit_kernel_csv(self.write(rows, fields), expected)
        json.dumps(result, allow_nan=False)
        self.assertTrue(result["diagnostic_only"])
        self.assertNotIn("performance_pass", result)
        self.assertNotIn("corrected_speedup", result)
        return result

    def assert_invalid(self, rows, message=None, expected=3, fields=None):
        result = self.audit(rows, expected, fields)
        self.assertFalse(result["valid"], result)
        self.assertNotIn("metrics", result)
        self.assertNotIn("tool_compatible", result)
        if message:
            self.assertIn(message, " ".join(result["errors"]))
        return result

    def test_two_equal_kernels_are_summed(self):
        result = self.audit(self.groups([[('K1', 40), ('K2', 40)]] * 3))
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["metrics"]["device_kernel_sum_us"], {"mean": 80, "median": 80})
        self.assertEqual(result["metrics"]["device_activity_span_us"]["median"], 80)
        self.assertEqual(result["tool_compatible"]["sum_of_per_kernel_medians_us"], 80)
        self.assertFalse(result["topology_changed_across_repeats"])

    def test_slow_kernel_is_not_filtered(self):
        result = self.audit(self.groups([[('K1', 100), ('K2', 10)]] * 3))
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["median"], 110)

    def test_same_name_multiple_launches_are_summed(self):
        result = self.audit(self.groups([[('triton_same', 40), ('triton_same', 40)]] * 3))
        self.assertEqual(result["tool_compatible"]["per_kernel_medians_us"], {"triton_same": 80})
        self.assertEqual(result["repeats"][0]["kernel_counts"], {"triton_same": 2})

    def test_kernel_sum_and_device_span_are_distinct(self):
        result = self.audit(self.groups([[('K1', 40), ('K2', 40)]] * 3, gap=10))
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["mean"], 80)
        self.assertEqual(result["metrics"]["device_activity_span_us"]["mean"], 90)

    def test_mean_and_median_are_distinct(self):
        result = self.audit(self.groups([[('K', 10)], [('K', 20)], [('K', 90)]]))
        self.assertEqual(result["metrics"]["device_kernel_sum_us"], {"mean": 40, "median": 20})

    def test_topology_change_is_diagnosed(self):
        result = self.audit(self.groups([[('K', 40), ('K', 40)], [('K', 40), ('K', 40)], [('K', 40)]]))
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["topology_changed_across_repeats"])
        self.assertEqual([r["kernel_count"] for r in result["repeats"]], [2, 2, 1])

    def test_tool_compatible_medians_only_use_present_groups(self):
        result = self.audit(self.groups([[('A', 1), ('B', 100)], [('A', 100)], [('B', 1)]]))
        self.assertEqual(result["tool_compatible"]["per_kernel_medians_us"], {"A": 50.5, "B": 50.5})
        self.assertEqual(result["tool_compatible"]["sum_of_per_kernel_medians_us"], 101)
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["median"], 100)

    def test_large_timestamp_preserves_submicrosecond_span(self):
        rows = [self.row("CannBenchCacheClean", "12345678901234567890.001", ".001", kind="tool"),
                self.row("triton_A", "12345678901234567890.002", ".003"),
                self.row("triton_B", "12345678901234567890.005", ".004")]
        result = self.audit(rows, expected=1)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["metrics"]["device_activity_span_us"]["median"], .007)
        self.assertEqual(result["repeats"][0]["first_start_us_decimal"], "12345678901234567890.002")

    def test_only_tool_compatible_output_rounds_to_two_decimals(self):
        result = self.audit(self.groups([[('A', '.123'), ('B', '.456')]] * 3))
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["median"], .579)
        self.assertEqual(result["tool_compatible"]["sum_of_per_kernel_medians_us"], .58)
        self.assertEqual(result["tool_compatible"]["per_kernel_medians_us"], {"A": .12, "B": .46})

    def test_all_kernel_work_is_included_without_name_heuristics(self):
        rows = self.groups([[('K1', 40), ('K2', 10)]] * 3)
        for row in rows:
            if row["Name"] == "K2":
                row["Type"] = "aclnnCopy"
        result = self.audit(rows)
        self.assertNotIn("extra_non_pypto_work", result)
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["median"], 50)
        self.assertNotIn("pypto_kernel_sum_us", result["metrics"])
        self.assertEqual(result["kernel_counts_all_repeats"], {"K1": 3, "K2": 3})

    def test_tool_name_or_type_can_supply_boundary(self):
        rows = self.groups([[('K1', 40)]] * 3)
        for row in rows:
            if row["Name"] == "clean":
                row["Name"], row["Type"] = "_ZCannBenchWarmupKernel", "other"
        self.assertTrue(self.audit(rows)["valid"])

    def test_task_ids_cannot_replace_boundaries(self):
        rows = [self.row("K", i * 100, 40) for i in range(3)]
        self.assert_invalid(rows, "no preceding dedicated tool boundary")

    def test_incomplete_repeat_count_rejected(self):
        for repeats in (1, 2, 4):
            with self.subTest(repeats=repeats):
                self.assert_invalid(self.groups([[('K', 40)]] * repeats), "expected 3")

    def test_empty_groups_rejected(self):
        rows = self.groups([[('K', 40)]] * 3)
        self.assert_invalid(rows + [self.row("CannBenchCacheClean", 4000, 1)], "empty invocation")
        self.assert_invalid([rows[0], self.row("CannBenchCacheClean", 1, 1)] + rows[1:], "empty invocation")

    def test_invalid_duration_is_never_silently_skipped(self):
        for duration in ("", "bad", "0", "-1", "NaN", "sNaN", "Infinity", "1e9999"):
            with self.subTest(duration=duration):
                rows = self.groups([[('K', 40)]] * 3)
                rows[1]["Duration(us)"] = duration
                self.assert_invalid(rows)

    def test_invalid_timestamp_rejected(self):
        for start in ("", "bad", "-1", "NaN", "Infinity"):
            with self.subTest(start=start):
                rows = self.groups([[('K', 40)]] * 3)
                rows[1]["Start Time(us)"] = start
                self.assert_invalid(rows)

    def test_missing_device_identity_is_explicit_limitation(self):
        result = self.audit(self.groups([[('K', 40)]] * 3), fields=["Name", "Type", "Start Time(us)", "Duration(us)"])
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["device_identity_complete"])
        self.assertIn("Device identity is missing", " ".join(result["limitations"]))

    def test_multiple_devices_are_rejected(self):
        rows = self.groups([[('K', 40)]] * 3)
        rows[3]["Device_id"] = "1"
        self.assert_invalid(rows, "multiple devices")

    def test_overlap_and_multiple_streams_are_not_called_e2e(self):
        rows = self.groups([[('K1', 40), ('K2', 40)]] * 3)
        for index in (2, 5, 8):
            rows[index]["Start Time(us)"] = rows[index - 1]["Start Time(us)"]
            rows[index]["Stream ID"] = "2"
        result = self.audit(rows)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["metrics"]["device_kernel_sum_us"]["median"], 80)
        self.assertEqual(result["metrics"]["device_activity_span_us"]["median"], 40)
        self.assertTrue(all(r["overlapping_device_events"] for r in result["repeats"]))
        self.assertIn("Multiple streams", " ".join(result["limitations"]))

    def test_work_overlapping_boundary_is_rejected(self):
        rows = self.groups([[('K', 40)]] * 3)
        rows[1]["Start Time(us)"] = "0.5"
        self.assert_invalid(rows, "overlaps its tool boundary")
        rows = self.groups([[('K', 40)]] * 3)
        rows[2]["Start Time(us)"] = "20"
        self.assert_invalid(rows, "overlaps the previous invocation")

    def test_malformed_header_and_row_rejected(self):
        for text in ("", "Name,Duration(us)\nK,1\n", "Name,Name,Start Time(us),Duration(us)\nK,K,1,1\n",
                     "Name,Type,Start Time(us),Duration(us)\nK,AI_VECTOR_CORE,1\n",
                     "Name,Type,Start Time(us),Duration(us)\nK,AI_VECTOR_CORE,1,1,extra\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                result = audit_kernel_csv(self.path)
                self.assertFalse(result["valid"])
                self.assertNotIn("metrics", result)

    def test_missing_file_and_invalid_protocol(self):
        self.assertFalse(audit_kernel_csv(self.path)["valid"])
        for expected in (0, -1, True, 3.0, "3"):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                audit_kernel_csv(self.path, expected)

    def test_csv_is_not_modified(self):
        self.write(self.groups([[('K', 40)]] * 3))
        before = self.path.read_bytes()
        self.assertTrue(audit_kernel_csv(self.path)["valid"])
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

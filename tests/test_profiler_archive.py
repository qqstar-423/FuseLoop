"""Synthetic regression checks for report-bound, per-iteration profiler evidence."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import patch

from lib.bench_parser import parse_cannbench_report_to_perf_result, run_perf_eval
from lib.profiler_archive import archive_profiler_data
import test_semantic_routing as semantic_routing_tests


def write_report(path, case_ids, speeds=None):
    speeds = speeds or [1.5] * len(case_ids)
    cases = [
        {"case_id": case_id, "status": "success", "baseline_perf_us": 100,
         "elapsed_us": 100 / speed, "speedup": speed, "perf_score": 0.4,
         "t_hw_us": 3.25}
        for case_id, speed in zip(case_ids, speeds)
    ]
    payload = {"operators": [{
        "score": 90, "performance_score": 12.75,
        "avg_speedup": sum(speeds) / len(speeds),
        "total_cases": len(cases), "passed_cases": len(cases), "failed_cases": 0,
        "cases": cases,
    }]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_csv(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Name,Duration(us)\n" + label + ",50\n", encoding="utf-8")
    return path


class ProfilerArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cli_writes_and_searches_the_same_explicit_report_directory(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                source = self.root / str(custom) / "cann-bench/src"
                source.mkdir(parents=True)
                expected = (self.root / "custom_reports" if custom else source.parent / "reports")
                launched = []

                def transport(command, **kwargs):
                    launched.append(command)
                    directory = Path(command[command.index("--reports-dir") + 1])
                    self.assertEqual(directory, expected)
                    report = write_report(directory / "mixedcase_eval_001.json", ["level3/op_1"])
                    stamp = time.time() + 1
                    os.utime(report, (stamp, stamp))
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch("lib.bench_parser._build_eval_env", return_value={"PYTHONPATH": ""}), \
                        patch("lib.bench_parser.subprocess.run", side_effect=transport):
                    success, _, report = run_perf_eval(
                        str(self.root / "MixedCase"), str(source),
                        reports_dir=str(expected) if custom else None)
                self.assertTrue(success)
                self.assertEqual(Path(report), expected / "mixedcase_eval_001.json")
                self.assertEqual(len(launched), 1)

    def test_report_parent_is_the_only_source_for_repo_and_src_layouts(self):
        for report_parent in ("cann-bench/reports", "cann-bench/src/reports"):
            with self.subTest(report_parent=report_parent):
                area = self.root / ("src_layout" if "/src/" in report_parent else "repo_layout")
                report = write_report(area / report_parent / "alias_eval.json",
                                      ["level3/renamed_operator_1"])
                source = report.parent / "prof_data"
                write_csv(source / "level3/renamed_operator/1/session/kernel_details.csv",
                          "selected_report")
                other_parent = ("cann-bench/reports" if "/src/" in report_parent
                                else "cann-bench/src/reports")
                write_csv(area / other_parent / "prof_data/level3/renamed_operator/1/session/kernel_details.csv",
                          "unrelated_report")
                evaluation = area / "work/eval/iter1"
                logs = []
                copied = archive_profiler_data(str(report), str(evaluation), log_callback=logs.append)
                self.assertEqual(Path(copied), evaluation / "prof_data")
                expected = Path(copied) / "level3/renamed_operator/1/session/kernel_details.csv"
                self.assertIn("selected_report", expected.read_text(encoding="utf-8"))
                self.assertNotIn("unrelated_report", expected.read_text(encoding="utf-8"))
                messages = "\n".join(logs)
                self.assertIn(str(source), messages)
                self.assertIn(str(evaluation / "prof_data"), messages)

    def test_missing_source_does_not_reuse_a_previous_destination(self):
        report = write_report(self.root / "reports/new_eval.json", ["level3/op_1"])
        evaluation = self.root / "work/eval/iter1"
        write_csv(evaluation / "prof_data/level3/op/1/old/kernel_details.csv", "old_iteration")
        # A plausible cann-bench fallback must not be selected either.
        write_csv(self.root / "cann-bench/reports/prof_data/level3/op/1/kernel_details.csv",
                  "unrelated_cached_report")
        logs = []
        copied = archive_profiler_data(str(report), str(evaluation), log_callback=logs.append)
        self.assertEqual(copied, "")
        self.assertFalse((evaluation / "prof_data").exists())
        perf = parse_cannbench_report_to_perf_result(str(report), copied)
        self.assertEqual(perf["source_csv_dir"], "")
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")
        self.assertEqual(perf["worst_6_cases"][0]["kernel_csv"], "")
        self.assertIn(str(report.parent / "prof_data"), "\n".join(logs))
        self.assertTrue(logs, "Missing profiler data must be visible in the logs")

    def test_replacing_destination_removes_old_case_files(self):
        report = write_report(self.root / "reports/current_eval.json", ["level3/op_1"])
        write_csv(report.parent / "prof_data/level3/op/1/current/kernel_details.csv", "current")
        evaluation = self.root / "work/eval/iter1"
        old = write_csv(evaluation / "prof_data/level3/op/2/old/kernel_details.csv", "old")
        copied = archive_profiler_data(str(report), str(evaluation))
        self.assertFalse(old.exists())
        self.assertTrue((Path(copied) / "level3/op/1/current/kernel_details.csv").is_file())

    def test_nested_case_ids_and_aliases_select_their_own_files(self):
        case_ids = ["level3/fused_alias/my_operator_1", "level3/another_alias/my_operator_1"]
        report = write_report(self.root / "reports/eval.json", case_ids, [1.1, 2.0])
        source = self.root / "prof_data"
        paths = [
            write_csv(source / "level3/fused_alias/my_operator/1/run/kernel_details.csv", "first_alias"),
            write_csv(source / "level3/another_alias/my_operator/1/run/kernel_details.csv", "second_alias"),
        ]
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        expected = dict(zip(case_ids, paths))
        for key in ("cases", "worst_6_cases"):
            for case in perf[key]:
                self.assertEqual(Path(case["kernel_csv"]), expected[case["case_id"]])
                self.assertTrue(Path(case["kernel_csv"]).is_file())
        self.assertEqual(perf["cases"], perf["worst_6_cases"])
        self.assertEqual(perf["avg_speedup"], 1.55)

    def test_historical_flat_operator_directory_remains_supported(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = self.root / "operator_prof_data"
        csv = write_csv(source / "7/run/kernel_details.csv", "legacy_flat")
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        self.assertEqual(Path(perf["cases"][0]["kernel_csv"]), csv)

    def test_full_profiler_root_does_not_match_an_unrelated_flat_case(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = self.root / "prof_data"
        write_csv(source / "7/kernel_details.csv", "unrelated_flat_case")
        write_csv(source / "level3/other_operator/7/kernel_details.csv", "other_operator")
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")

    def test_ambiguous_csvs_do_not_choose_a_file_or_use_flat_fallback(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = self.root / "prof_data"
        write_csv(source / "level3/my_operator/7/run_a/kernel_details.csv", "run_a")
        write_csv(source / "level3/my_operator/7/run_b/kernel_details.csv", "run_b")
        write_csv(source / "7/kernel_details.csv", "misleading_flat_fallback")
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")
        self.assertEqual(perf["worst_6_cases"][0]["kernel_csv"], "")

    def test_exact_empty_case_directory_does_not_use_an_unrelated_flat_case(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = self.root / "prof_data"
        (source / "level3/my_operator/7").mkdir(parents=True)
        write_csv(source / "7/kernel_details.csv", "unrelated_flat_case")
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")

    def test_batched_profiler_is_preserved_without_fabricating_case_links(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = report.parent / "prof_data"
        batch = write_csv(source / "_batched/session/kernel_details.csv", "batch")
        evaluation = self.root / "work/eval/iter1"
        copied = archive_profiler_data(str(report), str(evaluation))
        self.assertEqual((Path(copied) / batch.relative_to(source)).read_bytes(), batch.read_bytes())
        perf = parse_cannbench_report_to_perf_result(str(report), copied)
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")
        self.assertEqual(perf["worst_6_cases"][0]["kernel_csv"], "")

    def test_a_directory_named_kernel_details_csv_is_not_a_csv_file(self):
        report = write_report(self.root / "reports/eval.json", ["level3/my_operator_7"])
        source = self.root / "prof_data"
        (source / "level3/my_operator/7/kernel_details.csv").mkdir(parents=True)
        perf = parse_cannbench_report_to_perf_result(str(report), str(source))
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")


class Stage6ProfilerArchiveRoutingTests(unittest.TestCase):
    def test_retry_does_not_reuse_profiler_from_a_failed_attempt(self):
        flow = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        flow.setUp()
        self.addCleanup(flow.doCleanups)
        flow.perfs = {1: (1.5,)}
        attempts = []

        def performance(*_args, **kwargs):
            directory = Path(kwargs["reports_dir"])
            attempts.append(directory)
            self.assertEqual(directory.parent, flow.work / "eval/iter1/perf_reports")
            self.assertTrue(directory.name.startswith(f"attempt_{len(attempts)}_"))
            if len(attempts) == 1:
                write_csv(directory / "prof_data/synthetic_op/1/run/kernel_details.csv",
                          "failed_attempt")
                return False, "synthetic evaluator crash", ""
            report = write_report(directory / "synthetic_op_eval.json", ["synthetic_op_1"])
            return True, "", str(report)

        flow.fixture.performance = performance
        with patch("time.sleep"):
            flow.run_workflow(max_iterations=1, file_logs=True)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0], attempts[1])
        perf = json.loads((flow.work / "eval/iter1/perf_result.json").read_text(encoding="utf-8"))
        self.assertEqual(perf["source_csv_dir"], "")
        self.assertEqual(perf["cases"][0]["kernel_csv"], "")
        self.assertEqual(perf["worst_6_cases"][0]["kernel_csv"], "")
        logs = (flow.work / "log/workflow.log").read_text(encoding="utf-8")
        self.assertIn("0/1", logs)
        self.assertIn("synthetic_op_1", logs)

    def test_two_rounds_keep_work_local_csv_links_and_best_snapshot(self):
        flow = semantic_routing_tests.SemanticRoutingTests(methodName="runTest")
        flow.setUp()
        self.addCleanup(flow.doCleanups)
        flow.perfs = {1: (1.8, 1.5, 1.6), 2: (1.79, 1.5, 1.6)}
        case_ids = [
            "level3/fused_alias/synthetic_op_1",
            "level3/another_alias/synthetic_op_1",
            "level3/fused_alias/synthetic_op_2",
        ]
        reports = []

        def performance(*_args, **kwargs):
            iteration = flow.fixture.read_state()["iteration"]
            flow.fixture.events.append("stage6")
            self.assertIn("reports_dir", kwargs, "Stage6 must explicitly isolate each evaluator attempt")
            reports_dir = Path(kwargs["reports_dir"])
            self.assertEqual(reports_dir.parent, flow.work / f"eval/iter{iteration}/perf_reports")
            self.assertTrue(reports_dir.name.startswith("attempt_1_"))
            self.assertNotIn(reports_dir, [path.parent for path in reports])
            report = write_report(reports_dir / "synthetic_op_eval.json", case_ids, flow.perfs[iteration])
            for case_id in case_ids[:2]:
                relative_operator, number = case_id.rsplit("_", 1)
                write_csv(reports_dir / "prof_data" / relative_operator / number / "run/kernel_details.csv",
                          f"iteration_{iteration}_{relative_operator}")
            write_csv(reports_dir / "prof_data/_batched/run/kernel_details.csv", f"batch_{iteration}")
            reports.append(report)
            return True, "", str(report)

        flow.fixture.performance = performance
        flow.run_workflow(max_iterations=2, file_logs=True)
        self.assertEqual(len(reports), 2)
        for iteration in (1, 2):
            evaluation = flow.work / f"eval/iter{iteration}"
            perf = json.loads((evaluation / "perf_result.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(perf["source_csv_dir"]), evaluation / "prof_data")
            by_id = {case["case_id"]: case for case in perf["cases"]}
            self.assertEqual(len(by_id), 3)
            self.assertEqual(by_id[case_ids[2]]["kernel_csv"], "")
            for case in perf["worst_6_cases"]:
                self.assertEqual(case["kernel_csv"], by_id[case["case_id"]]["kernel_csv"])
            for case_id in case_ids[:2]:
                csv = Path(by_id[case_id]["kernel_csv"])
                self.assertTrue(csv.is_relative_to(evaluation / "prof_data"))
                self.assertIn(f"iteration_{iteration}_", csv.read_text(encoding="utf-8"))
            self.assertTrue((evaluation / "prof_data/_batched/run/kernel_details.csv").is_file())

        best = flow.selection_status()["best"]
        self.assertEqual(best["iteration"], 1)
        snapshot = json.loads(Path(best["performance_result"]).read_text(encoding="utf-8"))
        best_csvs = [Path(case["kernel_csv"]) for case in snapshot["cases"] if case["kernel_csv"]]
        self.assertEqual(len(best_csvs), 2)
        self.assertTrue(all(path.is_relative_to(flow.work / "selection") for path in best_csvs))
        original_best_bytes = {path: path.read_bytes() for path in best_csvs}
        # Mutating both raw evaluator runs must not change either the first
        # iteration evidence or the independently selected best implementation.
        for report in reports:
            for source in (report.parent / "prof_data").rglob("kernel_details.csv"):
                source.write_text("overwritten raw profiler", encoding="utf-8")
        first = json.loads((flow.work / "eval/iter1/perf_result.json").read_text(encoding="utf-8"))
        for case in first["cases"]:
            if case["kernel_csv"]:
                self.assertIn("iteration_1_", Path(case["kernel_csv"]).read_text(encoding="utf-8"))
        for path, content in original_best_bytes.items():
            self.assertEqual(path.read_bytes(), content)

        logs = (flow.work / "log/workflow.log").read_text(encoding="utf-8")
        normalized_logs = logs.replace("\\", "/")
        for iteration, report in enumerate(reports, 1):
            self.assertIn(str(report.parent / "prof_data").replace("\\", "/"), normalized_logs)
            self.assertIn(str(flow.work / f"eval/iter{iteration}/prof_data").replace("\\", "/"),
                          normalized_logs)
        self.assertIn("2/3", logs)
        self.assertIn(case_ids[2], logs)


if __name__ == "__main__":
    unittest.main()

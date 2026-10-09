"""One report per iteration, explicit provenance and safe atomic publication."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.profile_report import (profile_report_path, stamp_stage7_report,
                                validate_profile_report_directory, write_stage9_report)


class ProfileReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.decision = {
            "iteration": 3,
            "ledger_entry": {
                "evaluation_summary": "case7's measured time is 12us; no per-core data yet.",
                "case_analysis": [{
                    "case_id": "case7", "observation": "measured time is high.",
                    "explanation": "speculated uneven tiling; still to be verified.",
                    "evidence": "eval/iter3/perf_result.json; no kernel_csv.",
                    "next_action": "recheck the dispatcher and tiling strategy.",
                }],
            },
            "suggest_next": [{
                "task_id": "T1", "priority": "P1", "action": "check case7's tiling.",
                "reason": "per-core workload may be uneven.", "acceptance_checks": ["run case7 correctness and performance tests."],
                "changes": [{"file": "impl/case7.py", "operation": "inspect"}],
            }],
        }
        self.source = self.work / "stage9/iter3/request1/decision.json"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(json.dumps(self.decision, ensure_ascii=False), encoding="utf-8")

    def stage7(self, body="# Stage7 original\n\nkeep conclusions, evidence and blank lines.\n\n"):
        path = profile_report_path(self.work, 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def write_stage9(self):
        return write_stage9_report(self.work, 3, self.decision, self.source)

    def symlink(self, path, destination, directory=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(destination, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks are not available: {exc}")

    def test_path_is_canonical_and_does_not_create_directories(self):
        report = profile_report_path(self.work, 3)
        self.assertEqual(report, self.work / "profile/iter3/bottleneck_analysis.md")
        self.assertEqual(validate_profile_report_directory(self.work, 3), report)
        self.assertFalse(report.parent.exists())
        for iteration in (-1, True, "3", "../other"):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                profile_report_path(self.work, iteration)

    def test_stage7_body_is_preserved_and_stamping_is_idempotent(self):
        body = "# original\n\nSymptom: 12us.\n\nCause: to be verified.\n\n"
        report = self.stage7(body)
        self.assertEqual(stamp_stage7_report(self.work, 3), report)
        stamped = report.read_text(encoding="utf-8")
        self.assertTrue(stamped.endswith(body))
        self.assertIn("Analysis source: Stage7", stamped)
        self.assertIn("Iteration: 3", stamped)
        self.assertIn("not applicable (original Stage7 report)", stamped)
        with patch("lib.profile_report.atomic_write_text") as write:
            stamp_stage7_report(self.work, 3)
        write.assert_not_called()
        self.assertEqual(report.read_text(encoding="utf-8"), stamped)
        self.assertEqual(list(report.parent.iterdir()), [report])

    def test_stage7_replaces_own_source_block_only(self):
        report = self.stage7("# Original\n\n> Analysis source: user note\n")
        stamp_stage7_report(self.work, 3)
        stamped = report.read_text(encoding="utf-8").replace("Analysis source: Stage7", "Analysis source: OldStage")
        report.write_text(stamped, encoding="utf-8")
        stamp_stage7_report(self.work, 3)
        content = report.read_text(encoding="utf-8")
        self.assertEqual(content.count("profile-report-source:start"), 1)
        self.assertNotIn("OldStage", content)
        self.assertTrue(content.endswith("# Original\n\n> Analysis source: user note\n"))

    def test_stage7_rejects_missing_empty_and_source_only_reports(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            stamp_stage7_report(self.work, 3)
        for body in ("", " \n\t", "<!-- profile-report-source:start -->\n"
                     "> Analysis source: Stage7\n<!-- profile-report-source:end -->\n\n"):
            with self.subTest(body=body):
                report = self.stage7(body)
                before = report.read_bytes()
                with self.assertRaisesRegex(ValueError, "empty"):
                    stamp_stage7_report(self.work, 3)
                self.assertEqual(report.read_bytes(), before)

    def test_stage7_malformed_source_block_is_not_silently_removed(self):
        report = self.stage7("<!-- profile-report-source:start -->\n# Original\n")
        before = report.read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            stamp_stage7_report(self.work, 3)
        self.assertEqual(report.read_bytes(), before)

    def test_stage7_cannot_relabel_stage9_analysis(self):
        report = self.write_stage9()
        before = report.read_bytes()
        original_mtime = report.stat().st_mtime_ns
        with patch("lib.profile_report.atomic_write_text") as write:
            with self.assertRaisesRegex(ValueError, "Stage9 report to Stage7"):
                stamp_stage7_report(self.work, 3)
        write.assert_not_called()
        self.assertEqual(report.read_bytes(), before)
        self.assertEqual(report.stat().st_mtime_ns, original_mtime)

    def test_stage9_renders_supplied_findings_and_precise_provenance(self):
        original = deepcopy(self.decision)
        report = self.write_stage9()
        content = report.read_text(encoding="utf-8")
        self.assertIn("Analysis source: Stage9", content)
        self.assertIn("Iteration: 3", content)
        self.assertIn("stage9/iter3/request1/decision.json", content)
        self.assertIn(self.decision["ledger_entry"]["evaluation_summary"], content)
        for value in self.decision["ledger_entry"]["case_analysis"][0].values():
            self.assertIn(value, content)
        for label in ("Observation", "Cause", "Evidence", "Next step", "T1", "P1", "Acceptance"):
            self.assertIn(label, content)
        self.assertIn("the original profiling is not re-read when generating the report", content)
        self.assertIn("neither adds nor expands file modification permissions", content)
        self.assertIn("follow the validated original decision", content)
        self.assertNotIn('"changes"', content)
        self.assertEqual(self.decision, original)
        self.assertEqual(list(report.parent.iterdir()), [report])

    def test_stage9_replaces_same_report_and_is_repeatable(self):
        report = self.stage7("# Obsolete Stage7 body\n")
        stamp_stage7_report(self.work, 3)
        self.assertEqual(self.write_stage9(), report)
        first = report.read_bytes()
        self.assertNotIn(b"Obsolete", first)
        self.write_stage9()
        self.assertEqual(report.read_bytes(), first)
        self.decision["ledger_entry"]["evaluation_summary"] = "Summary after corrected review."
        second_source = self.source.parent.parent / "request2/decision.json"
        second_source.parent.mkdir()
        second_source.write_text(json.dumps(self.decision, ensure_ascii=False), encoding="utf-8")
        write_stage9_report(self.work, 3, self.decision, second_source)
        content = report.read_text(encoding="utf-8")
        self.assertIn("Summary after corrected review.", content)
        self.assertIn("request2/decision.json", content)
        self.assertNotIn("request1/decision.json", content)
        self.assertEqual(list(report.parent.iterdir()), [report])

    def test_other_iterations_and_non_markdown_evidence_are_untouched(self):
        other = self.work / "profile/iter2/bottleneck_analysis.md"
        other.parent.mkdir(parents=True)
        other.write_text("old iteration", encoding="utf-8")
        report = self.stage7()
        evidence = report.parent / "raw/metrics.csv"
        evidence.parent.mkdir()
        evidence.write_text("duration,12", encoding="utf-8")
        self.write_stage9()
        self.assertEqual(other.read_text(encoding="utf-8"), "old iteration")
        self.assertEqual(evidence.read_text(encoding="utf-8"), "duration,12")

    def test_rechecking_identical_stage9_report_is_read_only(self):
        report = self.write_stage9()
        original_bytes = report.read_bytes()
        original_mtime = report.stat().st_mtime_ns
        with patch("lib.profile_report.atomic_write_text") as write:
            self.assertEqual(self.write_stage9(), report)
        write.assert_not_called()
        self.assertEqual(report.read_bytes(), original_bytes)
        self.assertEqual(report.stat().st_mtime_ns, original_mtime)

    def test_competing_markdown_is_rejected_recursively_without_changes(self):
        for name in ("summary.md", "nested/extra.MARKDOWN", "nested/BOTTLENECK_ANALYSIS.MD"):
            with self.subTest(name=name):
                report = self.stage7()
                before = report.read_bytes()
                competitor = report.parent / name
                competitor.parent.mkdir(parents=True, exist_ok=True)
                competitor.write_text("competing report", encoding="utf-8")
                try:
                    for action in (lambda: validate_profile_report_directory(self.work, 3),
                                   lambda: stamp_stage7_report(self.work, 3), self.write_stage9):
                        with self.assertRaisesRegex(ValueError, "additional Markdown"):
                            action()
                    self.assertEqual(report.read_bytes(), before)
                    self.assertEqual(competitor.read_text(encoding="utf-8"), "competing report")
                finally:
                    competitor.unlink()

    def test_directory_at_report_path_is_rejected(self):
        report = profile_report_path(self.work, 3)
        report.mkdir(parents=True)
        for action in (lambda: stamp_stage7_report(self.work, 3), self.write_stage9):
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                action()
        self.assertTrue(report.is_dir())

    def test_malformed_decision_never_replaces_previous_report(self):
        report = self.stage7()
        before = report.read_bytes()
        for alter in (lambda decision: decision.update(iteration=2),
                      lambda decision: decision["ledger_entry"].update(evaluation_summary=""),
                      lambda decision: decision["ledger_entry"]["case_analysis"][0].update(evidence=""),
                      lambda decision: decision["suggest_next"][0].update(acceptance_checks=[None])):
            value = deepcopy(self.decision)
            alter(value)
            with self.assertRaises(ValueError):
                write_stage9_report(self.work, 3, value, self.source)
            self.assertEqual(report.read_bytes(), before)

    def test_atomic_write_failure_preserves_previous_report(self):
        report = self.stage7()
        before = report.read_bytes()
        with patch("lib.handoff.os.replace", side_effect=OSError("replacement failed")):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                self.write_stage9()
        self.assertEqual(report.read_bytes(), before)
        self.assertEqual(list(report.parent.iterdir()), [report])

    def test_decision_source_must_be_an_existing_file_inside_work(self):
        report = self.stage7()
        before = report.read_bytes()
        outside = self.root / "decision.json"
        outside.write_text("{}", encoding="utf-8")
        for source in (outside, self.source.parent, self.source.parent / "missing.json"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                write_stage9_report(self.work, 3, self.decision, source)
            self.assertEqual(report.read_bytes(), before)
        write_stage9_report(self.work, 3, self.decision, self.source.relative_to(self.work))

    def test_report_symlink_outside_work_is_rejected(self):
        outside = self.root / "outside.md"
        outside.write_text("do not change", encoding="utf-8")
        self.symlink(profile_report_path(self.work, 3), outside)
        for action in (lambda: stamp_stage7_report(self.work, 3), self.write_stage9):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(outside.read_text(encoding="utf-8"), "do not change")

    def test_iteration_directory_symlink_outside_work_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.symlink(self.work / "profile/iter3", outside, directory=True)
        with self.assertRaises(ValueError):
            self.write_stage9()
        self.assertEqual(list(outside.iterdir()), [])

    def test_nested_directory_symlink_outside_iteration_is_rejected(self):
        report = self.stage7()
        before = report.read_bytes()
        self.symlink(report.parent / "external", self.source.parent, directory=True)
        with self.assertRaises(ValueError):
            self.write_stage9()
        self.assertEqual(report.read_bytes(), before)

    def test_iteration_directory_cannot_alias_another_iteration(self):
        other_report = profile_report_path(self.work, 2)
        other_report.parent.mkdir(parents=True)
        other_report.write_text("# Iteration 2 report\n", encoding="utf-8")
        before = other_report.read_bytes()
        self.symlink(self.work / "profile/iter3", other_report.parent, directory=True)
        for action in (lambda: validate_profile_report_directory(self.work, 3),
                       lambda: stamp_stage7_report(self.work, 3), self.write_stage9):
            with self.assertRaisesRegex(ValueError, "must not alias"):
                action()
            self.assertEqual(other_report.read_bytes(), before)

    def test_profile_directory_cannot_alias_another_work_directory(self):
        other_directory = self.work / "selection"
        other_report = other_directory / "iter3/bottleneck_analysis.md"
        other_report.parent.mkdir(parents=True)
        other_report.write_text("# Selection evidence\n", encoding="utf-8")
        before = other_report.read_bytes()
        self.symlink(self.work / "profile", other_directory, directory=True)
        for action in (lambda: validate_profile_report_directory(self.work, 3),
                       lambda: stamp_stage7_report(self.work, 3), self.write_stage9):
            with self.assertRaisesRegex(ValueError, "must not alias"):
                action()
            self.assertEqual(other_report.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

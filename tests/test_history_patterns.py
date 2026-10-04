import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.history_manager import (
    append_pitfall, append_proven_pattern, append_regression_pattern,
    format_for_prompt, format_pitfalls_for_prompt,
    format_proven_patterns_for_prompt, format_regression_patterns_for_prompt,
    load_pitfalls, load_proven_patterns, load_regression_patterns, save_history,
)


class HistoryPatternTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        (self.work / "knowledge").mkdir()

    def environment(self):
        return {
            "framework": "Triton", "chip_model": "Ascend 910B4",
            "soc_version": "Ascend910B4", "npu_arch": "dav-c220",
            "programming_model": "SIMD", "device_id": 0,
            "source_kind": "performance_report",
            "source_path": "eval/iter3/perf_result.json",
        }

    def detailed_pattern(self, regression=False):
        why = "why_it_failed" if regression else "why_it_worked"
        return {
            "speedup_before": 2.0, "speedup_after": 1.8 if regression else 2.2,
            "delta_pct": -10.0 if regression else 10.0, "fusion_related": True,
            "what_changed": "adjusted tiling\n\nsecond line: keep the dataflow\n## iter999: this is body text\n- **Evidence**: also body\n",
            why: "first reason\n  indented continuation\nsecond reason",
            "case_analysis": [{"case_id": "case3", "observation": "0.683 → 0.977\nstill below target",
                               "explanation": "tail block improved; do not change schemes on the mean alone"}],
            "evidence": ["kernel_details.csv: memory traffic dropped", "multi-line evidence\nfully preserved"],
            "applicability": "same chip\nsmall shapes",
            "next_action": "keep optimizing the tail block\nconfirm all cases",
            "prev_iter": 2,
            "case_diffs": [{"case_id": "case3", "prev_speedup": 0.683, "curr_speedup": 0.977}],
            "evidence_files": {"performance": str(self.work / "eval/iter3/perf_result.json")},
            "decision_path": str(self.work / "review/iter3/decision.json"),
            "environment": self.environment(),
        }

    def test_all_details_and_multiline_values_round_trip(self):
        for regression, append, load, filename in (
            (False, append_proven_pattern, load_proven_patterns, "proven_patterns.md"),
            (True, append_regression_pattern, load_regression_patterns, "regression_patterns.md"),
        ):
            with self.subTest(regression=regression):
                value = self.detailed_pattern(regression)
                append(self.work, 3, value)
                self.assertEqual(load(self.work), [{"iter": 3, **value}])
                markdown = (self.work / "knowledge" / filename).read_text(encoding="utf-8")
                self.assertEqual(markdown.count("## iter3:"), 1)
                self.assertIn("- **Per-case analysis**:", markdown)
                self.assertIn("- **Framework and chip**:", markdown)
                self.assertEqual(markdown.count("tail block improved; do not change schemes on the mean alone"), 1)

    def test_string_list_case_analysis_round_trip(self):
        value = self.detailed_pattern()
        value["case_analysis"] = ["case1 improved", "case2\nstill needs analysis"]
        append_proven_pattern(self.work, 4, value)
        self.assertEqual(load_proven_patterns(self.work)[0]["case_analysis"], value["case_analysis"])

    def test_same_iteration_updates_without_losing_other_legacy_blocks(self):
        for regression, append, load, filename in (
            (False, append_proven_pattern, load_proven_patterns, "proven_patterns.md"),
            (True, append_regression_pattern, load_regression_patterns, "regression_patterns.md"),
        ):
            with self.subTest(regression=regression):
                sign, label = ("-", "Why it regressed") if regression else ("+", "Why it worked")
                old = (f"## iter1: speedup 1 → 1.1 ({sign}10%)\n\n"
                       f"- **What changed**: historical body\n- **{label}**: historical reason\n\n")
                duplicate = old.replace("iter1:", "iter3:")
                prefix = "# User's original heading\n\n> Keep this note.\n\n"
                path = self.work / "knowledge" / filename
                path.write_text(prefix + old + duplicate + duplicate, encoding="utf-8")
                value = self.detailed_pattern(regression)
                append(self.work, 3, value)
                append(self.work, 3, {**value, "what_changed": "second complete submission\nupdated"})
                content = path.read_text(encoding="utf-8")
                self.assertTrue(content.startswith(prefix + old))
                self.assertEqual(content.count("## iter3:"), 1)
                self.assertEqual([item["iter"] for item in load(self.work)], [1, 3])
                self.assertEqual(load(self.work)[1]["what_changed"], "second complete submission\nupdated")

    def test_old_short_records_and_unindented_multiline_remain_readable(self):
        (self.work / "knowledge/proven_patterns.md").write_text(
            "# old format\n\n## iter2: speedup 0.5 → 1.5 (+200%) 🔥fusion-related\n\n"
            "- **What changed**: old change\nsecond line of change\n- **Why it worked**: old reason\nsecond line of reason\n\n",
            encoding="utf-8")
        (self.work / "knowledge/regression_patterns.md").write_text(
            "## iter4: speedup 3 → 2.5 (-16.7%)\n\n"
            "- **What changed**: old regression\n- **Why it regressed**: insufficient capacity\n\n", encoding="utf-8")
        proven = load_proven_patterns(self.work)[0]
        self.assertEqual(proven["what_changed"], "old change\nsecond line of change")
        self.assertEqual(proven["why_it_worked"], "old reason\nsecond line of reason")
        self.assertTrue(proven["fusion_related"])
        self.assertEqual(load_regression_patterns(self.work)[0]["delta_pct"], -16.7)

    def test_prompts_preserve_all_history_and_complete_reasons_without_new_details(self):
        for regression, append, formatter, filename in (
            (False, append_proven_pattern, format_proven_patterns_for_prompt, "proven_patterns.md"),
            (True, append_regression_pattern, format_regression_patterns_for_prompt, "regression_patterns.md"),
        ):
            with self.subTest(regression=regression):
                value = self.detailed_pattern(regression)
                why = "why_it_failed" if regression else "why_it_worked"
                value.update(what_changed="change" * 1000 + "\nonly applies to specific shapes",
                             **{why: "reason" * 1000 + "\nthe trailing restriction must be preserved"})
                for iteration in range(1, 14):
                    append(self.work, iteration, value)
                prompt = formatter(self.work)
                self.assertIn("iter1:", prompt)
                self.assertIn("iter13:", prompt)
                self.assertIn(value["what_changed"], prompt)
                self.assertIn(value[why], prompt)
                self.assertIn(value["applicability"], prompt)
                if regression:
                    self.assertIn("re-verify when conditions change", prompt)
                    self.assertNotIn("these directions have been proven harmful", prompt)
                self.assertIn(filename, prompt)
                self.assertNotIn("tail block improved; do not change schemes on the mean alone", prompt)
                self.assertEqual(len((load_regression_patterns if regression else load_proven_patterns)(self.work)), 13)
        combined = format_for_prompt(self.work)
        self.assertIn("iter1:", combined)
        self.assertIn("iter13:", combined)
        self.assertIn("change" * 1000 + "\nonly applies to specific shapes", combined)
        self.assertIn("reason" * 1000 + "\nthe trailing restriction must be preserved", combined)
        self.assertNotIn("tail block improved; do not change schemes on the mean alone", combined)
        self.assertIn("proven_patterns.md", combined)
        self.assertIn("regression_patterns.md", combined)
        self.assertIn("Applicability: same chip\nsmall shapes", combined)
        self.assertIn("re-verify when conditions change", combined)
        self.assertNotIn("these directions have been proven harmful", combined)

    def test_pattern_prompts_show_recorded_environment_in_each_entry(self):
        old_environment = {**self.environment(), "chip_model": "Ascend 910B1",
                           "soc_version": "Ascend910B1", "programming_model": "SIMT"}
        for regression, append, formatter in (
            (False, append_proven_pattern, format_proven_patterns_for_prompt),
            (True, append_regression_pattern, format_regression_patterns_for_prompt),
        ):
            with self.subTest(regression=regression):
                append(self.work, 1, {**self.detailed_pattern(regression), "environment": old_environment})
                append(self.work, 3, self.detailed_pattern(regression))
                prompt = formatter(self.work)
                for text in ("Triton", "Ascend 910B1", "Ascend910B1", "SIMT",
                             "Ascend 910B4", "Ascend910B4", "SIMD", "First verify the experience's framework and chip"):
                    self.assertIn(text, prompt)
        combined = format_for_prompt(self.work)
        self.assertEqual(combined.count("Triton"), 4)
        for text in ("Ascend 910B1", "Ascend910B1", "SIMT",
                     "Ascend 910B4", "Ascend910B4", "SIMD"):
            self.assertEqual(combined.count(text), 2)

    def test_legacy_environment_is_not_inferred_or_written_from_current_device(self):
        (self.work / "device_info.json").write_text(
            json.dumps({"chip_model": "invented-current-chip", "framework": "invented-framework"}),
            encoding="utf-8")
        for regression, append, load, formatter, filename in (
            (False, append_proven_pattern, load_proven_patterns,
             format_proven_patterns_for_prompt, "proven_patterns.md"),
            (True, append_regression_pattern, load_regression_patterns,
             format_regression_patterns_for_prompt, "regression_patterns.md"),
        ):
            with self.subTest(regression=regression):
                value = self.detailed_pattern(regression)
                del value["environment"]
                append(self.work, 2, value)
                path = self.work / "knowledge" / filename
                before = path.read_bytes()
                self.assertNotIn("environment", load(self.work)[0])
                prompt = formatter(self.work)
                self.assertIn("not recorded", prompt)
                self.assertNotIn("invented-current-chip", prompt)
                self.assertNotIn("invented-framework", prompt)
                self.assertEqual(path.read_bytes(), before)
        combined = format_for_prompt(self.work)
        self.assertIn("not recorded", combined)
        self.assertNotIn("invented-current-chip", combined)
        self.assertNotIn("Triton", combined)

    def test_pitfall_environment_and_decision_path_round_trip_and_prompt(self):
        value = {"topic": "memory constraint", "verdict": "confirmed", "target_advice": "original advice",
                 "feedback": "measured feedback", "root_cause": "architecture limit", "correct_approach": "rearrange memory",
                 "question_path": "develop/iter2/question.md",
                 "decision_path": "review/iter3/decision.json", "environment": self.environment()}
        append_pitfall(self.work, 3, value)
        self.assertEqual(load_pitfalls(self.work), [{"iter": 3, **value}])
        text = (self.work / "knowledge/tech_lead_pitfalls.md").read_text(encoding="utf-8")
        self.assertIn("- **Framework and chip**:", text)
        self.assertIn("- **This round's decision file**: review/iter3/decision.json", text)
        prompt = format_pitfalls_for_prompt(self.work)
        for text in ("Triton", "Ascend 910B4", "Ascend910B4", "SIMD", "First verify the experience's framework and chip"):
            self.assertIn(text, prompt)

    def test_pitfall_retries_identify_question_and_preserve_other_rulings(self):
        question = "develop/iter2/question.md"
        value = {"topic": "sync limit\nsupplementary topic", "verdict": "confirmed", "target_advice": "old advice\nconcrete line numbers",
                 "feedback": "measurement did not hold\nwith log", "root_cause": "wrong assumption\nevidence",
                 "correct_approach": "verify the hardware\ndecide afterwards", "question_path": question}
        append_pitfall(self.work, 3, value)
        append_pitfall(self.work, 3, {**value, "topic": "updated ruling", "verdict": "rejected",
                                   "question_path": str(self.work / question)})
        append_pitfall(self.work, 3, {**value, "question_path": "develop/iter1/question.md"})
        append_pitfall(self.work, 4, value)
        records = load_pitfalls(self.work)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["topic"], "updated ruling")
        self.assertEqual(records[0]["verdict"], "rejected")
        self.assertEqual(records[0]["feedback"], value["feedback"])
        self.assertEqual(records[1]["topic"], value["topic"])
        self.assertEqual(records[2], {"iter": 4, **value})

    def test_legacy_pitfalls_read_and_new_question_does_not_erase_them(self):
        legacy = ("# pitfall log\n\n## iter1 ✅ misjudgment confirmed: old topic\n\n"
                  "- **Involved advice**: old advice\n- **cannbot feedback**: old feedback\n"
                  "- **Misjudgment confirmation reason**: old reason\n- **Correct practice**: old practice\n\n")
        path = self.work / "knowledge/tech_lead_pitfalls.md"
        path.write_text(legacy, encoding="utf-8")
        self.assertEqual(load_pitfalls(self.work)[0]["root_cause"], "old reason")
        self.assertNotIn("environment", load_pitfalls(self.work)[0])
        self.assertNotIn("decision_path", load_pitfalls(self.work)[0])
        prompt = format_pitfalls_for_prompt(self.work)
        self.assertIn("not recorded", prompt)
        self.assertNotIn("Triton", prompt)
        self.assertEqual(path.read_text(encoding="utf-8"), legacy)
        append_pitfall(self.work, 1, {"topic": "new question", "question_path": "develop/iter0/question.md"})
        self.assertTrue(path.read_text(encoding="utf-8").startswith(legacy))
        self.assertEqual(len(load_pitfalls(self.work)), 2)

    def test_history_write_failure_keeps_previous_json_and_cleans_temp_file(self):
        save_history(self.work, {"insights": ["saved"]})
        path = self.work / "knowledge/history.json"
        with patch("lib.handoff.os.replace", side_effect=OSError("synthetic replacement failure")):
            with self.assertRaises(OSError):
                save_history(self.work, {"insights": ["incomplete new value"]})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"insights": ["saved"]})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_markdown_write_failure_keeps_previous_records(self):
        append_proven_pattern(self.work, 1, self.detailed_pattern())
        path = self.work / "knowledge/proven_patterns.md"
        before = path.read_text(encoding="utf-8")
        with patch("lib.handoff.os.replace", side_effect=OSError("synthetic replacement failure")):
            with self.assertRaises(OSError):
                append_proven_pattern(self.work, 1, {"what_changed": "history must not be lost"})
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

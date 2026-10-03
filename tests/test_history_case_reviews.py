"""Current case coverage, per-round provenance and incremental long-term memory."""

from copy import deepcopy
import unittest

from lib.tech_lead import merge_tech_lead_update, validate_performance_pattern_details


class HistoryCaseReviewTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"operator_{index}" for index in range(1, 7)]
        self.base = {
            "rounds": [{"iter": 2, "avg_speedup": 1.4}],
            "ledger": [{"iter": 1, "direction": "old plan", "case_analysis": [{"old": True}]},
                       {"iter": 2, "avg_speedup_after": 1.4, "verdict": "no_change"}],
            "worst_cases_tracker": {"operator_7": "Earlier slow case, no new evidence"},
        }
        self.output = {
            "iteration": 2, "request_id": "current",
            "ledger_entry": {
                "evaluation_summary": "Smaller tiles improved the tail case; mean barely changed",
                "direction": "Continue testing the tail without altering proven paths",
                "modify_files": ["impl/kernel.py"], "readonly_files": [],
                "case_analysis": [{
                    "case_id": case_id, "observation": "Measured latency in current report",
                    "explanation": "Hypothesis: padding dominates, not yet proven",
                    "evidence": f"eval/iter2/perf_result.json cases[{case_id}]",
                    "next_action": "Change one tile parameter and compare with the baseline",
                } for case_id in self.ids],
            },
            "suggest_next": [{"priority": "P1", "action": "Compare tail tiles", "reason": "Slow case improved"}],
        }

    def merge(self, output=None, **kwargs):
        return merge_tech_lead_update(
            self.base, self.output if output is None else output, 2,
            require_evaluation_summary=True,
            performance_case_ids=kwargs.pop("performance_case_ids", self.ids), **kwargs)

    def test_all_six_cases_archived_without_losing_other_cases_or_rounds(self):
        before = deepcopy(self.base)
        result = self.merge()
        self.assertEqual(self.base, before)
        self.assertEqual(result["rounds"], before["rounds"])
        self.assertEqual(result["ledger"][0], before["ledger"][0])
        self.assertEqual(result["ledger"][1]["case_analysis"], self.output["ledger_entry"]["case_analysis"])
        self.assertEqual(result["ledger"][1]["verdict"], "no_change")
        self.assertEqual(result["worst_cases_tracker"]["operator_7"], before["worst_cases_tracker"]["operator_7"])
        self.assertEqual(set(result["worst_cases_tracker"]), set(self.ids) | {"operator_7"})
        self.assertTrue(all(result["worst_cases_tracker"][case_id]["iteration"] == 2 for case_id in self.ids))

    def test_missing_duplicate_or_invented_case_is_rejected(self):
        variants = [self.output["ledger_entry"]["case_analysis"][:-1],
                    self.output["ledger_entry"]["case_analysis"] + [self.output["ledger_entry"]["case_analysis"][0]]]
        renamed = deepcopy(self.output["ledger_entry"]["case_analysis"])
        renamed[0]["case_id"] = "case_1"
        variants.append(renamed)
        for analyses in variants:
            output = deepcopy(self.output)
            output["ledger_entry"]["case_analysis"] = analyses
            with self.subTest(analyses=analyses), self.assertRaises(ValueError):
                self.merge(output)

    def test_each_case_needs_observation_reason_evidence_and_next_action(self):
        for field in ("observation", "explanation", "evidence", "next_action"):
            output = deepcopy(self.output)
            output["ledger_entry"]["case_analysis"][0][field] = " "
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.merge(output)

    def test_result_summary_is_separate_and_required(self):
        output = deepcopy(self.output)
        output["ledger_entry"].pop("evaluation_summary")
        with self.assertRaisesRegex(ValueError, "evaluation_summary"):
            self.merge(output)

    def test_error_scene_cannot_invent_performance_case_analysis(self):
        with self.assertRaisesRegex(ValueError, "case_analysis"):
            self.merge(performance_case_ids=[])
        self.output["ledger_entry"].pop("case_analysis")
        self.base["ledger"][1]["case_analysis"] = [{"stale": True}]
        result = self.merge(performance_case_ids=[])
        self.assertNotIn("case_analysis", result["ledger"][1])
        self.assertEqual(result["worst_cases_tracker"], self.base["worst_cases_tracker"])

    def test_empty_tracker_update_does_not_clear_old_conclusions(self):
        self.output["worst_cases_tracker"] = {}
        result = self.merge()
        self.assertIn("operator_7", result["worst_cases_tracker"])

    def test_partial_tracker_update_keeps_other_cases(self):
        self.output["worst_cases_tracker"] = {"operator_8": "Different measured case, revised explanation"}
        result = self.merge()
        self.assertIn("operator_7", result["worst_cases_tracker"])
        self.assertIn("operator_8", result["worst_cases_tracker"])

    def test_legacy_list_tracker_is_preserved_when_current_reviews_arrive(self):
        self.base["worst_cases_tracker"] = ["old freeform slow-case note"]
        result = self.merge()
        self.assertEqual(result["legacy_worst_cases_tracker"], self.base["worst_cases_tracker"])
        self.assertEqual(set(result["worst_cases_tracker"]), set(self.ids))

    def test_same_round_rereview_updates_current_analysis_only(self):
        self.base = self.merge()
        original_old = deepcopy(self.base["ledger"][0])
        self.output["ledger_entry"]["case_analysis"][0]["explanation"] = "Measured comparison now supports the hypothesis"
        result = self.merge()
        self.assertEqual(len(result["ledger"]), 2)
        self.assertEqual(result["ledger"][0], original_old)
        self.assertEqual(result["worst_cases_tracker"][self.ids[0]]["explanation"],
                         self.output["ledger_entry"]["case_analysis"][0]["explanation"])

    def test_triggered_experience_requires_its_detailed_fields(self):
        complete = {"applicability": "Current chip and measured shapes", "next_action": "Verify a neighboring tile",
                    "evidence": ["eval/iter2/perf_result.json"],
                    "case_analysis": [{"case_id": "operator_1", "observation": "Latency decreased",
                                       "explanation": "Less padded work in the controlled comparison"}]}
        validate_performance_pattern_details(complete)
        for field in complete:
            incomplete = deepcopy(complete)
            incomplete.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_performance_pattern_details(incomplete)


if __name__ == "__main__":
    unittest.main()

"""Independent review errors must survive a malformed task plan."""

from copy import deepcopy
import unittest

from lib.tech_lead import (Stage9DecisionValidationError, merge_tech_lead_update,
                           validate_stage9_request_conditions, validate_human_responses)
from test_stage9_plan import decision


class Stage9ReviewErrorsTests(unittest.TestCase):
    def test_task_errors_do_not_hide_iteration_fusion_or_human_errors(self):
        raw = decision()
        raw["iteration"] = 2
        raw["suggest_next"][0]["changes"][1]["method"] = ""
        raw["fusion_kernel_strategy"] = [
            {"iter": 1, "direction": ""}, {"iter": 2, "evidence": ""},
        ]
        raw["human_responses"] = [
            {"message_id": "H1", "kind": "direction", "answer": ""},
            {"message_id": "H2", "kind": "conflict", "answer": "", "alternative": ""},
        ]
        before, base = deepcopy(raw), {"ledger": [{"iter": 0, "direction": "old"}]}
        base_before = deepcopy(base)
        with self.assertRaises(Stage9DecisionValidationError) as caught:
            merge_tech_lead_update(base, raw, 3, require_structured_plan=True,
                                  known_case_ids=["case7"],
                                  human_messages=[{"id": "H1"}, {"id": "H2"}])
        message = str(caught.exception)
        for token in ("method", "decision iteration", "fusion_kernel_strategy[0]",
                      "fusion_kernel_strategy[1]", "human_responses[0]",
                      "human_responses[1]", "alternative", "explicit P0"):
            self.assertIn(token, message)
        self.assertEqual(raw, before)
        self.assertEqual(base, base_before)

    def test_each_duplicate_case_is_reported_alongside_missing_cases(self):
        raw = decision()
        entry = {name: "measured" for name in
                 ("observation", "explanation", "evidence", "next_action")}
        raw["ledger_entry"]["case_analysis"] = [
            dict(entry, case_id="case7") for _ in range(3)]
        with self.assertRaises(Stage9DecisionValidationError) as caught:
            merge_tech_lead_update({}, raw, 3, require_structured_plan=True,
                                  performance_case_ids=["case7", "case8"])
        message = str(caught.exception)
        for token in ("case_analysis[1]", "case_analysis[2]", "case8"):
            self.assertIn(token, message)

    def test_pitfall_reports_all_independent_fields(self):
        with self.assertRaises(Stage9DecisionValidationError) as caught:
            validate_stage9_request_conditions({"pitfall": {}}, perf_diff={},
                                               performance_case_ids=[], has_question=True)
        self.assertEqual(len(caught.exception.errors), 6)
        self.assertIn("pitfall.root_cause", str(caught.exception))
        self.assertIn("pitfall.correct_approach", str(caught.exception))

    def test_malformed_human_fields_are_errors_not_type_exceptions(self):
        with self.assertRaises(Stage9DecisionValidationError) as caught:
            validate_human_responses({
                "human_responses": [None, {"message_id": [], "kind": {}, "answer": ""}],
                "suggest_next": [None, {"source": "human", "human_message_id": []}],
            }, [{"id": "H1"}])
        for token in ("human_responses[0]", "human_responses[1]", "H1", "invent human"):
            self.assertIn(token, str(caught.exception))


if __name__ == "__main__":
    unittest.main()

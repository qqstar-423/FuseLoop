"""Request-specific Stage9 contracts agree with their scene acceptance rules."""

from copy import deepcopy
import unittest

from lib.stage9_plan import build_decision_schema, build_decision_template
from lib.tech_lead import Stage9ConditionValidationError, validate_stage9_request_conditions


class Stage9RequestContractTests(unittest.TestCase):
    def validate(self, output, *, diff=None, cases=None, question=False):
        validate_stage9_request_conditions(
            output, perf_diff=diff or {}, performance_case_ids=cases or [], has_question=question)

    def test_precision_omission_and_empty_case_list_are_both_valid(self):
        for ledger in ({}, {"case_analysis": []}):
            self.validate({"ledger_entry": ledger})
        schema = build_decision_schema({}, performance_case_ids=[], has_question=False)
        self.assertNotIn("case_analysis", schema["properties"]["ledger_entry"]["required"])
        self.assertEqual(schema["properties"]["ledger_entry"]["properties"]["case_analysis"]["maxItems"], 0)
        for field in ("proven_pattern", "regression_pattern", "pitfall"):
            self.assertNotIn(field, schema["properties"])

    def test_unrequested_fields_are_not_silently_discarded_even_when_empty(self):
        for value in ({}, None, {"title": "custom error note"}):
            output = {"ledger_entry": {}, "regression_pattern": value, "pitfall": value}
            before = deepcopy(output)
            with self.subTest(value=value), self.assertRaises(Stage9ConditionValidationError) as error:
                self.validate(output)
            self.assertEqual(len(error.exception.errors), 2)
            self.assertIn("regression_pattern", error.exception.errors[0])
            self.assertIn("pitfall", error.exception.errors[1])
            self.assertEqual(output, before)

    def test_precision_case_and_unrequested_experiences_reported_together(self):
        output = {"ledger_entry": {"case_analysis": [{"case_id": f"op_{i}"} for i in range(16, 25)]},
                  "regression_pattern": {"title": "OOB error"}, "pitfall": {"title": "self-test differs"}}
        with self.assertRaises(Stage9ConditionValidationError) as error:
            self.validate(output)
        self.assertEqual(len(error.exception.errors), 3)
        for field in ("regression_pattern", "pitfall", "ledger_entry.case_analysis"):
            self.assertIn(field, str(error.exception))
        self.assertIn("evaluation_summary", str(error.exception))
        self.assertIn("suggest_next", str(error.exception))

    def test_triggered_experience_still_requires_all_details(self):
        for flag, field in (("has_improvement", "proven_pattern"), ("has_regression", "regression_pattern")):
            with self.subTest(field=field), self.assertRaises(Stage9ConditionValidationError) as error:
                self.validate({"ledger_entry": {}, field: {}}, diff={flag: True})
            for suffix in ("what_changed", "fusion_related", "applicability", "evidence", "case_analysis"):
                self.assertIn(f"{field}.{suffix}", str(error.exception))
            with self.assertRaises(Stage9ConditionValidationError) as missing:
                self.validate({"ledger_entry": {}}, diff={flag: True})
            self.assertIn(f"required {field}", str(missing.exception))

    def test_performance_case_contract_exposes_exact_requested_ids(self):
        schema = build_decision_schema({}, performance_case_ids=["op_7", "op_19"], has_question=False)
        ledger = schema["properties"]["ledger_entry"]
        self.assertIn("case_analysis", ledger["required"])
        cases = ledger["properties"]["case_analysis"]
        self.assertEqual((cases["minItems"], cases["maxItems"]), (2, 2))
        self.assertEqual(cases["items"]["properties"]["case_id"]["enum"], ["op_7", "op_19"])
        self.assertNotIn("enum", cases["items"]["properties"]["explanation"])
        with self.assertRaises(Stage9ConditionValidationError) as error:
            self.validate({"ledger_entry": {"case_analysis": [{"case_id": "op_7"}]}}, cases=["op_7", "op_19"])
        self.assertIn("op_19", str(error.exception))

    def test_pitfall_contract_is_only_enabled_for_a_developer_question(self):
        schema = build_decision_schema({}, performance_case_ids=[], has_question=True)
        template = build_decision_template(3, "request-3", [], perf_diff={}, has_question=True)
        self.assertIn("pitfall", schema["required"])
        self.assertEqual(set(template["pitfall"]), set(schema["properties"]["pitfall"]["required"]))
        pitfall = {field: "Evidence-based finding" for field in template["pitfall"]}
        pitfall["verdict"] = "confirmed"
        self.validate({"ledger_entry": {}, "pitfall": pitfall}, question=True)
        with self.assertRaises(Stage9ConditionValidationError):
            self.validate({"ledger_entry": {}}, question=True)
        with self.assertRaises(Stage9ConditionValidationError):
            self.validate({"ledger_entry": {}, "pitfall": {"title": "unstructured note"}}, question=True)


if __name__ == "__main__":
    unittest.main()

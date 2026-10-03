"""Independent malformed branches must not hide other Stage9 plan errors."""

from copy import deepcopy
import unittest

from lib.stage9_plan import (
    DecisionPlanValidationError, build_decision_schema, collect_decision_plan_errors,
    normalize_decision_plan, _validate_schema,
)


def make_decision(count=1):
    tasks = []
    for index in range(count):
        path, case_id = f"impl/c{index}.py", f"case{index}"
        tasks.append({
            "task_id": f"T{index}", "priority": "P1", "task_type": "modify",
            "action": "Tune the measured tile", "reason": "Measured padding overhead",
            "case_scope": "cases", "target_cases": [case_id], "operator_reason": "",
            "case_bindings": [{"case_id": case_id, "implementation_files": [path],
                               "route_evidence": [{"file": "impl/dispatcher.py", "location": case_id,
                                                   "explanation": "This shape routes to the named implementation"}]}],
            "changes": [{"file": path, "operation": "modify", "location": "kernel",
                         "method": "Change the tile while preserving input coverage"}],
            "acceptance_checks": ["Check correctness and measure the affected case"],
        })
    return {"plan_version": 2, "iteration": 3, "request_id": "aggregate-request",
            "ledger_entry": {"evaluation_summary": "Reviewed the measurements", "direction": "Tune tile sizes",
                             "readonly_files": ["impl/dispatcher.py"]}, "suggest_next": tasks}


class Stage9PlanAggregationTests(unittest.TestCase):
    def errors(self, value, **kwargs):
        original = deepcopy(value)
        errors = collect_decision_plan_errors(value, required=True, **kwargs)
        self.assertTrue(errors)
        self.assertEqual(value, original)
        with self.assertRaises(DecisionPlanValidationError) as caught:
            normalize_decision_plan(value, required=True, **kwargs)
        self.assertEqual(caught.exception.errors, errors)
        self.assertEqual(str(caught.exception), "; ".join(errors))
        self.assertEqual(value, original)
        return errors

    def test_two_tasks_readonly_conflicts_are_reported_together(self):
        value = make_decision(2)
        value["ledger_entry"]["readonly_files"] += ["impl/c0.py", "impl/c1.py"]
        errors = self.errors(value)
        self.assertEqual(len(errors), 2)
        self.assertIn("T0", errors[0])
        self.assertIn("T1", errors[1])
        self.assertTrue(all("conflicts with ledger_entry.readonly_files" in error for error in errors))

    def test_one_task_reports_every_write_and_readonly_conflict(self):
        value = make_decision()
        task = value["suggest_next"][0]
        task["case_bindings"][0]["implementation_files"] = ["impl/c0.py", "impl/c1.py"]
        task["changes"].append({**task["changes"][0], "file": "impl/c1.py"})
        value["ledger_entry"]["readonly_files"] = ["impl", "impl/c0.py", "impl/c1.py"]
        errors = self.errors(value)
        self.assertEqual(len(errors), 4)
        self.assertEqual(sum("writes 'impl/c0.py'" in error for error in errors), 2)
        self.assertEqual(sum("writes 'impl/c1.py'" in error for error in errors), 2)

    def test_all_unknown_case_sets_are_reported(self):
        value = make_decision(2)
        errors = self.errors(value, known_case_ids=["case7"])
        self.assertEqual(len(errors), 2)
        self.assertIn("case0", errors[0])
        self.assertIn("case1", errors[1])

    def test_multiple_bound_cases_missing_operations_are_reported(self):
        value = make_decision()
        task = value["suggest_next"][0]
        for index in (1, 2):
            task["target_cases"].append(f"case{index}")
            task["case_bindings"].append({**deepcopy(task["case_bindings"][0]),
                                          "case_id": f"case{index}",
                                          "implementation_files": [f"impl/c{index}.py"]})
        errors = self.errors(value)
        self.assertEqual(len(errors), 2)
        self.assertIn("case 'case1' has no modify operation", errors[0])
        self.assertIn("case 'case2' has no modify operation", errors[1])

    def test_schema_and_path_errors_continue_into_other_tasks(self):
        value = make_decision(2)
        value["request_id"] = None
        first, second = value["suggest_next"]
        first.pop("action")
        first.pop("reason")
        first["acceptance_checks"] = [None, ""]
        first["case_bindings"][0]["route_evidence"][0]["file"] = "../bad-evidence.py"
        second["changes"][0]["file"] = "/outside.py"
        second["case_bindings"][0]["implementation_files"] = ["C:/outside.py"]
        second["priority"] = "P7"
        errors = "\n".join(self.errors(value))
        for fragment in ("request_id must be string", "missing required fields: action, reason",
                         "acceptance_checks[0]", "acceptance_checks[1]", "../bad-evidence.py",
                         "/outside.py", "C:/outside.py", "priority must be one of"):
            self.assertIn(fragment, errors)

    def test_bad_subtrees_do_not_hide_valid_sibling_conflicts(self):
        value = make_decision(2)
        value["suggest_next"][0] = None
        task = value["suggest_next"][1]
        task["changes"].insert(0, 42)
        task["case_bindings"][0]["route_evidence"].insert(0, None)
        value["ledger_entry"]["readonly_files"] += [None, "impl/c1.py"]
        errors = "\n".join(self.errors(value))
        for fragment in ("suggest_next[0] must be object", "changes[0] must be object",
                         "route_evidence[0] must be object", "readonly_files[1] must be string",
                         "writes 'impl/c1.py'", "conflicts with"):
            self.assertIn(fragment, errors)

    def test_incomplete_binding_paths_do_not_invent_outside_or_missing_operation_errors(self):
        value = make_decision()
        value["suggest_next"][0]["case_bindings"][0]["implementation_files"] = [None]
        errors = self.errors(value)
        self.assertEqual(len(errors), 1)
        self.assertIn("implementation_files[0] must be string", errors[0])
        self.assertNotIn("outside", errors[0])
        self.assertNotIn("has no", errors[0])

    def test_bad_operation_or_task_type_is_not_guessed(self):
        value = make_decision(2)
        value["suggest_next"][0]["changes"][0]["operation"] = {"unknown": True}
        value["suggest_next"][1]["task_type"] = []
        value["ledger_entry"]["readonly_files"].append("impl/c1.py")
        errors = "\n".join(self.errors(value))
        self.assertIn("operation must be one of", errors)
        self.assertIn("task_type must be one of", errors)
        self.assertIn("writes 'impl/c1.py'", errors)
        self.assertNotIn("requires at least one", errors)
        self.assertNotIn("has no", errors)

    def test_incomplete_case_ids_do_not_invent_binding_coverage_errors(self):
        value = make_decision()
        value["suggest_next"][0]["target_cases"] = [None]
        value["ledger_entry"]["readonly_files"].append("impl/c0.py")
        errors = "\n".join(self.errors(value))
        self.assertIn("target_cases[0] must be string", errors)
        self.assertIn("conflicts with", errors)
        self.assertNotIn("cover target_cases", errors)

    def test_multiple_path_alias_and_unsafe_path_errors_are_reported(self):
        value = make_decision(2)
        value["ledger_entry"]["readonly_files"] = ["impl/a.py", "impl/./a.py", "../outside.py"]
        value["suggest_next"][0]["changes"][0]["file"] = "impl/*.py"
        value["suggest_next"][1]["changes"][0]["file"] = "impl/"
        errors = "\n".join(self.errors(value))
        for fragment in ("duplicate file aliases", "../outside.py", "impl/*.py", "'impl/'"):
            self.assertIn(fragment, errors)

    def test_duplicate_alias_does_not_hide_known_scope_and_operation_errors(self):
        value = make_decision()
        value["suggest_next"][0]["case_bindings"][0]["implementation_files"] = ["impl/other.py", "impl/./other.py"]
        errors = self.errors(value)
        self.assertEqual(len(errors), 3)
        self.assertIn("duplicate file aliases", errors[0])
        self.assertIn("outside its case_bindings", errors[1])
        self.assertIn("case 'case0' has no modify operation", errors[2])

    def test_legacy_and_version_rejection_boundaries_are_unchanged(self):
        legacy = {"suggest_next": "legacy data"}
        self.assertEqual(collect_decision_plan_errors(legacy), [])
        self.assertEqual(normalize_decision_plan(legacy), legacy)
        self.assertIn("plan_version=2", " ".join(self.errors(legacy)))
        value = make_decision()
        value["plan_version"] = 1
        self.assertIn("must be 2", " ".join(self.errors(value)))

    def test_dynamic_schema_supported_keywords_collect_instead_of_raising_keyerror(self):
        schema = build_decision_schema(perf_diff={"has_improvement": True},
                                       performance_case_ids=[], has_question=False)
        value = make_decision()
        value["ledger_entry"]["case_analysis"] = [{"case_id": "case0", "observation": "Seen",
                                                   "explanation": "Hypothesis", "evidence": "report",
                                                   "next_action": "Compare"}]
        value["proven_pattern"] = {"what_changed": "tile", "why_it_worked": "Less padding",
                                   "fusion_related": None, "case_analysis": [], "evidence": {},
                                   "applicability": "Measured shapes", "next_action": "Test a neighboring size"}
        with self.assertRaises(DecisionPlanValidationError) as caught:
            _validate_schema(value, schema, "decision")
        errors = str(caught.exception)
        for fragment in ("case_analysis must contain at most 0", "fusion_related must be boolean",
                         "case_analysis must contain at least 1", "evidence has an invalid value type"):
            self.assertIn(fragment, errors)

    def test_large_valid_plan_preserves_all_tasks_and_canonical_history(self):
        value = make_decision(1200)
        value["ledger_entry"]["readonly_files"] = [f"reference/r{index}.py" for index in range(1200)]
        original = deepcopy(value)
        self.assertEqual(collect_decision_plan_errors(value), [])
        normalized = normalize_decision_plan(value, known_case_ids=[f"case{index}" for index in range(1200)])
        self.assertEqual(normalized["ledger_entry"]["action_plan"]["tasks"], value["suggest_next"])
        self.assertEqual(normalized["ledger_entry"]["modify_files"], [f"impl/c{index}.py" for index in range(1200)])
        self.assertEqual(value, original)


if __name__ == "__main__":
    unittest.main()

"""Diagnostic extraction must never repair or authorize an invalid Stage9 task."""

from copy import deepcopy
import unittest

from lib.stage9_plan import decision_file_targets_for_diagnostics, normalize_decision_plan


def decision():
    return {
        "plan_version": 2, "iteration": 3, "request_id": "diagnostic-request",
        "ledger_entry": {"evaluation_summary": "Case19 passed; inspect its remaining overhead",
                         "direction": "Check profiling alongside the implementation",
                         "readonly_files": ["task/golden.py"]},
        "suggest_next": [{
            "task_id": "T3", "priority": "P2", "task_type": "inspect",
            "action": "Find case19 overhead", "reason": "Its measured latency is still high",
            "case_scope": "cases", "target_cases": ["case19"], "operator_reason": "",
            "case_bindings": [{
                "case_id": "case19", "implementation_files": ["impl/kernel.py"],
                "route_evidence": [{"file": "impl/dispatcher.py", "location": "case19 branch",
                                    "explanation": "The input shape selects kernel.py"}],
            }],
            "changes": [{"file": "eval/iter3/prof_data", "operation": "inspect",
                         "location": "case19 kernel timings", "method": "Compare compute and transfer timings"}],
            "acceptance_checks": ["Link every hypothesis to measured kernel timings"],
        }],
    }


class Stage9FileDiagnosticsTests(unittest.TestCase):
    def test_semantically_invalid_task_still_exposes_directory_for_physical_diagnostics(self):
        value = decision()
        with self.assertRaisesRegex(ValueError, "no inspect operation"):
            normalize_decision_plan(value, required=True)
        result = decision_file_targets_for_diagnostics(value)
        self.assertEqual(result["suggest_next"][0]["inspect_files"], ["eval/iter3/prof_data"])
        self.assertEqual(result["suggest_next"][0]["modify_files"], [])
        self.assertEqual(result["ledger_entry"]["modify_files"], [])
        self.assertEqual(result["suggest_next"][0]["changes"], value["suggest_next"][0]["changes"])
        self.assertEqual(result["suggest_next"][0]["case_bindings"], value["suggest_next"][0]["case_bindings"])
        self.assertNotIn("action_plan", result["ledger_entry"])

    def test_bad_case_mapping_is_not_fixed_or_accepted(self):
        value = decision()
        value["suggest_next"][0]["case_bindings"][0]["case_id"] = "case20"
        result = decision_file_targets_for_diagnostics(value)
        self.assertEqual(result["suggest_next"][0]["target_cases"], ["case19"])
        self.assertEqual(result["suggest_next"][0]["case_bindings"][0]["case_id"], "case20")
        with self.assertRaisesRegex(ValueError, "cover target_cases"):
            normalize_decision_plan(value, required=True)

    def test_extraction_is_an_independent_copy_without_input_changes(self):
        value = decision()
        original = deepcopy(value)
        result = decision_file_targets_for_diagnostics(value)
        result["suggest_next"][0]["case_bindings"][0]["implementation_files"].append("impl/other.py")
        result["ledger_entry"]["readonly_files"].clear()
        self.assertEqual(value, original)

    def test_all_path_locations_use_existing_normalization(self):
        value = decision()
        value["ledger_entry"]["readonly_files"] = ["task\\golden.py"]
        task = value["suggest_next"][0]
        task["changes"][0]["file"] = "eval\\iter3\\prof_data"
        task["case_bindings"][0]["implementation_files"] = ["impl/./kernel.py"]
        task["case_bindings"][0]["route_evidence"][0]["file"] = "impl\\dispatcher.py"
        result = decision_file_targets_for_diagnostics(value)
        actual = result["suggest_next"][0]
        self.assertEqual(result["ledger_entry"]["readonly_files"], ["task/golden.py"])
        self.assertEqual(actual["changes"][0]["file"], "eval/iter3/prof_data")
        self.assertEqual(actual["case_bindings"][0]["implementation_files"], ["impl/kernel.py"])
        self.assertEqual(actual["case_bindings"][0]["route_evidence"][0]["file"], "impl/dispatcher.py")

    def test_unsafe_paths_are_rejected_in_every_location(self):
        def assign(value, location, path):
            task = value["suggest_next"][0]
            if location == "readonly":
                value["ledger_entry"]["readonly_files"] = [path]
            elif location == "change":
                task["changes"][0]["file"] = path
            elif location == "implementation":
                task["case_bindings"][0]["implementation_files"] = [path]
            else:
                task["case_bindings"][0]["route_evidence"][0]["file"] = path

        for location in ("readonly", "change", "implementation", "evidence"):
            for path in ("../escape.py", "/tmp/kernel.py", "C:/kernel.py", "impl/*.py", "impl/", ""):
                with self.subTest(location=location, path=path):
                    value = decision()
                    assign(value, location, path)
                    original = deepcopy(value)
                    with self.assertRaises(ValueError):
                        decision_file_targets_for_diagnostics(value)
                    self.assertEqual(value, original)

    def test_missing_and_malformed_structure_is_rejected_without_guessing(self):
        invalid = [None, [], "decision"]
        for key in ("plan_version", "iteration", "ledger_entry", "suggest_next"):
            value = decision()
            value.pop(key)
            invalid.append(value)
        for key, replacement in (("case_bindings", None), ("changes", "inspect profiling"),
                                 ("task_id", None), ("target_cases", {})):
            value = decision()
            value["suggest_next"][0][key] = replacement
            invalid.append(value)
        value = decision()
        value["suggest_next"][0]["changes"][0].pop("operation")
        invalid.append(value)
        value = decision()
        value["suggest_next"][0]["case_bindings"][0]["route_evidence"][0]["file"] = 3
        invalid.append(value)
        value = decision()
        value["plan_version"] = 1
        invalid.append(value)
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                decision_file_targets_for_diagnostics(value)

    def test_writes_are_derived_only_from_modify_create_and_readonly_is_preserved(self):
        value = decision()
        task = value["suggest_next"][0]
        task["task_type"] = "modify"
        task["changes"] = [
            {"file": path, "operation": operation, "location": "kernel", "method": "Measured tile experiment"}
            for path, operation in (("impl/a.py", "modify"), ("impl/new.py", "create"),
                                    ("impl/a.py", "inspect"), ("impl/a.py", "modify"))
        ]
        second = deepcopy(task)
        second["task_id"] = "T4"
        second["changes"] = [{"file": "impl/new.py", "operation": "inspect", "location": "kernel",
                              "method": "Read only"}]
        value["suggest_next"].append(second)
        value["ledger_entry"]["readonly_files"].append("impl/a.py")
        result = decision_file_targets_for_diagnostics(value)
        self.assertEqual(result["ledger_entry"]["modify_files"], ["impl/a.py", "impl/new.py"])
        self.assertEqual(result["ledger_entry"]["readonly_files"], ["task/golden.py", "impl/a.py"])
        self.assertEqual(result["suggest_next"][0]["modify_files"], ["impl/a.py", "impl/new.py"])
        self.assertEqual(result["suggest_next"][0]["inspect_files"], ["impl/a.py"])
        self.assertEqual(result["suggest_next"][1]["modify_files"], [])
        self.assertEqual([item["task_id"] for item in result["suggest_next"]], ["T3", "T4"])
        # Diagnostics expose conflicts; they never remove them to make a plan valid.
        with self.assertRaisesRegex(ValueError, "conflicts"):
            normalize_decision_plan(value, required=True)

    def test_model_cannot_supply_derived_authorization_fields(self):
        for location in ("ledger", "task"):
            value = decision()
            target = value["ledger_entry"] if location == "ledger" else value["suggest_next"][0]
            target["modify_files"] = ["impl/invented.py"]
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "unsupported"):
                decision_file_targets_for_diagnostics(value)

    def test_duplicate_normalized_paths_are_not_silently_dropped(self):
        value = decision()
        value["suggest_next"][0]["case_bindings"][0]["implementation_files"] = ["impl/kernel.py", "impl/./kernel.py"]
        with self.assertRaisesRegex(ValueError, "duplicate file aliases"):
            decision_file_targets_for_diagnostics(value)


if __name__ == "__main__":
    unittest.main()

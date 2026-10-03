"""Large multi-task plans must preserve every file and global protection."""

from copy import deepcopy
import unittest

from lib.stage9_plan import normalize_decision_plan, validate_current_plan
from test_stage9_plan import decision


def many_file_decision():
    value = decision()
    prototype = value["suggest_next"][0]
    tasks = []
    expected = []
    for index in range(200):
        task = deepcopy(prototype)
        case_id = f"case_{index}"
        files = [f"impl/group_{index}/part_{part}.py" for part in range(10)]
        # Multiple tasks may legitimately modify the same shared helper.
        files.append("impl/shared.py")
        task.update(task_id=f"T{index}", target_cases=[case_id])
        task["case_bindings"][0].update(case_id=case_id, implementation_files=files)
        task["changes"] = [deepcopy(prototype["changes"][0])] + [
            dict(file=path, operation="modify", location="kernel",
                 method="Update the kernel together with its shared helper.")
            for path in files]
        tasks.append(task)
        expected.extend(files)
    value["suggest_next"] = tasks
    return value, list(dict.fromkeys(expected))


class Stage9ManyFilesTests(unittest.TestCase):
    def test_two_thousand_files_and_shared_helper_are_preserved_through_handoff(self):
        raw, expected = many_file_decision()
        normalized = normalize_decision_plan(
            raw, required=True, known_case_ids=[f"case_{index}" for index in range(200)])
        ledger, tasks = validate_current_plan(normalized["ledger_entry"], normalized["suggest_next"])
        self.assertEqual(len(expected), 2001)
        self.assertEqual(ledger["modify_files"], expected)
        self.assertEqual(ledger["action_plan"]["tasks"], raw["suggest_next"])
        self.assertEqual(len(tasks), 200)
        self.assertNotIn("impl/dispatcher.py", ledger["modify_files"])
        self.assertEqual(tasks[-1]["modify_files"], raw["suggest_next"][-1]["case_bindings"][0]["implementation_files"])

    def test_last_file_in_large_plan_cannot_bypass_global_readonly(self):
        raw, _ = many_file_decision()
        raw["ledger_entry"]["readonly_files"].append("impl/group_199/part_9.py")
        with self.assertRaisesRegex(ValueError, "group_199/part_9.*readonly_files"):
            normalize_decision_plan(raw, required=True)

    def test_omitted_mapping_in_last_task_is_still_rejected(self):
        raw, _ = many_file_decision()
        raw["suggest_next"][-1]["case_bindings"][0]["implementation_files"].remove("impl/group_199/part_9.py")
        with self.assertRaisesRegex(ValueError, "group_199/part_9.*outside"):
            normalize_decision_plan(raw, required=True)


if __name__ == "__main__":
    unittest.main()

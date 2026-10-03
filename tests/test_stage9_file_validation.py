"""Independent filesystem errors must reach the same Stage9 repair request."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import orchestrator
from lib.tech_lead import Stage9DecisionValidationError


def raw_plan(*tasks, readonly=()):
    return {
        "plan_version": 2,
        "ledger_entry": {"readonly_files": list(readonly)},
        "suggest_next": [
            {"task_id": f"T{index}", "changes": [
                {"file": path, "operation": operation} for path, operation in task],
             "case_bindings": []}
            for index, task in enumerate(tasks, 1)
        ],
    }


class Stage9FileValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage9_files_aggregate_")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        (self.work / "impl").mkdir()

    def write(self, relative):
        target = self.work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic source\n", encoding="utf-8")
        return target

    def rejected(self, plan, **kwargs):
        before = deepcopy(plan)
        with self.assertRaises(Stage9DecisionValidationError) as caught:
            orchestrator._validate_stage9_file_targets(self.work, plan, **kwargs)
        self.assertEqual(plan, before, "Diagnostics must not repair or authorize model output")
        return caught.exception.errors

    def test_two_tasks_report_both_readonly_conflicts(self):
        self.write("impl/a.py")
        self.write("impl/b.py")
        errors = self.rejected(raw_plan([("impl/a.py", "modify")], [("impl/b.py", "modify")],
                                        readonly=["impl/a.py", "impl/b.py"]))
        self.assertEqual(len(errors), 2)
        self.assertIn("(T1).changes[0].file", errors[0])
        self.assertIn("impl/a.py", errors[0])
        self.assertIn("(T2).changes[0].file", errors[1])
        self.assertIn("impl/b.py", errors[1])

    def test_directory_missing_file_and_existing_create_all_reported(self):
        (self.work / "eval/iter3/prof_data").mkdir(parents=True)
        self.write("impl/existing.py")
        errors = self.rejected(raw_plan(
            [("eval/iter3/prof_data", "inspect"), ("impl/missing.py", "modify")],
            [("impl/existing.py", "create")]))
        self.assertEqual(len(errors), 3)
        self.assertTrue(any("目录" in item and "prof_data" in item for item in errors))
        self.assertTrue(any("不存在" in item and "missing.py" in item for item in errors))
        self.assertTrue(any("已存在" in item and "existing.py" in item for item in errors))

    def test_bindings_and_route_evidence_missing_are_not_hidden(self):
        plan = raw_plan([("impl/missing.py", "inspect")])
        plan["suggest_next"][0]["case_bindings"] = [{
            "implementation_files": ["impl/variant.py", "impl/variant2.py"],
            "route_evidence": [{"file": "impl/dispatch.py"}, {"file": "task/mapping.json"}],
        }]
        errors = self.rejected(plan)
        self.assertEqual(len(errors), 5)
        self.assertTrue(any("case_bindings[0].implementation_files[1]" in item for item in errors))
        self.assertTrue(any("route_evidence[1].file" in item for item in errors))

    def test_duplicate_missing_file_is_reported_once(self):
        plan = raw_plan([("impl/missing.py", "inspect"), ("impl/missing.py", "inspect")])
        plan["suggest_next"][0]["case_bindings"] = [{
            "implementation_files": ["impl/missing.py"],
            "route_evidence": [{"file": "impl/missing.py"}],
        }]
        self.assertEqual(len(self.rejected(plan)), 1)

    def test_raw_bad_structure_does_not_hide_valid_sibling_paths(self):
        plan = raw_plan([("impl/missing.py", "inspect")])
        plan["ledger_entry"] = None
        plan["suggest_next"].insert(0, "bad task")
        plan["suggest_next"][1]["changes"].extend([
            "bad change", {"file": {}, "operation": "inspect"},
            {"file": "impl/second_missing.py", "operation": "inspect"},
        ])
        errors = self.rejected(plan)
        self.assertEqual(len(errors), 6)
        self.assertTrue(any("missing.py" in item for item in errors))
        self.assertTrue(any("second_missing.py" in item for item in errors))

    def test_missing_version_does_not_require_model_authored_permission_lists(self):
        self.write("impl/existing.py")
        plan = raw_plan([("impl/existing.py", "modify"), ("impl/missing.py", "inspect")])
        del plan["plan_version"]
        errors = self.rejected(plan, raw_plan=True)
        self.assertEqual(len(errors), 1)
        self.assertIn("impl/missing.py", errors[0])
        self.assertNotIn("权限范围冲突", errors[0])

    def test_unsafe_paths_never_reach_filesystem_but_siblings_do(self):
        plan = raw_plan([
            ("../outside.py", "modify"), ("C:/outside.py", "inspect"),
            ("impl\x00bad.py", "inspect"), ("impl/*.py", "inspect"),
            ("impl/missing.py", "inspect"),
        ])
        original_stat = Path.stat
        visited = []

        def stat(path, *args, **kwargs):
            visited.append(str(path))
            self.assertNotIn("outside.py", str(path))
            self.assertNotIn("\x00", str(path))
            self.assertNotIn("*", str(path))
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", stat):
            errors = self.rejected(plan)
        self.assertEqual(len(errors), 5)
        self.assertTrue(any("missing.py" in path for path in visited))

    def test_resolve_failure_does_not_hide_missing_sibling(self):
        failing = self.work / "impl/bad_link.py"
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == failing:
                raise RuntimeError("synthetic symlink loop")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            errors = self.rejected(raw_plan([
                ("impl/bad_link.py", "inspect"), ("impl/missing.py", "inspect")]))
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("无法解析" in item for item in errors))
        self.assertTrue(any("missing.py" in item for item in errors))

    def test_create_binding_and_resume_existing_create_remain_valid(self):
        plan = raw_plan([("impl/new.py", "create")])
        plan["suggest_next"][0]["case_bindings"] = [{
            "implementation_files": ["impl/new.py"], "route_evidence": [],
        }]
        orchestrator._validate_stage9_file_targets(self.work, plan)
        self.assertFalse((self.work / "impl/new.py").exists())
        self.write("impl/new.py")
        orchestrator._validate_stage9_file_targets(self.work, plan, allow_existing_creates=True)

    def test_pending_restore_base_paths_are_checked_in_snapshot(self):
        self.write("selection/records/iter1/impl/existing.py")
        plan = raw_plan([("impl/existing.py", "modify"), ("impl/missing.py", "modify")])
        errors = self.rejected(plan, implementation_base=self.work / "selection/records/iter1/impl")
        self.assertEqual(len(errors), 1)
        self.assertIn("impl/missing.py", errors[0])
        self.assertFalse((self.work / "impl/existing.py").exists())


if __name__ == "__main__":
    unittest.main()

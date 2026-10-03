"""Concrete multi-file plans must remain applicable and respect readonly files."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import orchestrator


def local_plan(*tasks, readonly=()):
    """Already-normalized view: isolate filesystem checks from the JSON schema."""
    suggestions = []
    writable = []
    for index, changes in enumerate(tasks, 1):
        writes = list(dict.fromkeys(path for path, operation in changes
                                    if operation in {"modify", "create"}))
        writable.extend(writes)
        suggestions.append({
            "task_id": f"T{index}", "modify_files": writes,
            "inspect_files": [path for path, operation in changes if operation == "inspect"],
            "changes": [{"file": path, "operation": operation} for path, operation in changes],
            "case_bindings": [],
        })
    return {"ledger_entry": {"modify_files": list(dict.fromkeys(writable)),
                              "readonly_files": list(readonly)},
            "suggest_next": suggestions}


class Stage9FileTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="stage9_file_targets_")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        (self.work / "impl").mkdir()

    def write(self, relative, text="synthetic source\n"):
        target = self.work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def validate(self, decision, **options):
        return orchestrator._validate_stage9_file_targets(self.work, decision, **options)

    def test_hardlink_to_readonly_file_is_rejected(self):
        protected = self.write("impl/protected.py")
        alias = self.work / "impl/alias.py"
        os.link(protected, alias)
        self.assertTrue(protected.samefile(alias))
        with self.assertRaisesRegex(ValueError, "实际指向同一文件"):
            self.validate(local_plan([("impl/alias.py", "modify")],
                                     readonly=["impl/protected.py"]))

    def test_hardlink_readonly_conflict_across_tasks_is_rejected(self):
        protected = self.write("impl/protected.py")
        self.write("impl/other.py")
        os.link(protected, self.work / "impl/alias.py")
        with self.assertRaisesRegex(ValueError, "实际指向同一文件"):
            self.validate(local_plan([("impl/other.py", "modify")],
                                     [("impl/alias.py", "modify")],
                                     readonly=["impl/protected.py"]))

    def test_inspecting_readonly_hardlink_is_allowed(self):
        protected = self.write("impl/protected.py")
        os.link(protected, self.work / "impl/alias.py")
        self.validate(local_plan([("impl/alias.py", "inspect")],
                                 readonly=["impl/protected.py"]))

    def test_undeclared_hardlink_to_fixed_input_cannot_be_authorized(self):
        protected = self.write("task/golden.py")
        os.link(protected, self.work / "impl/alias.py")
        with self.assertRaisesRegex(ValueError, "多个硬链接"):
            self.validate(local_plan([("impl/alias.py", "modify")]))
        self.assertEqual(protected.read_text(encoding="utf-8"), "synthetic source\n")

    def test_multiple_hardlinks_within_impl_require_independent_copy_before_writing(self):
        original = self.write("impl/kernel.py")
        os.link(original, self.work / "impl/alias.py")
        with self.assertRaisesRegex(ValueError, "多个硬链接"):
            self.validate(local_plan([("impl/kernel.py", "modify"), ("impl/alias.py", "modify")]))
        self.validate(local_plan([("impl/alias.py", "inspect")]))

    def test_new_file_under_existing_file_is_rejected(self):
        self.write("impl/existing.py")
        with self.assertRaisesRegex(ValueError, "父级已是文件"):
            self.validate(local_plan([("impl/existing.py/nested/new.py", "create")]))

    def test_same_task_parent_and_child_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "另一待写文件的目录"):
            self.validate(local_plan([("impl/new.py", "create"),
                                      ("impl/new.py/child.py", "create")]))

    def test_cross_task_parent_and_child_files_are_rejected(self):
        for tasks in (([("impl/new.py", "create")], [("impl/new.py/sub/child.py", "create")]),
                      ([("impl/new.py/sub/child.py", "create")], [("impl/new.py", "create")])):
            with self.subTest(tasks=tasks):
                with self.assertRaisesRegex(ValueError, "另一待写文件的目录"):
                    self.validate(local_plan(*tasks))

    def test_many_files_sharing_new_directory_are_allowed(self):
        tasks = [[(f"impl/new_package/group{group}/kernel{index}.py", "create")
                  for index in range(30)] for group in range(20)]
        decision = local_plan(*tasks)
        self.assertEqual(len(decision["ledger_entry"]["modify_files"]), 600)
        self.validate(decision)
        self.assertFalse((self.work / "impl/new_package").exists())

    def test_shared_existing_directory_and_repeated_modify_are_allowed(self):
        self.write("impl/package/kernel.py")
        self.write("impl/package/helper.py")
        self.validate(local_plan([("impl/package/kernel.py", "modify")],
                                 [("impl/package/helper.py", "modify"),
                                  ("impl/package/kernel.py", "modify")]))

    def test_many_existing_files_use_cached_identity_checks(self):
        for index in range(120):
            self.write(f"impl/write{index}.py")
            self.write(f"impl/read{index}.py")
        decision = local_plan([(f"impl/write{index}.py", "modify") for index in range(120)],
                              readonly=[f"impl/read{index}.py" for index in range(120)])
        original_stat = Path.stat
        stat_calls = []

        def counted_stat(path, *args, **kwargs):
            stat_calls.append(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", counted_stat), \
                patch.object(Path, "samefile", side_effect=AssertionError("Do not compare every pair")):
            self.validate(decision)
        # resolve() and existence checks vary by OS. A loose linear bound still
        # catches accidental per-pair stats across all readonly/write files.
        self.assertLess(len(stat_calls), 240 * 15)

    def test_restore_base_hardlink_readonly_conflict_is_rejected(self):
        base = self.work / "selection/records/iter1/impl"
        base.mkdir(parents=True)
        protected = base / "protected.py"
        protected.write_text("snapshot", encoding="utf-8")
        os.link(protected, base / "alias.py")
        with self.assertRaisesRegex(ValueError, "实际指向同一文件"):
            self.validate(local_plan([("impl/alias.py", "modify")],
                                     readonly=["impl/protected.py"]), implementation_base=base)

    def test_resume_existing_created_file_still_checks_parent_conflicts(self):
        self.write("impl/new.py")
        with self.assertRaisesRegex(ValueError, "另一待写文件的目录|父级已是文件"):
            self.validate(local_plan([("impl/new.py", "create"),
                                      ("impl/new.py/child.py", "create")]),
                          allow_existing_creates=True)

    def test_program_owned_and_readonly_inputs_cannot_be_declared_writable(self):
        files = [f"{directory}/synthetic.json" for directory in (
            "task", "example", "knowledge", "eval", "selection", "fusion", "human_review",
            "build", "profile", "search", "log", "operator_iter")]
        files.extend([".state.json", "ANALYSIS.md", "fusion_requirements.en.json", "device_info.json"])
        for relative in files:
            with self.subTest(relative=relative):
                self.write(relative)
                with self.assertRaisesRegex(ValueError, "不可修改只读输入或程序维护文件"):
                    self.validate(local_plan([(relative, "modify")]))

    def test_create_inside_protected_area_is_rejected_without_existing_file(self):
        with self.assertRaisesRegex(ValueError, "不可修改只读输入或程序维护文件"):
            self.validate(local_plan([("knowledge/fake_history.json", "create")]))

    def test_declared_inspections_of_program_owned_inputs_are_allowed(self):
        self.write("task/golden.py")
        self.write("knowledge/history.json")
        self.validate(local_plan([("task/golden.py", "inspect"),
                                 ("knowledge/history.json", "inspect")]))

    def test_protected_names_do_not_block_similar_directory_names(self):
        self.validate(local_plan([("knowledge_helper/new.py", "create")]))

    def test_implementation_alias_to_protected_input_is_rejected(self):
        actual = self.write("task/golden.py").resolve()
        alias = self.work / "impl/alias.py"
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            result = original_resolve(path, *args, **kwargs)
            return actual if result == alias else result

        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "实际指向只读输入或程序维护文件"):
                self.validate(local_plan([("impl/alias.py", "modify")]))

    def test_logical_impl_is_validated_against_selected_restore_snapshot(self):
        base = self.work / "selection/records/iter1/impl"
        self.write("selection/records/iter1/impl/kernel.py")
        self.validate(local_plan([("impl/kernel.py", "modify")]), implementation_base=base)
        self.assertFalse((self.work / "impl/kernel.py").exists())

    def test_snapshot_itself_is_not_writable_even_with_restore_base(self):
        base = self.work / "selection/records/iter1/impl"
        self.write("selection/records/iter1/impl/kernel.py")
        with self.assertRaisesRegex(ValueError, "不可修改只读输入或程序维护文件"):
            self.validate(local_plan([("selection/records/iter1/impl/kernel.py", "modify")]),
                          implementation_base=base)

    def test_restore_base_does_not_exempt_alias_escaping_to_protected_input(self):
        base = (self.work / "selection/records/iter1/impl").resolve()
        base.mkdir(parents=True)
        actual = self.write("task/golden.py").resolve()
        alias = base / "alias.py"
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            result = original_resolve(path, *args, **kwargs)
            return actual if result == alias else result

        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "实际指向只读输入或程序维护文件"):
                self.validate(local_plan([("impl/alias.py", "modify")]), implementation_base=base)

    def test_normal_impl_link_to_snapshot_is_not_a_rollback_exemption(self):
        base = self.work / "impl"
        snapshot = self.work / "selection/records/iter1/impl"
        self.write("selection/records/iter1/impl/kernel.py")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            result = original_resolve(path, *args, **kwargs)
            if result == base or result.is_relative_to(base):
                return snapshot / result.relative_to(base)
            return result

        with patch.object(Path, "resolve", resolve):
            with self.assertRaisesRegex(ValueError, "实际指向只读输入或程序维护文件"):
                self.validate(local_plan([("impl/kernel.py", "modify")]), implementation_base=base)


if __name__ == "__main__":
    unittest.main()

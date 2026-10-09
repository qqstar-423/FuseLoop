"""Stage9 advice must identify real targets consistent with its current scope."""

from copy import deepcopy
import unittest

from lib.tech_lead import merge_tech_lead_update, validate_advice_file_scope


class Stage9AdviceScopeTests(unittest.TestCase):
    def setUp(self):
        self.c2 = "impl/cann_bench/fused_conv_tanh/c2/test_fused_conv_tanh.py"
        self.c3 = "impl/cann_bench/fused_conv_tanh/c3/test_fused_conv_tanh.py"
        self.base = {"rounds": [{"iter": 1, "avg_speedup": 3.9199}], "ledger": []}
        self.output = {
            "iteration": 1,
            "ledger_entry": {
                "direction": "Inspect case20 dispatch before deciding whether to change tiles",
                "modify_files": [self.c2], "readonly_files": [self.c3],
            },
            "suggest_next": [{
                "priority": "P1", "action": "Inspect case20's bfloat16 path in c3",
                "reason": "The measured case routes through c3; no c3 edit in this plan",
                "inspect_files": [self.c3], "modify_files": [],
            }],
        }

    def merge(self, **kwargs):
        return merge_tech_lead_update(self.base, self.output, 1, **{
            "require_file_scope": True, **kwargs,
        })

    def test_readonly_c3_may_be_inspected_without_permission_to_modify(self):
        result = self.merge()
        self.assertEqual(result["suggest_next"][0]["inspect_files"], [self.c3])
        self.assertEqual(result["suggest_next"][0]["modify_files"], [])

    def test_case20_modification_is_rejected_while_c3_is_readonly(self):
        self.output["suggest_next"][0]["modify_files"] = [self.c3]
        before = deepcopy((self.base, self.output))
        with self.assertRaisesRegex(ValueError, r"suggest_next\[0\].modify_files.*c3.*readonly_files"):
            self.merge()
        self.assertEqual((self.base, self.output), before)

    def test_case20_modification_requires_current_ledger_permission(self):
        self.output["ledger_entry"]["readonly_files"] = []
        self.output["suggest_next"][0]["modify_files"] = [self.c3]
        with self.assertRaisesRegex(ValueError, "not in ledger_entry.modify_files"):
            self.merge()
        self.output["ledger_entry"]["modify_files"] = [self.c3]
        result = self.merge()
        self.assertEqual(result["suggest_next"][0]["modify_files"], [self.c3])

    def test_mixed_inspection_and_modification_can_target_different_files(self):
        self.output["suggest_next"][0]["modify_files"] = [self.c2]
        result = self.merge()
        self.assertEqual(result["suggest_next"][0]["inspect_files"], [self.c3])
        self.assertEqual(result["suggest_next"][0]["modify_files"], [self.c2])

    def test_missing_or_empty_scope_is_rejected(self):
        for missing in ("inspect_files", "modify_files"):
            old = self.output["suggest_next"][0].pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                self.merge()
            self.output["suggest_next"][0][missing] = old
        self.output["suggest_next"][0]["inspect_files"] = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.merge()

    def test_new_scope_is_validated_even_if_legacy_flag_is_not_enabled(self):
        self.output["suggest_next"][0]["modify_files"] = [self.c3]
        with self.assertRaisesRegex(ValueError, "readonly_files"):
            self.merge(require_file_scope=False)

    def test_legacy_decisions_remain_readable_without_new_contract(self):
        for field in ("inspect_files", "modify_files"):
            self.output["suggest_next"][0].pop(field)
        self.output["ledger_entry"]["fix_plan"] = {"old": "historical plan"}
        result = self.merge(require_file_scope=False)
        self.assertEqual(result["ledger"][0]["fix_plan"], {"old": "historical plan"})
        with self.assertRaises(ValueError):
            self.merge()

    def test_nested_fix_plan_cannot_override_authoritative_scope(self):
        self.output["ledger_entry"]["fix_plan"] = {"modify_files": [self.c3]}
        with self.assertRaisesRegex(ValueError, "fix_plan.*single current plan"):
            self.merge()

    def test_slash_and_dot_aliases_normalize_before_saving(self):
        self.output["ledger_entry"]["modify_files"] = ["./" + self.c2.replace("/", "\\")]
        self.output["suggest_next"][0]["modify_files"] = [self.c2, "./" + self.c2]
        before = deepcopy(self.output)
        result = self.merge()
        self.assertEqual(result["ledger"][0]["modify_files"], [self.c2])
        self.assertEqual(result["suggest_next"][0]["modify_files"], [self.c2])
        self.assertEqual(self.output, before)

    def test_alias_cannot_bypass_readonly(self):
        self.output["suggest_next"][0]["modify_files"] = ["./" + self.c3.replace("/", "\\")]
        with self.assertRaisesRegex(ValueError, "readonly_files"):
            self.merge()

    def test_parent_readonly_path_cannot_be_bypassed_by_child_file(self):
        self.output["ledger_entry"]["readonly_files"] = ["impl/cann_bench/fused_conv_tanh/c3"]
        self.output["ledger_entry"]["modify_files"] = [self.c3]
        with self.assertRaisesRegex(ValueError, "overlaps readonly_files"):
            self.merge()

    def test_child_readonly_path_cannot_be_bypassed_by_parent_permission(self):
        self.output["ledger_entry"]["modify_files"] = ["impl/cann_bench/fused_conv_tanh"]
        with self.assertRaisesRegex(ValueError, "overlaps readonly_files"):
            self.merge()

    def test_absolute_escape_directory_and_glob_paths_are_rejected(self):
        paths = ["/impl/kernel.py", "C:/impl/kernel.py", "C:kernel.py",
                 "\\\\host\\share\\kernel.py", "../kernel.py", "impl/../kernel.py",
                 "impl\\..\\kernel.py", "impl/", "impl\\", ".", "./",
                 "impl/*.py", "impl/a?.py", "impl/[ab].py", "impl/<iter>/kernel.py",
                 "impl/kernel.py:stream", " impl/kernel.py", "impl/ker\nnel.py"]
        for field in ("inspect_files", "modify_files"):
            for path in paths:
                before = deepcopy(self.output)
                self.output["suggest_next"][0][field] = [path]
                with self.subTest(field=field, path=path), self.assertRaises(ValueError):
                    self.merge()
                self.output = before

    def test_ledger_paths_must_also_be_concrete(self):
        for field in ("modify_files", "readonly_files"):
            self.output["ledger_entry"][field] = ["impl/*.py"]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "concrete work-relative file"):
                self.merge()
            self.output["ledger_entry"][field] = []

    def test_confirmed_exit_may_omit_suggestions(self):
        self.output["suggest_next"] = []
        with self.assertRaisesRegex(ValueError, "suggest_next"):
            self.merge()
        self.assertEqual(self.merge(allow_empty_suggestions=True)["suggest_next"], [])

    def test_success_preserves_previous_ledger_and_performance(self):
        self.base["ledger"] = [{"iter": 0, "direction": "earlier plan", "modify_files": ["old.py"]}]
        before = deepcopy((self.base, self.output))
        result = self.merge()
        self.assertEqual(result["ledger"][0], self.base["ledger"][0])
        self.assertEqual(result["rounds"], self.base["rounds"])
        self.assertEqual((self.base, self.output), before)

    def test_resume_validator_rejects_legacy_scope_without_requiring_other_fields(self):
        ledger = self.output["ledger_entry"]
        legacy = [{"priority": "P1", "action": "Inspect case20", "reason": "Slow tail"}]
        self.assertEqual(validate_advice_file_scope(ledger, legacy), (ledger, legacy))
        with self.assertRaisesRegex(ValueError, "inspect_files"):
            validate_advice_file_scope(ledger, legacy, required=True)
        # Resume validation is deliberately independent of iteration, measured
        # metrics, and task payloads already committed by the workflow.
        result = validate_advice_file_scope(ledger, self.output["suggest_next"], required=True)
        self.assertEqual(result, (ledger, self.output["suggest_next"]))

    def test_resume_validator_rejects_malformed_inputs_with_value_error(self):
        for ledger, suggestions in ((None, []), ({}, None), ({}, [None]), ({}, [{}])):
            with self.subTest(ledger=ledger, suggestions=suggestions), self.assertRaises(ValueError):
                validate_advice_file_scope(ledger, suggestions, required=True)


if __name__ == "__main__":
    unittest.main()

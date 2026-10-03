"""Offline validation of model-owned Stage9 decisions and protected history."""

from copy import deepcopy
import unittest

from lib.tech_lead import merge_tech_lead_update


class TechLeadMergeTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "rounds": [{"iter": 1, "avg_speedup": 1.9}, {"iter": 2, "avg_speedup": 2.1}],
            "ledger": [
                {"iter": 1, "direction": "Keep the proven kernel", "modify_files": ["old.py"],
                 "readonly_files": [], "avg_speedup_after": 1.9, "verdict": "baseline"},
                {"iter": 2, "reason": "perf_optimize", "direction": "(pending tech_lead)",
                 "modify_files": [], "readonly_files": [], "avg_speedup_before": 1.9,
                 "avg_speedup_after": 2.1, "delta": 0.2, "verdict": "big_win"},
            ],
            "insights": ["old insight"], "bottleneck_now": "launch overhead",
            "suggest_next": [{"priority": "P0", "action": "old plan", "reason": "old evidence"}],
            "worst_cases_tracker": {"case1": "still slow"},
            "fusion_kernel_strategy": [{"iter": 1, "direction": "two kernels", "status": "verified"}],
            "other_program_data": {"keep": [1, 2]},
        }
        self.output = {
            "iteration": 2, "request_id": "this-request",
            "ledger_entry": {"direction": "Tune only the slow shape", "modify_files": ["impl/kernel.py"],
                             "readonly_files": ["impl/proven.py"], "fix_plan": {"note": "preserve fast shapes"}},
            "insights": ["Tiling reduced padding"], "bottleneck_now": "small-shape padding",
            "suggest_next": [{"priority": "P0", "action": "Tune tail tile", "reason": "case1 is still below 1"}],
            "worst_cases_tracker": {"case1": "padding dominates"},
            "fusion_kernel_strategy": [{"iter": 2, "direction": "shape routing", "evidence": "measured",
                                        "status": "verified"}],
        }

    def merge(self, output=None, base=None, **kwargs):
        return merge_tech_lead_update(self.base if base is None else base,
                                     self.output if output is None else output, 2, **kwargs)

    def test_protects_old_ledger_all_rounds_and_current_hard_metrics(self):
        result = self.merge(reason="perf_optimize\nAdditional review instructions")
        self.assertEqual(result["rounds"], self.base["rounds"])
        self.assertEqual(result["ledger"][0], self.base["ledger"][0])
        for field in ("iter", "avg_speedup_before", "avg_speedup_after", "delta", "verdict"):
            self.assertEqual(result["ledger"][1][field], self.base["ledger"][1][field])
        self.assertEqual(result["ledger"][1]["reason"], "perf_optimize")
        self.assertEqual(result["ledger"][1]["direction"], self.output["ledger_entry"]["direction"])
        self.assertEqual(result["other_program_data"], self.base["other_program_data"])

    def test_no_input_or_nested_alias_is_mutated(self):
        base_before, output_before = deepcopy(self.base), deepcopy(self.output)
        result = self.merge()
        result["ledger"][1]["modify_files"].append("extra.py")
        result["rounds"][0]["avg_speedup"] = 999
        result["other_program_data"]["keep"].append(3)
        result["fusion_kernel_strategy"][-1]["direction"] = "changed"
        self.assertEqual(self.base, base_before)
        self.assertEqual(self.output, output_before)

    def test_repeat_is_idempotent_for_ledger_and_fusion_attempt(self):
        first = self.merge(reason="perf_optimize")
        second = self.merge(base=first, reason="perf_optimize")
        self.assertEqual(first, second)
        self.assertEqual(len(second["ledger"]), 2)
        self.assertEqual(len(second["fusion_kernel_strategy"]), 2)

    def test_missing_current_ledger_creates_only_unmeasured_fields(self):
        self.base["ledger"].pop()
        result = self.merge(reason="precision_fail")
        current = result["ledger"][-1]
        self.assertEqual(current["iter"], 2)
        self.assertEqual(current["reason"], "precision_fail")
        for field in ("avg_speedup_before", "avg_speedup_after", "delta", "verdict"):
            self.assertIsNone(current[field])
        self.assertEqual(result["ledger"][0], self.base["ledger"][0])

    def test_reason_preserved_when_caller_has_no_new_reason(self):
        self.base["ledger"][-1]["reason"] = "build_fail"
        self.assertEqual(self.merge()["ledger"][-1]["reason"], "build_fail")

    def test_model_conclusions_replace_but_omitted_conclusions_stay(self):
        self.output.pop("insights")
        result = self.merge()
        self.assertEqual(result["insights"], self.base["insights"])
        self.assertEqual(result["suggest_next"], self.output["suggest_next"])
        self.assertEqual(result["worst_cases_tracker"], self.output["worst_cases_tracker"])

    def test_empty_suggestions_only_allowed_for_confirmed_exit(self):
        self.output["suggest_next"] = []
        with self.assertRaises(ValueError):
            self.merge()
        self.assertEqual(self.merge(allow_empty_suggestions=True)["suggest_next"], [])
        self.output.pop("suggest_next")
        with self.assertRaises(ValueError):
            self.merge(allow_empty_suggestions=True)

    def test_absent_fix_plan_does_not_keep_old_nested_plan(self):
        self.base["ledger"][-1]["fix_plan"] = {"old": "do not reuse"}
        self.output["ledger_entry"].pop("fix_plan")
        self.assertNotIn("fix_plan", self.merge()["ledger"][-1])

    def test_current_fusion_dict_and_list_are_supported(self):
        entry = self.output["fusion_kernel_strategy"][0]
        self.output["fusion_kernel_strategy"] = entry
        self.assertEqual(self.merge()["fusion_kernel_strategy"][-1], entry)
        self.output["fusion_kernel_strategy"] = [entry, deepcopy(entry)]
        self.assertEqual(len(self.merge()["fusion_kernel_strategy"]), 2)

    def test_experience_is_validated_but_not_saved_or_reused(self):
        self.base.update({"proven_pattern": {"old": True}, "regression_pattern": {"old": True},
                          "pitfall": {"old": True}, "_pending_proven_pattern": {"old": True},
                          "_pending_regression_pattern": {"old": True}, "_pending_pitfall": {"old": True}})
        self.output["proven_pattern"] = {"what_changed": "smaller tile", "why_it_worked": "less padding",
                                         "fusion_related": False, "case_analysis": [{"case": "case1"}]}
        self.output["regression_pattern"] = {"what_changed": "larger tile", "why_it_failed": "too much padding",
                                             "fusion_related": False}
        self.output["pitfall"] = {"verdict": "rejected", "topic": "tile capacity", "target_advice": "use tile 32",
                                  "feedback": "cannot use it", "root_cause": "wrong capacity assumption",
                                  "correct_approach": "use the verified capacity"}
        result = self.merge()
        for field in ("proven_pattern", "regression_pattern", "pitfall", "_pending_proven_pattern",
                      "_pending_regression_pattern", "_pending_pitfall"):
            self.assertNotIn(field, result)
        self.assertIn("case_analysis", self.output["proven_pattern"])

    def test_invalid_outputs_never_partially_mutate_history(self):
        invalid = []
        for iteration in (1, 3, True, "2", 2.0, None):
            candidate = deepcopy(self.output)
            candidate["iteration"] = iteration
            invalid.append(candidate)
        for field in ("ledger", "rounds", "avg_speedup", "exit_decision"):
            candidate = deepcopy(self.output)
            candidate[field] = []
            invalid.append(candidate)
        for field in ("iter", "reason", "avg_speedup_after", "verdict"):
            candidate = deepcopy(self.output)
            candidate["ledger_entry"][field] = 999
            invalid.append(candidate)
        for field, value in (("ledger_entry", None), ("insights", "wrong"),
                             ("bottleneck_now", {}), ("worst_cases_tracker", []),
                             ("suggest_next", []), ("fusion_kernel_strategy", [{"iter": 1}])):
            candidate = deepcopy(self.output)
            candidate[field] = value
            invalid.append(candidate)
        for value in ({"priority": "P3", "action": "change", "reason": "evidence"},
                      {"priority": "P0", "action": "", "reason": "evidence"},
                      {"priority": "P0", "action": "change"}):
            candidate = deepcopy(self.output)
            candidate["suggest_next"] = [value]
            invalid.append(candidate)
        for field, value in (("direction", " "), ("modify_files", "kernel.py"),
                             ("readonly_files", [None]), ("fix_plan", []),
                             ("readonly_files", ["./impl\\kernel.py"])):
            candidate = deepcopy(self.output)
            candidate["ledger_entry"][field] = value
            invalid.append(candidate)
        original = deepcopy(self.base)
        for candidate in invalid:
            with self.subTest(output=candidate), self.assertRaises(ValueError):
                self.merge(candidate)
            self.assertEqual(self.base, original)

    def test_invalid_optional_experience_and_pitfall_are_rejected(self):
        values = [
            ("proven_pattern", None),
            ("proven_pattern", {"what_changed": "tile", "why_it_worked": "padding", "fusion_related": "false"}),
            ("regression_pattern", {"what_changed": "tile", "fusion_related": True}),
            ("pitfall", {"verdict": "partial"}),
            ("pitfall", {"verdict": "confirmed", "topic": "missing evidence"}),
        ]
        for field, value in values:
            candidate = deepcopy(self.output)
            candidate[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.merge(candidate)


if __name__ == "__main__":
    unittest.main()

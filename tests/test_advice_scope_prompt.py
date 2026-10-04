from pathlib import Path
import tempfile
import unittest

from lib.history_manager import format_for_prompt, get_latest_fix_plan, load_history, save_history


class AdviceScopePromptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)

    def save_advice(self, suggestions):
        history = load_history(self.work)
        history["suggest_next"] = suggestions
        history["ledger"] = [{
            "iter": 2, "direction": "First check case20's actual execution path",
            "evaluation_summary": "case20 still has tail-block waiting", "reason": "perf_optimize",
            "modify_files": ["impl/c2.py"], "readonly_files": ["impl/c3.py"],
        }]
        save_history(self.work, history)

    def test_each_advice_displays_inspection_and_modification_separately(self):
        self.save_advice([
            {"priority": "P0", "action": "Check case20's load distribution; do not adjust the traversal order yet",
             "reason": "verify the root cause first", "inspect_files": ["impl/c3.py"], "modify_files": [],
             "source": "human", "human_message_id": "human-7"},
            {"priority": "P1", "action": "Adjust the fp32 tail block", "reason": "only affects c2",
             "inspect_files": ["impl/c2.py"], "modify_files": ["impl/c2.py"]},
        ])
        prompt = format_for_prompt(self.work)
        self.assertIn("[human feedback human-7]", prompt)
        self.assertIn("Inspect files (no modification rights): impl/c3.py", prompt)
        self.assertIn("Modify files (still within this round's allowed scope): none, inspect only", prompt)
        self.assertIn("Modify files (still within this round's allowed scope): impl/c2.py", prompt)
        self.assertNotIn("Modify files (still within this round's allowed scope): impl/c3.py", prompt)

    def test_legacy_advice_remains_visible_without_claiming_new_validation(self):
        for suggestions in ([{"priority": "P1", "action": "Check case20; consider adjusting the traversal order",
                               "reason": "the original suggestion had no per-item file list"}],
                            ["Check case20; consider adjusting the traversal order"],
                            "Check case20; consider adjusting the traversal order"):
            with self.subTest(suggestions=suggestions):
                self.save_advice(suggestions)
                prompt = format_for_prompt(self.work)
                self.assertIn("Check case20; consider adjusting the traversal order", prompt)
                self.assertIn("not validated under the new rules", prompt)
                self.assertIn("Stage9 must complete it", prompt)

    def test_empty_or_malformed_scope_is_not_presented_as_valid_inspection_only(self):
        for scope in ({"inspect_files": [], "modify_files": []},
                      {"inspect_files": "impl/c3.py", "modify_files": []},
                      {"inspect_files": [None], "modify_files": []}):
            with self.subTest(scope=scope):
                self.save_advice([{"priority": "P1", "action": "Check the load", "reason": "to be verified", **scope}])
                prompt = format_for_prompt(self.work)
                self.assertIn("not validated under the new rules", prompt)
                self.assertNotIn("none, inspect only", prompt)

    def test_plan_overview_and_research_notes_do_not_extend_execution_scope(self):
        self.save_advice([{"priority": "P1", "action": "Check the load", "reason": "to be verified",
                           "inspect_files": ["impl/c3.py"], "modify_files": []}])
        history = load_history(self.work)
        history["worst_cases_tracker"] = {"case20": {"iteration": 2,
            "explanation": "tail-block utilization to be verified", "next_action": "consider adjusting c3's traversal order"}}
        save_history(self.work, history)
        prompt = format_for_prompt(self.work)
        plan = get_latest_fix_plan(self.work)
        self.assertIn("suggest_next is the single list of this round's executable items", prompt)
        self.assertIn("Next actions are research leads only", prompt)
        self.assertIn("Direction overview (adds no execution directives or modification permissions)", plan)
        self.assertIn("Files that must not be modified (absolutely untouchable):\n  - impl/c3.py", plan)
        self.assertIn("old fix_plan never expand permissions by themselves", plan)


if __name__ == "__main__":
    unittest.main()

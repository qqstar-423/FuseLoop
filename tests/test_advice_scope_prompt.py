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
            "iter": 2, "direction": "先检查 case20 的实际执行路径",
            "evaluation_summary": "case20 仍有尾块等待", "reason": "perf_optimize",
            "modify_files": ["impl/c2.py"], "readonly_files": ["impl/c3.py"],
        }]
        save_history(self.work, history)

    def test_each_advice_displays_inspection_and_modification_separately(self):
        self.save_advice([
            {"priority": "P0", "action": "检查 case20 的负载分布，暂不调整遍历顺序",
             "reason": "先核对根因", "inspect_files": ["impl/c3.py"], "modify_files": [],
             "source": "human", "human_message_id": "human-7"},
            {"priority": "P1", "action": "调整 fp32 尾块", "reason": "只影响 c2",
             "inspect_files": ["impl/c2.py"], "modify_files": ["impl/c2.py"]},
        ])
        prompt = format_for_prompt(self.work)
        self.assertIn("[人工意见 human-7]", prompt)
        self.assertIn("检查文件（不授予修改权）: impl/c3.py", prompt)
        self.assertIn("修改文件（仍须在本轮允许范围内）: 无，仅检查", prompt)
        self.assertIn("修改文件（仍须在本轮允许范围内）: impl/c2.py", prompt)
        self.assertNotIn("修改文件（仍须在本轮允许范围内）: impl/c3.py", prompt)

    def test_legacy_advice_remains_visible_without_claiming_new_validation(self):
        for suggestions in ([{"priority": "P1", "action": "检查 case20，考虑调整遍历顺序",
                               "reason": "原始建议没有逐条文件列表"}],
                            ["检查 case20，考虑调整遍历顺序"],
                            "检查 case20，考虑调整遍历顺序"):
            with self.subTest(suggestions=suggestions):
                self.save_advice(suggestions)
                prompt = format_for_prompt(self.work)
                self.assertIn("检查 case20，考虑调整遍历顺序", prompt)
                self.assertIn("未按新规则校验", prompt)
                self.assertIn("需 Stage9 补全", prompt)

    def test_empty_or_malformed_scope_is_not_presented_as_valid_inspection_only(self):
        for scope in ({"inspect_files": [], "modify_files": []},
                      {"inspect_files": "impl/c3.py", "modify_files": []},
                      {"inspect_files": [None], "modify_files": []}):
            with self.subTest(scope=scope):
                self.save_advice([{"priority": "P1", "action": "检查负载", "reason": "待验证", **scope}])
                prompt = format_for_prompt(self.work)
                self.assertIn("未按新规则校验", prompt)
                self.assertNotIn("无，仅检查", prompt)

    def test_plan_overview_and_research_notes_do_not_extend_execution_scope(self):
        self.save_advice([{"priority": "P1", "action": "检查负载", "reason": "待验证",
                           "inspect_files": ["impl/c3.py"], "modify_files": []}])
        history = load_history(self.work)
        history["worst_cases_tracker"] = {"case20": {"iteration": 2,
            "explanation": "尾块利用率待验证", "next_action": "考虑调整 c3 遍历顺序"}}
        save_history(self.work, history)
        prompt = format_for_prompt(self.work)
        plan = get_latest_fix_plan(self.work)
        self.assertIn("suggest_next 是本轮可执行事项的唯一清单", prompt)
        self.assertIn("后续动作只作研究线索", prompt)
        self.assertIn("方向概览（不追加执行指令或修改权限）", plan)
        self.assertIn("不可修改的文件（绝对不能动）:\n  - impl/c3.py", plan)
        self.assertIn("旧 fix_plan 均不能自行扩大权限", plan)


if __name__ == "__main__":
    unittest.main()

"""Stage3 must receive the executable v2 task, not just its short title."""
import copy
import tempfile
import unittest

from lib.history_manager import (
    format_for_prompt, get_latest_fix_plan, load_history, save_history,
    update_from_tech_lead,
)


class Stage9PlanPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = self.tmp.name
        self.task = {
            "task_id": "T7", "priority": "P0", "task_type": "modify",
            "action": "修复目标尾块", "reason": "精度报告指向尾块 mask",
            "case_scope": "cases", "target_cases": ["float32_case7"],
            "operator_reason": "",
            "case_bindings": [{
                "case_id": "float32_case7", "implementation_files": ["impl/c2.py"],
                "route_evidence": [{"file": "impl/dispatcher.py", "location": "dispatch.float32",
                                    "explanation": "输入dtype选择c2"}],
            }],
            "changes": [{"file": "impl/c2.py", "operation": "modify", "location": "kernel.tail",
                         "method": "按有效元素数屏蔽加载和写回"},
                        {"file": "task/golden.py", "operation": "inspect", "location": "reference",
                         "method": "确认边界元素定义，不修改参考实现"}],
            "acceptance_checks": ["运行全部给定case精度自测并留日志", "对照连续调用结果"],
            "source": "human", "human_message_id": "human-007",
        }
        self.history = {
            "suggest_next": [dict(self.task, inspect_files=["task/golden.py"], modify_files=["impl/c2.py"])],
            "ledger": [{"iter": 3, "reason": "precision_fail", "evaluation_summary": "本轮尾块错误",
                        "direction": "优先恢复正确性", "modify_files": ["impl/c2.py"],
                        "readonly_files": ["task/golden.py"],
                        "action_plan": {"version": 2, "tasks": [copy.deepcopy(self.task)]}}],
        }
        save_history(self.work, self.history)

    def test_v2_renders_case_route_each_method_and_acceptance(self):
        prompt = format_for_prompt(self.work)
        for expected in ("T7", "float32_case7", "impl/c2.py", "impl/dispatcher.py",
                         "dispatch.float32", "输入dtype选择c2", "kernel.tail",
                         "按有效元素数屏蔽加载和写回", "[检查，不修改] task/golden.py",
                         "运行全部给定case精度自测并留日志", "对照连续调用结果",
                         "[人工意见 human-007]"):
            self.assertIn(expected, prompt)
        self.assertNotIn("旧版任务仅作回顾", prompt)

    def test_current_readonly_is_never_truncated(self):
        paths = [f"impl/readonly_{index}.py" for index in range(8)]
        self.history["ledger"][-1]["readonly_files"] = paths
        save_history(self.work, self.history)
        for prompt in (format_for_prompt(self.work), get_latest_fix_plan(self.work)):
            for path in paths:
                self.assertIn(path, prompt)

    def test_operator_inspection_uses_reason_without_inventing_cases(self):
        task = copy.deepcopy(self.task)
        task.update(task_type="inspect", case_scope="operator", target_cases=[], case_bindings=[],
                    operator_reason="构建入口错误影响算子整体安装",
                    changes=[{"file": "impl/setup.py", "operation": "inspect", "location": "setup",
                              "method": "检查包名与导出配置"}])
        self.history["suggest_next"] = [task]
        self.history["ledger"][-1]["action_plan"]["tasks"] = [task]
        self.history["ledger"][-1]["modify_files"] = []
        self.history["ledger"][-1]["readonly_files"] = []
        save_history(self.work, self.history)
        prompt = format_for_prompt(self.work)
        self.assertIn("构建入口错误影响算子整体安装", prompt)
        self.assertIn("检查包名与导出配置", prompt)
        self.assertNotIn("目标 case（完整编号）:", prompt)
        self.assertIn("可修改的文件: 无", get_latest_fix_plan(self.work))

    def test_legacy_scoped_advice_is_readable_but_requires_v2(self):
        self.history["suggest_next"] = [{"priority": "P1", "action": "检查尾块",
                                          "reason": "旧建议", "inspect_files": ["impl/c2.py"],
                                          "modify_files": []}]
        self.history["ledger"][-1].pop("action_plan")
        save_history(self.work, self.history)
        self.assertIn("旧版任务仅作回顾", format_for_prompt(self.work))
        self.assertIn("恢复执行前需 Stage9 重新填写", get_latest_fix_plan(self.work))

    def test_background_does_not_authorize_additional_changes(self):
        prompt = format_for_prompt(self.work) + get_latest_fix_plan(self.work)
        self.assertIn("case_analysis.next_action", prompt)
        self.assertIn("不额外授权修改文件", prompt)
        self.assertIn("不是模型另填的一份授权", prompt)

    def test_versioned_decision_cannot_bypass_validation_via_legacy_helper(self):
        before = load_history(self.work)
        with self.assertRaisesRegex(ValueError, "merge_tech_lead_update"):
            update_from_tech_lead(self.work, {"plan_version": 2, "suggest_next": []}, 3)
        self.assertEqual(before, load_history(self.work))
        update_from_tech_lead(self.work, {"bottleneck_now": "legacy compatible"}, 3)
        self.assertEqual("legacy compatible", load_history(self.work)["bottleneck_now"])

    def test_explicit_iteration_uses_matching_task_and_scope_in_unordered_history(self):
        old = copy.deepcopy(self.history["ledger"][0])
        old.update(iter=1, modify_files=["impl/old.py"], readonly_files=["impl/old_readonly.py"])
        old["action_plan"]["tasks"][0]["task_id"] = "OLD_TASK"
        self.history["ledger"].append(old)
        self.history["suggest_next"] = old["action_plan"]["tasks"]
        save_history(self.work, self.history)
        prompt = format_for_prompt(self.work, max_ledger=1, current_iteration=3)
        self.assertIn("[P0] T7", prompt)
        self.assertNotIn("[P0] OLD_TASK", prompt)
        self.assertIn("当前计划可修改: impl/c2.py", prompt)
        self.assertNotIn("当前计划可修改: impl/old.py", prompt)
        plan = get_latest_fix_plan(self.work, current_iteration=3)
        self.assertIn("task/golden.py", plan)
        self.assertNotIn("impl/old.py", plan)
        self.assertNotIn("最后一", prompt + plan)
        self.assertEqual("", get_latest_fix_plan(self.work, current_iteration=9))
        missing = format_for_prompt(self.work, current_iteration=9)
        self.assertNotIn("[P0] OLD_TASK", missing)
        self.assertNotIn("当前计划可修改:", missing)


if __name__ == "__main__":
    unittest.main()

"""Actual Stage9 requests retain scene boundaries when human feedback is added."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
from lib.history_manager import load_history
import test_human_routing as human_routing


SCENE_MARKERS = {
    "compile": "当前场景：编译失败",
    "precision": "当前场景：精度失败",
    "evaluation_error": "当前场景：零分或评测异常",
    "optimization": "当前场景：未全部达标，普通性能优化",
    "stagnation": "当前场景：未全部达标，停滞审查",
    "all_passed": "当前场景：全部 case 达标，继续优化或收尾",
}


class HumanSceneMatrixTests(unittest.TestCase):
    def setUp(self):
        self.flow = human_routing.HumanRoutingTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.work = self.flow.work

    def read_request(self, path):
        request = json.loads(path.read_text(encoding="utf-8"))
        role = Path(request["role_path"]).read_text(encoding="utf-8")
        prompt = Path(request["prompt_path"]).read_text(encoding="utf-8")
        return request, role, prompt

    def assert_single_scene(self, scene, role):
        self.assertIn(SCENE_MARKERS[scene], role)
        for other, marker in SCENE_MARKERS.items():
            if other != scene:
                self.assertNotIn(marker, role)

    def assert_proactive_scene(self, scene):
        flow = self.flow
        target = 2 if scene == "stagnation" else 1
        flow.routing.perfs = {1: (0.8,), 2: (0.8,)}
        if scene == "compile":
            flow.routing.build_failures = {1}
        elif scene == "precision":
            flow.routing.precision_failures = {1}
        elif scene == "evaluation_error":
            flow.routing.perfs = {1: (0.0,)}
        elif scene == "all_passed":
            flow.routing.perfs = {1: (1.6,)}
        elif scene == "stagnation":
            flow.routing.config["workflow"]["semantic_exit"]["underperforming_window"] = 1
        sent = []

        def arriving(stage, iteration, _prompt):
            if stage == "stage4" and iteration == target and not sent:
                sent.append(flow.submit("先保持当前融合方向，按实际失败原因修复"))

        flow.on_stage = arriving
        flow.run_flow(target)
        message_id = sent[0]["id"]
        decision = next(item for item in flow.decisions
                        if item[0] == target and message_id in item[1])
        request, role, prompt = self.read_request(
            Path(decision[2]["decision_path"]).parent / "request.json")
        self.assertEqual(request["scene"], scene)
        self.assertEqual(request["phase"], "feedback")
        self.assert_single_scene(scene, role)
        self.assertIn("人工意见处理：显著关联", role)
        self.assertIn(message_id, prompt)
        self.assertIn("P0（人工建议）", flow.fixture.prompts["stage3"])
        self.assertEqual(flow.human.all_messages()[0]["status"], "executed")
        payload = json.loads(Path(request["decision_path"]).read_text(encoding="utf-8"))
        self.assertTrue(any(item.get("priority") == "P0" and item.get("source") == "human"
                            and item.get("human_message_id") == message_id
                            for item in payload["suggest_next"]))
        inputs = {item["key"] for item in request["default_inputs"] + request["missing_inputs"]}
        if scene in {"compile", "precision"}:
            self.assertFalse(inputs & {"perf_result", "profiler", "bottleneck", "fix_directive"})
            self.assertNotIn("条件任务：详细记录本轮性能经验", role)
            self.assertNotIn("性能证据口径", role)
        elif scene in {"evaluation_error", "all_passed"}:
            self.assertFalse(inputs & {"bottleneck", "fix_directive", "search_report"})
            self.assertIn("perf_result", inputs)
        else:
            self.assertTrue({"bottleneck", "fix_directive", "perf_result"} <= inputs)
        self.assertEqual(flow.questions, [], "A proactive opinion must not create an automatic consultation")

    def test_proactive_feedback_keeps_compile_scene(self):
        self.assert_proactive_scene("compile")

    def test_proactive_feedback_keeps_precision_scene(self):
        self.assert_proactive_scene("precision")

    def test_proactive_feedback_keeps_evaluation_error_scene(self):
        self.assert_proactive_scene("evaluation_error")

    def test_proactive_feedback_keeps_ordinary_optimization_scene(self):
        self.assert_proactive_scene("optimization")

    def test_proactive_feedback_keeps_stagnation_scene(self):
        self.assert_proactive_scene("stagnation")

    def test_proactive_feedback_keeps_all_passed_scene(self):
        self.assert_proactive_scene("all_passed")

    def prepare_consultation(self):
        flow = self.flow
        flow.routing.config["workflow"]["human_review"]["consultation_enabled"] = True
        flow.routing.config["workflow"]["semantic_exit"]["underperforming_window"] = 1
        flow.routing.perfs = {1: (0.8,), 2: (0.8,), 3: (0.7,), 4: (0.8,)}

    def test_consultation_defers_pattern_and_pitfall_until_feedback(self):
        self.prepare_consultation()
        flow = self.flow
        question = self.work / "develop/iter3/question.md"

        def arriving(stage, iteration, _prompt):
            if stage == "stage4" and iteration == 4:
                question.write_text("# 开发异议\n强制扩大分块使尾块开销增加。\n", encoding="utf-8")

        flow.on_stage = arriving

        def final_decision(_output, payload, iteration, _prompt):
            if iteration == 4:
                self.assertNotIn("## tech_lead 裁定", question.read_text(encoding="utf-8"))
                self.assertNotIn("stage9_decision_path", load_history(str(self.work))["ledger"][-1])
                payload["pitfall"] = {
                    "verdict": "confirmed", "topic": "尾块分块",
                    "target_advice": "强制扩大分块", "feedback": "尾块填充开销增加",
                    "root_cause": "未考虑尾块比例", "correct_approach": "按 shape 验证分块大小",
                }

        flow.on_decision = final_decision

        def reply():
            flow.sleep_hook = None
            flow.submit("选 A，保留方案并修复慢 case 的分块")

        flow.sleep_hook = reply
        flow.run_flow(4)
        requests = [self.read_request(path) for path in
                    (self.work / "knowledge/stage9/iter4").glob("*/request.json")]
        by_phase = {request["phase"]: (request, role, prompt) for request, role, prompt in requests}
        self.assertEqual(set(by_phase), {"consultation", "feedback"})
        for request, role, _prompt in requests:
            self.assertEqual(request["scene"], "stagnation")
            self.assert_single_scene("stagnation", role)
        consulting, consult_role, consult_prompt = by_phase["consultation"]
        self.assertFalse(Path(consulting["decision_path"]).exists())
        for excluded in ("当前阶段：提交最终决策", "条件任务：详细记录本轮性能经验",
                         "条件任务：裁定开发节点"):
            self.assertNotIn(excluded, consult_role)
        self.assertNotIn("仅向本次 decision 提交当前轮账本", consult_prompt)
        feedback, feedback_role, _prompt = by_phase["feedback"]
        for required in ("当前阶段：提交最终决策", "条件任务：详细记录本轮性能经验",
                         "条件任务：裁定开发节点", "人工意见处理：显著关联"):
            self.assertIn(required, feedback_role)
        self.assertTrue(feedback["perf_diff"]["has_improvement"])
        self.assertEqual(feedback["question_path"], str(question))
        self.assertIn("## tech_lead 裁定", question.read_text(encoding="utf-8"))
        self.assertIn("## iter4:", (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8"))
        self.assertTrue((self.work / "knowledge/tech_lead_pitfalls.md").is_file())
        self.assertEqual(len(load_history(str(self.work))["rounds"]), 4)
        self.assertEqual(flow.human.all_messages()[0]["status"], "executed")

    def test_wait_recovery_keeps_saved_scene_when_live_selection_is_unavailable(self):
        self.prepare_consultation()
        flow = self.flow

        def interrupt():
            flow.sleep_hook = None
            raise RuntimeError("interrupt consultation for scene recovery")

        flow.sleep_hook = interrupt
        with self.assertRaisesRegex(RuntimeError, "interrupt consultation"):
            flow.run_flow(4)
        active = flow.human.active_consultation()
        self.assertEqual(flow.fixture.read_state()["stage9_context"]["scene"], "stagnation")
        flow.submit("选 A，先继续局部优化")
        unavailable = deepcopy(flow.routing.selection_status())
        unavailable.update(eligible=False, review_fusion=False, should_exit=False,
                           reason="Synthetic temporarily unavailable selection evidence")
        with patch.object(orchestrator, "load_selection_status", return_value=unavailable):
            flow.run_flow()
        request = flow.decisions[-1][2]
        self.assertEqual(request["scene"], "stagnation")
        self.assertEqual(request["fail_reason"], active["fail_reason"])
        self.assertEqual(request["phase"], "feedback")
        self.assertEqual(flow.fixture.events.count("stage6"), 4)
        self.assertEqual(len(flow.questions), 1)
        completed = flow.human.get_consultation(active["request_id"])
        self.assertEqual(completed["deadline"], active["deadline"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(flow.human.state()["stagnation"]["seen_iterations"], [1, 2, 3, 4])

    def test_passed_recovery_rechecks_archive_when_live_report_is_missing(self):
        flow = self.flow
        flow.routing.perfs = {1: (2.0,), 2: (1.8,)}
        flow.routing.interrupt_stage9_iteration = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            flow.run_flow(2)
        saved = flow.fixture.read_state()["stage9_context"]
        self.assertEqual(saved["scene"], "all_passed")
        self.assertTrue(saved["perf_diff"]["has_regression"])
        (self.work / "eval/iter2/perf_result.json").unlink()
        flow.routing.interrupt_stage9_iteration = None
        flow.submit("保留达标方案，检查本轮退步原因")
        flow.run_flow()
        final = flow.decisions[-1][2]
        request, role, prompt = self.read_request(
            Path(final["decision_path"]).parent / "request.json")
        self.assertEqual(request["scene"], "all_passed")
        self.assertEqual(request["fail_reason"], saved["fail_reason"])
        comparison = request["perf_diff"]
        self.assertTrue(comparison["comparable"])
        self.assertTrue(comparison["has_regression"])
        self.assertEqual(comparison["delta_pct"], saved["perf_diff"]["delta_pct"])
        self.assertEqual(comparison["current_report_source"], "verified_selection_archive")
        self.assertIn("selection/records/", comparison["current_report"].replace("\\", "/"))
        self.assertFalse((self.work / "eval/iter2/perf_result.json").exists())
        self.assertIn("已验证性能归档", prompt)
        self.assert_single_scene("all_passed", role)
        self.assertIn("保留必要瓶颈分析", role)
        self.assertIn("条件任务：详细记录本轮性能经验", role)
        self.assertIn("perf_result", {item["key"] for item in request["missing_inputs"]})
        self.assertIn("缺失", prompt)
        self.assertIn("## iter2:", (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8"))
        self.assertEqual(flow.fixture.events.count("stage6"), 2)
        self.assertNotIn("stage7", flow.fixture.events)
        self.assertNotIn("stage8", flow.fixture.events)
        self.assertEqual(flow.human.all_messages()[0]["status"], "executed")


if __name__ == "__main__":
    unittest.main()

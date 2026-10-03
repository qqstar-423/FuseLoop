"""Scene boundaries prevent unavailable evidence and unrelated tasks being required."""

from pathlib import Path
import unittest

from lib.stage9_scenes import (SCENES, build_scene_role, classify_scene,
                               is_performance_scene, scene_input_keys)


ROLES = Path(__file__).resolve().parents[1] / "roles"


class Stage9ScenesTests(unittest.TestCase):
    def test_original_failure_wins_over_old_success_or_stagnation(self):
        for old in ({"eligible": True, "review_fusion": True},
                    {"eligible": True, "window": {"status": "passed"}}):
            for reason, expected in (("build_fail", "compile"),
                                     ("precision_fail", "precision"),
                                     ("score_zero: missing report\nold anticheat text", "evaluation_error")):
                with self.subTest(reason=reason, old=old):
                    self.assertEqual(classify_scene(reason, old, {"all_cases_pass": True}), expected)

    def test_only_valid_stagnation_requests_focus_review(self):
        self.assertEqual(classify_scene("perf_optimize", {"eligible": True, "review_fusion": True}),
                         "stagnation")
        self.assertEqual(classify_scene("perf_optimize", {"eligible": False, "review_fusion": True}),
                         "optimization")
        self.assertEqual(classify_scene("perf_optimize", {"eligible": True, "review_fusion": False}),
                         "optimization")

    def test_passed_route_and_evaluation_errors_are_distinct(self):
        self.assertEqual(classify_scene("perf_pass_semantic_review\nexit directive"), "all_passed")
        self.assertEqual(classify_scene("", metrics={"score_error_code": "missing_report"}),
                         "evaluation_error")
        self.assertEqual(classify_scene("", metrics={"avg_speedup": 0}), "evaluation_error")
        self.assertEqual(classify_scene("", metrics={"all_cases_pass": True}), "all_passed")
        self.assertEqual(classify_scene(""), "optimization")

    def test_failure_default_inputs_do_not_require_unproduced_performance(self):
        self.assertEqual(scene_input_keys("compile"), {"build_log", "self_test"})
        self.assertIn("precision_reports", scene_input_keys("precision"))
        for scene in ("compile", "precision"):
            self.assertFalse(is_performance_scene(scene))
            self.assertFalse(scene_input_keys(scene) & {"perf_result", "bottleneck", "fix_directive",
                                                        "fusion_library", "profiler"})

    def test_all_passed_retains_profiling_without_new_stage7_or_stage8(self):
        keys = scene_input_keys("all_passed")
        self.assertTrue({"perf_result", "profiler", "perf_reports", "fusion_library"} <= keys)
        self.assertFalse(keys & {"bottleneck", "fix_directive", "search_report"})
        role = build_scene_role(ROLES, "all_passed")
        self.assertIn("保留必要瓶颈分析", role)
        self.assertNotIn("profile/<iter>/bottleneck_analysis.md", role)
        self.assertNotIn("search/<iter>/FIX_DIRECTIVE.md", role)

    def test_actual_role_contains_one_scene_not_all_scene_tasks(self):
        markers = {
            "compile": "当前场景：编译失败", "precision": "当前场景：精度失败",
            "evaluation_error": "当前场景：零分或评测异常",
            "optimization": "当前场景：未全部达标，普通性能优化",
            "stagnation": "当前场景：未全部达标，停滞审查",
            "all_passed": "当前场景：全部 case 达标，继续优化或收尾",
        }
        for scene in SCENES:
            with self.subTest(scene=scene):
                role = build_scene_role(ROLES, scene)
                self.assertIn(markers[scene], role)
                for other, marker in markers.items():
                    if other != scene:
                        self.assertNotIn(marker, role)
                self.assertIn("当前阶段：提交最终决策", role)

    def test_compile_precision_roles_do_not_request_slow_case_or_candidate_comparison(self):
        for scene in ("compile", "precision"):
            role = build_scene_role(ROLES, scene, perf_diff={"has_improvement": True})
            self.assertNotIn("最慢的 6 个 case", role)
            self.assertNotIn("fusion/fusion_library.json", role)
            self.assertNotIn("条件任务：详细记录本轮性能经验", role)
            self.assertNotIn("skills/triton-profiling-analysis/SKILL.md", role)

    def test_patterns_only_required_by_valid_measured_performance(self):
        for scene in ("optimization", "stagnation", "all_passed"):
            baseline = build_scene_role(ROLES, scene)
            self.assertNotIn("条件任务：详细记录本轮性能经验", baseline)
            for trigger in ("has_improvement", "has_regression"):
                role = build_scene_role(ROLES, scene, perf_diff={trigger: True})
                self.assertIn("条件任务：详细记录本轮性能经验", role)
                self.assertIn("单轮平均加速比提升 ≥5%", role)
                self.assertIn("退步 ≥5%", role)

    def test_technical_dispute_is_conditional_and_available_in_failure_scenes(self):
        for scene in SCENES:
            self.assertNotIn("条件任务：裁定开发节点", build_scene_role(ROLES, scene))
            role = build_scene_role(ROLES, scene, has_question=True)
            self.assertIn("条件任务：裁定开发节点", role)
            self.assertIn("root_cause", role)

    def test_consultation_never_requires_final_decision_even_if_experience_due(self):
        role = build_scene_role(ROLES, "stagnation", phase="consultation",
                                has_question=True, has_human=True,
                                perf_diff={"has_regression": True})
        self.assertIn("当前阶段：只生成咨询问题", role)
        self.assertIn("recommended_option", role)
        self.assertNotIn("当前阶段：提交最终决策", role)
        self.assertNotIn("条件任务：详细记录本轮性能经验", role)
        self.assertNotIn("条件任务：裁定开发节点", role)
        self.assertNotIn("必填字段", role)

    def test_proactive_and_feedback_roles_preserve_p0_and_reply_identity(self):
        for phase, has_human in (("decision", True), ("feedback", False)):
            role = build_scene_role(ROLES, "precision", phase=phase, has_human=has_human)
            self.assertIn("human_responses", role)
            self.assertIn('source="human"', role)
            self.assertIn("human_message_id", role)
            self.assertIn("alternative", role)
            self.assertIn("不能降成 P1/P2", role)
            self.assertIn("未收到人工意见", role)

    def test_invalid_scene_or_phase_fails_before_role_loading(self):
        for scene in ("other", "../compile"):
            with self.assertRaises(ValueError):
                build_scene_role(ROLES, scene)
            with self.assertRaises(ValueError):
                scene_input_keys(scene)
        with self.assertRaises(ValueError):
            build_scene_role(ROLES, "compile", phase="consultation")
        with self.assertRaises(ValueError):
            build_scene_role(ROLES, "stagnation", phase="unrecognized")


if __name__ == "__main__":
    unittest.main()

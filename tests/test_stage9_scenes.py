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
        self.assertIn("Keep the necessary bottleneck analysis", role)
        self.assertNotIn("profile/<iter>/bottleneck_analysis.md", role)
        self.assertNotIn("search/<iter>/FIX_DIRECTIVE.md", role)

    def test_actual_role_contains_one_scene_not_all_scene_tasks(self):
        markers = {
            "compile": "Current Scenario: Build Failure", "precision": "Current Scenario: Precision Failure",
            "evaluation_error": "Current Scenario: Zero Score or Evaluation Anomaly",
            "optimization": "Current Scenario: Not All Cases Meet Target, Ordinary Performance Optimization",
            "stagnation": "Current Scenario: Not All Cases Meet Target, Stagnation Review",
            "all_passed": "Current Scenario: All Cases Meet Target, Continue Optimizing or Wrap Up",
        }
        for scene in SCENES:
            with self.subTest(scene=scene):
                role = build_scene_role(ROLES, scene)
                self.assertIn(markers[scene], role)
                for other, marker in markers.items():
                    if other != scene:
                        self.assertNotIn(marker, role)
                self.assertIn("Current Stage: Submit Final Decision", role)

    def test_compile_precision_roles_do_not_request_slow_case_or_candidate_comparison(self):
        for scene in ("compile", "precision"):
            role = build_scene_role(ROLES, scene, perf_diff={"has_improvement": True})
            self.assertNotIn("six slowest", role)
            self.assertNotIn("fusion/fusion_library.json", role)
            self.assertNotIn("## Conditional Task: Record This Round's Performance Experience in Detail", role)
            self.assertNotIn("knowledge/profiling_guide.md", role)

    def test_patterns_only_required_by_valid_measured_performance(self):
        for scene in ("optimization", "stagnation", "all_passed"):
            baseline = build_scene_role(ROLES, scene)
            self.assertNotIn("## Conditional Task: Record This Round's Performance Experience in Detail", baseline)
            for trigger in ("has_improvement", "has_regression"):
                role = build_scene_role(ROLES, scene, perf_diff={trigger: True})
                self.assertIn("## Conditional Task: Record This Round's Performance Experience in Detail", role)
                self.assertIn("average speedup gain of ≥5%", role)
                self.assertIn("regression of ≥5%", role)

    def test_technical_dispute_is_conditional_and_available_in_failure_scenes(self):
        for scene in SCENES:
            self.assertNotIn("## Conditional Task: Adjudicate the Developer Node", build_scene_role(ROLES, scene))
            role = build_scene_role(ROLES, scene, has_question=True)
            self.assertIn("## Conditional Task: Adjudicate the Developer Node", role)
            self.assertIn("root_cause", role)

    def test_invalid_scene_fails_before_role_loading(self):
        for scene in ("other", "../compile"):
            with self.assertRaises(ValueError):
                build_scene_role(ROLES, scene)
            with self.assertRaises(ValueError):
                scene_input_keys(scene)


if __name__ == "__main__":
    unittest.main()

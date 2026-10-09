"""Inspect the actual persisted role, prompt and input manifest sent to Stage9."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import orchestrator
from lib.stage9_scenes import classify_scene
from lib.state import State
from lib.history_manager import load_history
from lib.human_review import HumanReview
import test_semantic_routing as semantic


class Stage9SceneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.flow = semantic.SemanticRoutingTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.work = self.flow.work
        self.fixture = self.flow.fixture
        self.flow.perfs = {1: (0.8,), 2: (0.8,), 3: (0.8,)}
        self.flow.config["workflow"]["semantic_exit"]["passed_window"] = 8

    def plant_stale_inputs(self, iteration):
        for path in (f"profile/iter{iteration}/bottleneck_analysis.md",
                     f"search/iter{iteration}/FIX_DIRECTIVE.md", f"search/iter{iteration}/SEARCH_REPORT.md"):
            target = self.work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("STALE_ANALYSIS_FROM_AN_OLDER_RUN", encoding="utf-8")
        self.fixture.write_json(f"eval/iter{iteration}/perf_result.json", {"avg_speedup": 99, "perf_pass": True})
        for folder in ("perf_reports", "precision_reports", "prof_data"):
            (self.work / f"eval/iter{iteration}" / folder).mkdir(parents=True, exist_ok=True)

    def produce_fresh_profiler(self):
        original = self.fixture.performance

        def performance(*args, **kwargs):
            result = original(*args, **kwargs)
            csv = Path(result[2]).parent / "prof_data/synthetic_op/1/session/kernel_details.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text("Name,Duration(us)\ncurrent_kernel,1\n", encoding="utf-8")
            return result

        self.fixture.performance = performance

    def artifacts(self, iteration):
        files = sorted((self.work / f"knowledge/stage9/iter{iteration}").glob("*/request.json"),
                       key=lambda path: path.stat().st_mtime_ns)
        path = files[-1]
        request = json.loads(path.read_text(encoding="utf-8"))
        role = Path(request["role_path"]).read_text(encoding="utf-8")
        prompt = Path(request["prompt_path"]).read_text(encoding="utf-8")
        return request, role, prompt

    def assert_scene(self, scene, *, iteration=1, expected_keys=(), forbidden_keys=()):
        request, role, prompt = self.artifacts(iteration)
        self.assertEqual(request["scene"], scene)
        self.assertEqual(request["phase"], "decision")
        self.assertEqual(role.count("## Current Scenario:"), 1)
        self.assertIn("## Current Stage: Submit Final Decision", role)
        keys = {item["key"] for item in request["default_inputs"]}
        self.assertTrue(set(expected_keys) <= keys, keys)
        self.assertFalse(set(forbidden_keys) & keys, keys)
        for item in request["default_inputs"]:
            self.assertTrue(Path(item["path"]).exists())
            self.assertIn(item["path"], prompt)
            self.assertIn(item["purpose"], prompt)
        if scene in {"compile", "precision", "evaluation_error", "all_passed"}:
            for relative in (f"profile/iter{iteration}/bottleneck_analysis.md",
                             f"search/iter{iteration}/FIX_DIRECTIVE.md", f"search/iter{iteration}/SEARCH_REPORT.md"):
                self.assertNotIn(str(self.work / relative), prompt)
        if scene in {"compile", "precision", "evaluation_error"}:
            self.assertNotIn("## Conditional Task: Record This Round's Performance Experience in Detail", role)
            self.assertNotIn("## Performance Evidence Basis", role)
            self.assertEqual(request["perf_diff"], {})
        return request, role, prompt

    def test_compile_actual_prompt_omits_existing_performance_and_search(self):
        self.plant_stale_inputs(1)
        self.flow.build_failures = {1}
        self.flow.run_workflow(max_iterations=1)
        _, _, prompt = self.assert_scene("compile", expected_keys={"build_log"},
                                        forbidden_keys={"perf_result", "precision_result", "bottleneck", "fix_directive", "profiler"})
        self.assertNotIn(str(self.work / "eval/iter1/perf_result.json"), prompt)

    def test_precision_actual_prompt_keeps_correctness_and_omits_old_performance(self):
        self.plant_stale_inputs(1)
        self.flow.precision_failures = {1}
        self.flow.run_workflow(max_iterations=1)
        _, _, prompt = self.assert_scene("precision", expected_keys={"build_log", "precision_result", "precision_reports"},
                                        forbidden_keys={"perf_result", "bottleneck", "fix_directive", "profiler"})
        self.assertNotIn(str(self.work / "eval/iter1/perf_result.json"), prompt)

    def test_zero_score_actual_prompt_does_not_assume_cheating_or_optimize(self):
        self.plant_stale_inputs(1)
        self.produce_fresh_profiler()
        self.flow.perfs[1] = (0.0,)
        self.flow.run_workflow(max_iterations=1)
        _, _, prompt = self.assert_scene("evaluation_error", expected_keys={"perf_result", "profiler"},
                                        forbidden_keys={"bottleneck", "fix_directive", "search_report"})
        self.assertIn("A missing report or zero score alone does not prove a violation", prompt)
        self.assertNotIn("Anti-cheating triggered! Your first task", prompt)

    def test_normal_optimization_actual_prompt_keeps_stage7_stage8(self):
        self.plant_stale_inputs(1)
        self.flow.run_workflow(max_iterations=1)
        _, role, prompt = self.assert_scene("optimization", expected_keys={"perf_result", "bottleneck", "fix_directive", "search_report"})
        self.assertNotIn("[Scenario 2 stagnation", prompt)
        self.assertNotIn("## Conditional Task: Record This Round's Performance Experience in Detail", role)
        guide = Path(__file__).resolve().parents[1] / "knowledge/profiling_guide.md"
        self.assertTrue(guide.is_file())
        self.assertIn(str(guide), self.fixture.prompts["stage7"])

    def test_stagnation_actual_prompt_keeps_trends_candidates_and_window(self):
        self.flow.run_workflow(max_iterations=3, file_logs=True)
        request, role, prompt = self.assert_scene("stagnation", iteration=3,
                                                expected_keys={"perf_result", "bottleneck", "fix_directive", "fusion_library"})
        self.assertIn("[Scenario 2 stagnation", prompt)
        self.assertIn('"case_trends"', prompt)
        self.assertIn("slow cases keep approaching 1", role)
        logs = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        for token in ("[Stage9 scene]", "scene=stagnation", "phase=decision", request["request_id"],
                      request["role_path"], request["prompt_path"]):
            self.assertIn(token, logs)

    def test_all_passed_actual_prompt_ignores_even_existing_stage7_stage8(self):
        self.plant_stale_inputs(1)
        self.produce_fresh_profiler()
        self.flow.perfs[1] = (1.6,)
        self.flow.run_workflow(max_iterations=1)
        _, role, prompt = self.assert_scene("all_passed", expected_keys={"perf_result", "profiler", "perf_reports"},
                                      forbidden_keys={"bottleneck", "fix_directive", "search_report"})
        self.assertIn("Keep the necessary bottleneck analysis", role)
        self.assertNotIn("profile/<iter>/bottleneck_analysis.md", role)
        guide = Path(__file__).resolve().parents[1] / "knowledge/profiling_guide.md"
        self.assertTrue(guide.is_file())
        self.assertIn(str(guide), prompt)
        self.assertIn("knowledge/profiling_guide.md", role)

    def test_failed_stage7_stops_before_search_and_review(self):
        original = self.fixture.agent
        def absent_agent(agent, role, work, prompt, **kwargs):
            if Path(role).name == "n2_stage7_kerminal_profile.md":
                return False
            return original(agent, role, work, prompt, **kwargs)
        self.fixture.agent = absent_agent
        with self.assertRaisesRegex(RuntimeError, "Stage7"):
            self.flow.run_workflow(max_iterations=1)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage7")
        self.assertFalse(any(stage in {"stage8", "stage9", "stage3"}
                             for _, stage, _ in self.flow.seen))
        self.assertFalse((self.work / "profile/iter1/bottleneck_analysis.md").exists())

    def test_missing_stage8_outputs_are_explicit_and_not_silently_assumed(self):
        original = self.fixture.agent
        def absent_agent(agent, role, work, prompt, **kwargs):
            if Path(role).name == "n3_stage8_search.md":
                return False
            return original(agent, role, work, prompt, **kwargs)
        self.fixture.agent = absent_agent
        self.flow.run_workflow(max_iterations=1)
        request, _, prompt = self.artifacts(1)
        self.assertTrue({"fix_directive", "search_report"} <=
                        {item["key"] for item in request["missing_inputs"]})
        self.assertIn("bottleneck", {item["key"] for item in request["default_inputs"]})
        self.assertIn("Materials missing for the current scene", prompt)
        self.assertIn("Stage8", prompt + json.dumps(request, ensure_ascii=False))

    def test_saved_stagnation_scene_survives_unavailable_live_selection(self):
        self.flow.interrupt_stage9_iteration = 3
        with self.assertRaises(RuntimeError):
            self.flow.run_workflow(max_iterations=3)
        state = State.load_or_create(str(self.work))
        self.assertEqual(state.stage9_context["scene"], "stagnation")
        self.flow.interrupt_stage9_iteration = None
        dirs = {name: str(self.work / name / "iter3") for name in ("build", "eval", "profile", "search", "develop")}
        with self.fixture.patches(), patch.object(orchestrator, "load_selection_status",
                return_value={"eligible": False, "window": None, "best": None}):
            orchestrator.run_tech_lead(Mock(), str(Path(orchestrator.__file__).parent / "roles"), str(self.work),
                                       "synthetic_op", state, Mock(), {"N4": Mock()}, dirs,
                                       fail_reason="perf_optimize", perf_diff={})
        request, role, _ = self.artifacts(3)
        self.assertEqual(request["scene"], "stagnation")
        self.assertIn("Current Scenario: Not All Cases Meet Target, Stagnation Review", role)
        self.assertEqual(HumanReview(str(self.work)).state()["stagnation"]["count"], 1)

    def test_actual_role_cannot_use_previous_iteration_stagnation(self):
        self.flow.run_workflow(max_iterations=1)
        state = State.load_or_create(str(self.work))
        state.iteration = 2
        state.stage9_context = {}
        self.fixture.write_json(".state.json", {**self.fixture.read_state(), "iteration": 2})
        dirs = {name: str(self.work / name / "iter2") for name in ("build", "eval", "profile", "search", "develop")}
        for directory in dirs.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        old = {"eligible": True, "latest_iteration": 1, "review_fusion": True, "window": {"status": "underperforming"}}
        with self.fixture.patches(), patch.object(orchestrator, "load_selection_status", return_value=old):
            orchestrator.run_tech_lead(Mock(), str(Path(orchestrator.__file__).parent / "roles"), str(self.work),
                                       "synthetic_op", state, Mock(), {"N4": Mock()}, dirs,
                                       fail_reason="perf_optimize", perf_diff={})
        request, role, _ = self.artifacts(2)
        self.assertEqual(request["scene"], "optimization")
        self.assertNotIn("Current Scenario: Not All Cases Meet Target, Stagnation Review", role)

    def test_classifier_respects_current_iteration_and_explicit_underperforming_route(self):
        for old in ({"eligible": True, "latest_iteration": 3, "review_fusion": True},
                    {"eligible": True, "latest_iteration": 3, "window": {"status": "passed"}}):
            self.assertEqual(classify_scene("perf_optimize", old, iteration=4), "optimization")
        self.assertEqual(classify_scene("perf_optimize", {"eligible": True, "latest_iteration": 4,
                                                          "window": {"status": "passed"}}, iteration=4), "optimization")

"""Offline workflow checks with real evidence, selection and Tech Lead bookkeeping."""

from contextlib import ExitStack
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import orchestrator
from lib.history_manager import load_history
import test_fusion_routing as fusion_routing_tests


class SemanticRoutingTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the synthetic transport fixture; all routing and storage run normally.
        self.fixture = fusion_routing_tests.FusionRoutingTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.work = self.fixture.work
        self.original_agent = self.fixture.agent
        self.fixture.agent = self.agent
        self.fixture.performance = self.performance
        self.seen = []
        self.perfs = {}
        self.precision_failures = set()
        self.stale_precision_responses = {}
        self.build_failures = set()
        self.interrupt_stage9_iteration = None
        self.stage9_callback = None
        self.config = {
            "paths": {"cannbench_repo": str(self.work / "absent_cannbench")},
            "fusion_selection": {"top_n": 2},
            "workflow": {"semantic_exit": {"passed_window": 2, "underperforming_window": 2}},
        }

    def write_json(self, path, value):
        self.fixture.write_json(path, value)

    def agent(self, agent, role, work, prompt, **kwargs):
        stage = fusion_routing_tests.ROLES[Path(role).name]
        iteration = self.fixture.read_state()["iteration"]
        self.seen.append((iteration, stage, prompt))
        if stage in {"stage4", "stage5"}:
            self.fixture.events.append(stage)
            self.fixture.prompts[stage] = prompt
            if stage == "stage4":
                status = "FAIL" if iteration in self.build_failures else "SUCCESS"
                (self.work / f"build/iter{iteration}/build.log").write_text(
                    f"STATUS: {status}\n", encoding="utf-8")
            else:
                if iteration in self.stale_precision_responses:
                    return self.stale_precision_responses[iteration]
                passed = iteration not in self.precision_failures
                total = len(self.perfs.get(iteration, (1.5,)))
                self.write_json(f"eval/iter{iteration}/precision_result.json", {
                    "precision_overall": passed, "total_cases": total,
                    "passed_cases": total if passed else 0, "failed_cases": 0 if passed else total,
                })
            return True

        if stage == "stage9":
            self.fixture.events.append(stage)
            self.fixture.prompts[stage] = prompt
            if iteration == self.interrupt_stage9_iteration:
                raise RuntimeError("synthetic interruption before Tech Lead returns")
            output, payload = fusion_routing_tests.stage9_payload(prompt, iteration)
            request = json.loads((output.parent / "request.json").read_text(encoding="utf-8"))
            perf_diff = request.get("perf_diff", {})
            if perf_diff.get("has_improvement"):
                payload["proven_pattern"] = {
                    "what_changed": "LOCAL_TILE_SUCCESS", "why_it_worked": "Less launch overhead",
                    "fusion_related": False,
                    "case_analysis": [{"case_id": "synthetic_op_1", "observation": "Latency fell",
                                       "explanation": "The same-shape tile eliminated redundant launches"}],
                    "evidence": [f"eval/iter{iteration}/perf_result.json"],
                    "applicability": "Small same-shape sigmoid cases with excessive launch overhead",
                    "next_action": "Keep the measured tile and evaluate neighboring sizes",
                }
            if perf_diff.get("has_regression"):
                payload["regression_pattern"] = {
                    "what_changed": "LOCAL_TILE_REGRESSION", "why_it_failed": "More padding overhead",
                    "fusion_related": False,
                    "case_analysis": [{"case_id": "synthetic_op_1", "observation": "Latency rose",
                                       "explanation": "The larger tile padded work past the input boundary"}],
                    "evidence": [f"eval/iter{iteration}/perf_result.json"],
                    "applicability": "Tail-heavy small shapes where padding exceeds useful work",
                    "next_action": "Restore the smaller tile before testing a different local change",
                }
            if self.stage9_callback is not None:
                return self.stage9_callback(output, payload, iteration, prompt)
            self.write_json(output, payload)
            return True

        result = self.original_agent(agent, role, work, prompt, **kwargs)
        if stage == "stage7":
            (self.work / f"profile/iter{iteration}/bottleneck_analysis.md").write_text(
                "The slow case is improving with local tile tuning.", encoding="utf-8")
        if stage == "stage8":
            (self.work / f"search/iter{iteration}/FIX_DIRECTIVE.md").write_text(
                "Compare local tile changes before replacing the fusion method.", encoding="utf-8")
        if stage in {"stage2", "stage3"}:
            (self.work / "impl/synthetic_op.py").write_text(
                f"# implementation revision {iteration}\n", encoding="utf-8")
            # This self-test describes the revision evaluated on the next loop.
            total = len(self.perfs.get(iteration + 1, (1.5,)))
            path = self.work / f"develop/iter{iteration}/self_test_result.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["provided_cases"].update(total=total, passed_cases=total)
            self.write_json(path, report)
        return result

    def performance(self, *_args, **_kwargs):
        iteration = self.fixture.read_state()["iteration"]
        self.fixture.events.append("stage6")
        speeds = self.perfs[iteration]
        report = self.work / f"cannbench_iter{iteration}.json"
        self.write_json(report, {"operators": [{
            "score": 90 if all(speed > 0 for speed in speeds) else 0,
            "performance_score": 12.75,
            "avg_speedup": sum(speeds) / len(speeds), "total_cases": len(speeds),
            "passed_cases": len(speeds), "failed_cases": 0,
            "cases": [{"case_id": f"synthetic_op_{index}", "status": "success",
                       "baseline_perf_us": 100, "elapsed_us": 100 / speed if speed > 0 else 200,
                       "speedup": speed, "perf_score": 0.4, "t_hw_us": 3.25}
                      for index, speed in enumerate(speeds, 1)],
        }]})
        return True, "", str(report)

    def run_workflow(self, *, max_iterations=None, file_logs=False):
        if max_iterations is not None:
            self.fixture.checkpoint(max_iterations=max_iterations)
        with self.fixture.patches(), ExitStack() as stack:
            if file_logs:
                # Real file handlers verify the visible output, beyond mocked log calls.
                (self.work / "log").mkdir(exist_ok=True)
                for factory, filename in (("setup_logger", "workflow.log"),
                                          ("setup_state_logger", "state_transitions.log")):
                    logger = logging.Logger(f"offline-{factory}", level=logging.DEBUG)
                    handler = logging.FileHandler(self.work / "log" / filename, encoding="utf-8")
                    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                    logger.addHandler(handler)
                    stack.callback(handler.close)
                    stack.enter_context(patch.object(orchestrator, factory, return_value=logger))
            with patch.object(orchestrator, "load_config", return_value=self.config):
                orchestrator.main()

    def selection_status(self):
        from lib.semantic_exit import load_selection_status
        return load_selection_status(str(self.work), config=self.config)

    def test_all_case_pass_continues_until_program_x_window(self):
        self.perfs = {1: (1.6,), 2: (1.59,), 3: (1.6,)}
        self.run_workflow(max_iterations=8, file_logs=True)
        state = self.fixture.read_state()
        self.assertEqual(state["iteration"], 3)
        self.assertNotEqual(state["stopped_by"], "max_iterations")
        self.assertEqual([iteration for iteration, stage, _ in self.seen if stage == "stage3"], [1, 2])
        self.assertEqual([iteration for iteration, stage, _ in self.seen if stage == "stage9"], [1, 2, 3])
        self.assertFalse(any(stage in {"stage7", "stage8"} for _, stage, _ in self.seen))
        best = self.selection_status()["best"]
        self.assertEqual(best["iteration"], 1)
        self.assertIn(best["implementation_dir"], self.fixture.prompts["stage10"])
        self.assertEqual((Path(best["implementation_dir"]) / "synthetic_op.py").read_text(encoding="utf-8"),
                         "# implementation revision 0\n")
        logs = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        for token in ("[best implementation] updated iter1", "[best implementation] kept iter1", "avg_speedup=1.6",
                      "avg_speed=1.6", "HAP.performance_score=12.75", '"perf_score": 0.4',
                      "all_cases_pass=True", "min_case_speedup=1.6", "fusion methods=['F2']",
                      best["implementation_dir"], best["performance_report"], best["performance_result"],
                      best["manifest_path"], best["evidence_paths"]["decision_rationale"]):
            self.assertIn(token, logs)
        self.assertNotIn("[Scenario 2 stagnation", logs)

    def test_iteration_limit_reports_best_evaluated_snapshot_and_keeps_live_revision(self):
        self.perfs = {1: (1.6,), 2: (1.4,)}
        self.run_workflow(max_iterations=2)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "max_iterations")
        best = self.selection_status()["best"]
        self.assertEqual(best["iteration"], 1)
        self.assertIn(best["implementation_dir"], self.fixture.prompts["stage10"])
        self.assertIn(json.dumps(best["performance_report"]), self.fixture.prompts["stage10"])
        self.assertEqual((Path(best["implementation_dir"]) / "synthetic_op.py").read_text(encoding="utf-8"),
                         "# implementation revision 0\n")
        self.assertEqual((self.work / "impl/synthetic_op.py").read_text(encoding="utf-8"),
                         "# implementation revision 2\n")

    def test_y_window_requests_review_and_keeps_slow_case_trends_in_stage9(self):
        self.perfs = {1: (0.683, 4.685), 2: (0.812, 4.588), 3: (0.977, 4.471)}
        self.run_workflow(max_iterations=3, file_logs=True)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "max_iterations")
        self.assertEqual([stage for iteration, stage, _ in self.seen if iteration == 3],
                         ["stage4", "stage5", "stage7", "stage8", "stage9", "stage3", "stage10"])
        status = self.selection_status()
        self.assertTrue(status["review_fusion"])
        self.assertFalse(status["should_exit"])
        prompt = self.fixture.prompts["stage9"]
        for token in ("synthetic_op_1", "0.683", "0.977", "F2 probability=0.8"):
            self.assertIn(token, prompt)
        self.assertIn("never mandatory fusion replacement", prompt)
        self.assertIn('"stagnated": true', prompt)
        notice = orchestrator.fusion_review_hint(status, 3)
        self.assertIn(notice, prompt)
        self.assertIn("y=2", notice)
        self.assertIn("iter1→iter3", notice)
        self.assertIn("1.49% < 5%", notice)
        self.assertIn("slow cases keep improving", notice)
        self.assertTrue(all("[Scenario 2 stagnation" not in text for iteration, stage, text in self.seen
                            if stage == "stage9" and iteration < 3))
        workflow_log = (self.work / "log/workflow.log").read_text(encoding="utf-8")
        state_log = (self.work / "log/state_transitions.log").read_text(encoding="utf-8")
        self.assertIn("[WARNING] ===== Scenario 2 stagnation trigger: entry 1 into this scene =====", state_log)
        self.assertIn("[WARNING] [Stage9 prompt addition]\n" + notice, workflow_log)
        inputs = workflow_log.split("[Stage9 fusion review inputs and evidence sources]\n", 1)[1].split("\n[", 1)[0]
        for path in (self.fixture.library_path, self.work / "develop/iter2/fusion_scheme_rationale.md",
                     self.work / "develop/iter2/fusion_library.json", self.work / "develop/iter2/self_test_report.md",
                     self.work / "selection/state.json", Path(status["best"]["manifest_path"]),
                     self.work / "profile/iter3/bottleneck_analysis.md", self.work / "search/iter3/FIX_DIRECTIVE.md"):
            self.assertIn(str(path).replace('\\', '/'), inputs.replace('\\', '/'))
        self.assertNotIn(str(self.work / "develop/iter3/fusion_scheme_rationale.md"), inputs)
        for stage in ("stage7", "stage8", "stage9"):
            self.assertIn(str(self.work / "develop/iter2/fusion_scheme_rationale.md"),
                          self.fixture.prompts[stage])
            self.assertIn("relative to working directory: `develop/iter2/fusion_scheme_rationale.md`", self.fixture.prompts[stage])
            self.assertIn("iteration template: `develop/<iter>/fusion_scheme_rationale.md`", self.fixture.prompts[stage])
            self.assertNotIn(str(self.work / "develop/iter3/fusion_scheme_rationale.md"),
                             self.fixture.prompts[stage])

    def test_resume_underperforming_stage9_keeps_review_notice_without_recounting(self):
        self.perfs = {1: (0.8, 2.0), 2: (0.81, 2.0), 3: (0.82, 2.0)}
        self.interrupt_stage9_iteration = 3
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.run_workflow(max_iterations=3)
        status_before = self.selection_status()
        self.interrupt_stage9_iteration = None
        self.run_workflow(file_logs=True)
        self.assertEqual(self.fixture.events.count("stage6"), 3)
        self.assertEqual(self.selection_status()["window"], status_before["window"])
        self.assertIn(orchestrator.fusion_review_hint(status_before, 3), self.fixture.prompts["stage9"])
        self.assertIn("[Stage9 fusion review inputs and evidence sources]",
                      (self.work / "log/workflow.log").read_text(encoding="utf-8"))

    def test_regression_and_improvement_still_write_knowledge_via_real_tech_lead(self):
        self.perfs = {1: (2.0,), 2: (1.9,), 3: (2.1,)}
        self.config["workflow"]["semantic_exit"]["passed_window"] = 8
        self.run_workflow(max_iterations=3)
        regression = (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8")
        proven = (self.work / "knowledge/proven_patterns.md").read_text(encoding="utf-8")
        self.assertIn("LOCAL_TILE_REGRESSION", regression)
        self.assertIn("More padding overhead", regression)
        self.assertIn("The larger tile padded work past the input boundary", regression)
        self.assertIn("Tail-heavy small shapes", regression)
        self.assertIn("Restore the smaller tile", regression)
        self.assertIn("-5.0", regression)
        self.assertIn("LOCAL_TILE_SUCCESS", proven)
        self.assertIn("Less launch overhead", proven)
        self.assertIn("The same-shape tile eliminated redundant launches", proven)
        self.assertIn("Small same-shape sigmoid cases", proven)
        self.assertIn("Keep the measured tile", proven)
        self.assertIn("10.5", proven)
        for report in (regression, proven):
            self.assertIn("Per-case analysis", report)
            self.assertIn("synthetic_op_1", report)
            self.assertIn("perf_result.json", report)
        self.assertNotIn("tech_lead not filled in", regression + proven)
        self.assertNotIn("to be analyzed", regression + proven)
        self.assertEqual([iteration for iteration, stage, _ in self.seen if stage == "stage9"], [1, 2, 3])
        history = load_history(str(self.work))
        self.assertEqual([entry["iter"] for entry in history["ledger"]], [1, 2, 3])
        self.assertEqual([entry["direction"] for entry in history["ledger"]],
                         ["LOCAL_TILE_ITER_1", "LOCAL_TILE_ITER_2", "LOCAL_TILE_ITER_3"])
        self.assertEqual([entry["avg_speedup_after"] for entry in history["ledger"]], [2.0, 1.9, 2.1])
        self.assertEqual([entry["iter"] for entry in history["rounds"]], [1, 2, 3])
        self.assertNotIn("_pending_proven_pattern", history)
        self.assertNotIn("_pending_regression_pattern", history)

    def test_failure_and_zero_score_do_not_replace_best_or_fill_window(self):
        self.perfs = {1: (1.6,), 2: (3.0,), 3: (0.0,), 4: (4.0,)}
        self.precision_failures = {2}
        self.build_failures = {4}
        self.run_workflow(max_iterations=4)
        status = self.selection_status()
        self.assertEqual(status["best"]["iteration"], 1)
        self.assertFalse(status["should_exit"])
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "max_iterations")

    def test_stagnation_exit_records_current_regression_before_final_summary(self):
        self.perfs = {1: (2.0,), 2: (2.0,), 3: (1.9,)}
        self.run_workflow(max_iterations=8)
        self.assertEqual(self.fixture.read_state()["iteration"], 3)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "semantic_stagnation")
        regression = (self.work / "knowledge/regression_patterns.md").read_text(encoding="utf-8")
        self.assertIn("LOCAL_TILE_REGRESSION", regression)
        self.assertIn("More padding overhead", regression)
        self.assertIn("-5.0", regression)
        self.assertEqual([stage for iteration, stage, _ in self.seen if iteration == 3],
                         ["stage4", "stage5", "stage9", "stage10"])

    def test_resume_stage9_does_not_count_the_same_evaluation_twice(self):
        self.perfs = {1: (1.6,), 2: (1.59,), 3: (1.6,)}
        self.interrupt_stage9_iteration = 1
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.run_workflow(max_iterations=8)
        self.assertEqual(self.fixture.read_state()["current_stage"], "iter1_stage9")
        self.interrupt_stage9_iteration = None
        self.run_workflow()
        self.assertEqual(self.fixture.read_state()["iteration"], 3)
        self.assertEqual(self.fixture.events.count("stage6"), 3)
        self.assertFalse(any(stage in {"stage7", "stage8"} for _, stage, _ in self.seen))
        self.assertEqual(self.selection_status()["best"]["iteration"], 1)

    def test_resume_final_iteration_finishes_review_without_repeating_performance(self):
        self.perfs = {1: (1.6,)}
        self.interrupt_stage9_iteration = 1
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.run_workflow(max_iterations=1)
        self.interrupt_stage9_iteration = None
        self.run_workflow()
        self.assertEqual(self.fixture.read_state()["iteration"], 1)
        self.assertEqual(self.fixture.read_state()["stopped_by"], "max_iterations")
        self.assertEqual(self.fixture.events.count("stage6"), 1)
        self.assertEqual(self.fixture.events.count("stage9"), 2)
        self.assertEqual(self.fixture.events.count("stage3"), 1)
        self.assertIn(self.selection_status()["best"]["implementation_dir"],
                      self.fixture.prompts["stage10"])

    def assert_old_precision_report_is_not_bound_to_new_code(self, agent_ok):
        self.perfs = {1: (1.6,), 2: (3.0,)}
        self.stale_precision_responses = {2: agent_ok}
        old_report = {
            "precision_overall": True, "total_cases": 1,
            "passed_cases": 1, "failed_cases": 0,
        }
        path = self.work / "eval/iter2/precision_result.json"
        self.write_json(path, old_report)
        before = orchestrator._file_revision(path)
        self.run_workflow(max_iterations=2)
        self.assertEqual(orchestrator._file_revision(path), before)
        self.assertEqual(self.fixture.events.count("stage6"), 2)
        binding = json.loads((self.work / "eval/iter2/precision_binding.json").read_text(encoding="utf-8"))
        self.assertFalse(binding["eligible"])
        status = self.selection_status()
        self.assertFalse(status["eligible"])
        self.assertEqual(status["best"]["iteration"], 1)
        self.assertEqual(status["window"]["valid_samples"], 1)
        self.assertIn(status["best"]["implementation_dir"], self.fixture.prompts["stage10"])

    def test_successful_precision_agent_cannot_rebind_an_untouched_old_report(self):
        self.assert_old_precision_report_is_not_bound_to_new_code(True)

    def test_failed_precision_agent_cannot_rebind_an_untouched_old_report(self):
        self.assert_old_precision_report_is_not_bound_to_new_code(False)

    def test_runtime_device_change_separates_results_with_unchanged_cached_hardware(self):
        from lib.semantic_exit import load_selection_status
        self.perfs = {1: (1.6,)}
        self.run_workflow(max_iterations=1)
        device_path = self.work / "device_info.json"
        original_bytes = device_path.read_bytes()
        hardware = json.loads(original_bytes)
        old_context = orchestrator.comparison_context(self.work / "task", hardware, self.config)
        changed_config = {**self.config, "hardware": {"device_id": 1}}
        new_context = orchestrator.comparison_context(self.work / "task", hardware, changed_config)
        self.assertEqual(old_context["evaluation_device_id"], 0)
        self.assertEqual(new_context["evaluation_device_id"], 1)
        self.assertEqual(new_context["hardware_config"], {"device_id": 1})
        self.assertEqual(old_context["hardware"], new_context["hardware"])
        self.assertEqual(device_path.read_bytes(), original_bytes)
        self.assertNotEqual(old_context, new_context)
        self.assertEqual(load_selection_status(self.work, old_context, self.config)["best"]["iteration"], 1)
        self.assertIsNone(load_selection_status(self.work, new_context, changed_config)["best"])

    def test_metadata_change_changes_context_and_excludes_prior_baseline_results(self):
        from lib.semantic_exit import load_selection_status
        baseline_path = self.work / "metadata/synthetic_baseline.json"
        self.write_json(baseline_path, {"synthetic_op_1": {"baseline_us": 100}})
        self.perfs = {1: (1.6,)}
        self.run_workflow(max_iterations=1)
        hardware = json.loads((self.work / "device_info.json").read_text(encoding="utf-8"))
        old_context = orchestrator.comparison_context(self.work / "task", hardware, self.config)
        self.write_json(baseline_path, {"synthetic_op_1": {"baseline_us": 120}})
        new_context = orchestrator.comparison_context(self.work / "task", hardware, self.config)
        self.assertIsNotNone(old_context["baseline_metadata_sha256"])
        self.assertNotEqual(old_context["baseline_metadata_sha256"], new_context["baseline_metadata_sha256"])
        self.assertEqual(old_context["task_sha256"], new_context["task_sha256"])
        self.assertEqual(old_context["evaluator_sha256"], new_context["evaluator_sha256"])
        self.assertEqual(load_selection_status(self.work, old_context, self.config)["best"]["iteration"], 1)
        self.assertIsNone(load_selection_status(self.work, new_context, self.config)["best"])


if __name__ == "__main__":
    unittest.main()

"""Offline integration checks for Stage1.5 and downstream library handoffs.

The Jev runner and prompt formatter are boundary doubles: probability validation
and formatter content are covered by the fusion-selection unit tests. Actual
orchestrator routing, Stage9, Stage3, history and checkpoints run unchanged.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

import orchestrator
from lib.state import State


ROLES = {
    "n1_stage1_requirements_analysis.md": "stage1",
    "n1_stage2_first_impl.md": "stage2",
    "n1_stage3_fix_and_optimize.md": "stage3",
    "n2_stage4_build.md": "stage4",
    "n2_stage5_precision_eval.md": "stage5",
    "n2_stage7_kerminal_profile.md": "stage7",
    "n3_stage8_search.md": "stage8",
    "n4_stage9_tech_lead_guide.md": "stage9",
    "n5_stage10_kerminal_report.md": "stage10",
}


def stage9_request(prompt):
    """Read the real, per-call output contract without consulting model-owned history."""
    def field(label):
        lines = [line[len(label):].strip() for line in prompt.splitlines()
                 if line.startswith(label)]
        if len(lines) != 1:
            raise AssertionError(f"Expected one {label} line, got {lines!r}")
        return lines[0]

    output = Path(field("Stage9 输出文件："))
    request_id = field("Stage9 请求编号：")
    if not output.is_absolute():
        raise AssertionError("Stage9 output path must be absolute")
    UUID(request_id)
    return output, request_id


def stage9_plan_task(known_case_ids=None, *, task_id="T1", file="impl/synthetic_op.py",
                     operation="modify"):
    """Fill the real versioned task contract, including an explicit routing claim."""
    targets = list(known_case_ids or [])[:1]
    return {
        "task_id": task_id,
        "priority": "P1",
        "task_type": "inspect" if operation == "inspect" else "modify",
        "action": "Tune tile size" if operation != "inspect" else "Inspect tile traversal",
        "reason": "Measured launch and padding overhead need a controlled experiment",
        "case_scope": "cases" if targets else "operator",
        "target_cases": targets,
        "operator_reason": "" if targets else "Repair the shared operator implementation before evaluation",
        "case_bindings": [{
            "case_id": case_id,
            "implementation_files": [file],
            "route_evidence": [{"file": file, "location": "synthetic operator entry",
                                "explanation": "The synthetic fixture has one shared implementation"}],
        } for case_id in targets],
        "changes": [{"file": file, "operation": operation, "location": "tile traversal",
                     "method": "Change tile size while preserving the operator semantics"
                     if operation != "inspect" else "Read traversal and report load-imbalance evidence"}],
        "acceptance_checks": ["All provided cases pass precision checks",
                              "Compare each affected case using the same performance measurement"],
    }


def stage9_payload(prompt, iteration):
    output, request_id = stage9_request(prompt)
    request = json.loads((output.parent / "request.json").read_text(encoding="utf-8"))
    if request.get("correction_attempt"):
        if request["correction_attempt"] not in (1, 2):
            raise AssertionError(f"Unexpected Stage9 correction attempt: {request['correction_attempt']}")
        root_request = json.loads((output.parent.parent / "request.json").read_text(encoding="utf-8"))
        if root_request.get("correction_attempt") != 0:
            raise AssertionError(f"Stage9 corrections must be siblings below the first request: {output}")
        expected_suffix = (Path("knowledge/stage9") / f"iter{iteration}"
                           / root_request["request_id"] / f"retry{request['correction_attempt']}" / "decision.json")
    else:
        expected_suffix = Path("knowledge/stage9") / f"iter{iteration}" / request_id / "decision.json"
    if output.parts[-len(expected_suffix.parts):] != expected_suffix.parts:
        raise AssertionError(f"Unexpected Stage9 output location: {output}")
    case_analysis = [{
        "case_id": case_id, "observation": "Current report still shows tile overhead",
        "explanation": "Hypothesis: tail padding contributes; validate with a controlled tile change",
        "evidence": f"eval/iter{iteration}/perf_result.json cases[{case_id}]",
        "next_action": "Change only the affected tile, then repeat precision and performance checks",
    } for case_id in request.get("performance_case_ids") or []]
    return output, {
        "plan_version": 2,
        "iteration": iteration,
        "request_id": request_id,
        "ledger_entry": {
            "evaluation_summary": f"Reviewed the actual implementation and current reports at iter{iteration}",
            "direction": f"LOCAL_TILE_ITER_{iteration}",
            "readonly_files": ["task/golden.py"],
            "case_analysis": case_analysis,
        },
        "suggest_next": [stage9_plan_task(request.get("known_case_ids"))],
    }


class FusionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        (self.work / "task").mkdir()
        self.library_path = self.work / "fusion/fusion_library.json"
        self.write_json("workflow_target.json", {
            "schema_version": 1, "framework": "Triton", "backend": "triton-ascend",
        })
        template = self.work / "absent_cannbench/examples/triton_ascend_cann_example/cann_bench"
        template.mkdir(parents=True)
        (template / "__init__.py").write_text("# synthetic Triton template\n", encoding="utf-8")
        self.write_json("device_info.json", {
            "chip_model": "offline_test", "soc_version": "offline_test",
            "npu_arch": "offline_test", "ai_core_num": 1,
            "ub_size_kb": 1, "l1_size_kb": 1, "l0a_size_kb": 1, "device_id": 0,
            "detect_method": "synthetic",
        })
        self.report = self.work / "cannbench_fixture.json"
        self.write_json(self.report, {"operators": [{
            "score": 60, "avg_speedup": 0.5, "total_cases": 1,
            "passed_cases": 1, "failed_cases": 0,
            "cases": [{"case_id": "synthetic_op_1", "status": "success",
                       "baseline_perf_us": 100, "elapsed_us": 200, "speedup": 0.5}],
        }]})
        self.events = []
        self.prompts = {}
        self.selection_error = None
        self.stage1_success = True

    def write_json(self, path, value):
        path = self.work / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def prepare_requirements(self):
        (self.work / "ANALYSIS.md").write_text("Synthetic operator requirements", encoding="utf-8")
        self.write_json("fusion_requirements.en.json", {
            "operator": "synthetic_op", "semantics": "y = sigmoid(x)",
        })

    def checkpoint(self, stage="N1_phase1", max_iterations=0):
        state = State.load_or_create(str(self.work), "synthetic_op", max_iterations)
        state.current_stage = stage
        state.max_iterations = max_iterations
        state.flush()

    def read_state(self):
        return json.loads((self.work / ".state.json").read_text(encoding="utf-8"))

    def write_fusion_artifacts(self, iteration):
        directory = self.work / "develop" / f"iter{iteration}"
        directory.mkdir(parents=True, exist_ok=True)
        library = json.loads(self.library_path.read_text(encoding="utf-8"))
        library["selection"] = {
            "method_ids": ["F2"], "implementation_plan": "Shape-aware tiled sigmoid",
            "reason": "Measured local tuning remains promising",
            "target_cases": ["all"], "actual_changes": [f"Tune iteration {iteration}"],
            "expected_benefits": ["Reduce tile overhead"],
        }
        self.write_json(directory / "fusion_library.json", library)
        (directory / "design_rationale.md").write_text("Tiled sigmoid design", encoding="utf-8")
        (directory / "融合方案选择决策依据.md").write_text(
            f"F2 selected; target all cases; local tile tuning in iteration {iteration}.", encoding="utf-8")
        (directory / "self_test_report.md").write_text(
            "Given cases pass. Same-shape repeated calls with new input pass.", encoding="utf-8")
        (directory / "self_test.log").write_text(
            "All provided cases pass; 2 calls with changed inputs match reference. "
            "No weights parameter. No bias parameter.", encoding="utf-8")
        self.write_json(directory / "self_test_result.json", {
            "schema_version": 1,
            "provided_cases": {"executed": True, "passed": True, "total": 1,
                               "passed_cases": 1, "evidence_path": "self_test.log"},
            "continuous_calls": {
                "executed": True, "passed": True, "same_shape": True, "call_count": 2,
                "reference_checked_each_call": True, "evidence_path": "self_test.log",
                "changes": {
                    "inputs": {"applicable": True, "changed": True, "passed": True},
                    "weights": {"applicable": False, "reason": "No weights parameter."},
                    "bias": {"applicable": False, "reason": "No bias parameter."},
                },
            },
        })

    def agent(self, _agent, role, _work, prompt, **_kwargs):
        stage = ROLES[Path(role).name]
        self.events.append(stage)
        self.prompts[stage] = prompt
        if stage == "stage1":
            if not self.stage1_success:
                return False
            self.prepare_requirements()
        elif stage == "stage2":
            (self.work / "impl/synthetic_op.py").write_text("# offline fixture\n", encoding="utf-8")
            self.write_fusion_artifacts(0)
        elif stage == "stage3":
            self.write_fusion_artifacts(self.read_state()["iteration"])
        elif stage == "stage4":
            (self.work / "build/iter1/build.log").write_text("STATUS: SUCCESS\n", encoding="utf-8")
        elif stage == "stage5":
            self.write_json("eval/iter1/precision_result.json", {
                "precision_overall": True, "total_cases": 1, "passed_cases": 1,
            })
        elif stage == "stage7":
            iteration = self.read_state()["iteration"]
            (self.work / f"profile/iter{iteration}/bottleneck_analysis.md").write_text(
                "Synthetic profiler evidence shows tile overhead.", encoding="utf-8")
        elif stage == "stage9":
            output, payload = stage9_payload(prompt, self.read_state()["iteration"])
            self.write_json(output, payload)
        return True

    def selection(self, work_dir, **_kwargs):
        self.events.append("stage1.5")
        self.assertEqual(Path(work_dir), self.work)
        self.assertTrue((self.work / "ANALYSIS.md").is_file())
        self.assertTrue((self.work / "fusion_requirements.en.json").is_file())
        if self.selection_error:
            raise self.selection_error
        library = {"schema_version": 1, "top_n": 1, "candidates": [{
            "rank": 1, "probability": 0.8,
            "method": {"id": "F2", "name": "Vertical"},
        }]}
        self.write_json(self.library_path, library)
        return library

    def format_library(self, work_dir, stage):
        self.assertEqual(Path(work_dir), self.work)
        self.assertTrue(self.library_path.is_file(), "Selection must finish before library handoff")
        # A stage-specific marker detects accidental omission or wrong dispatch.
        instruction = "highest_probability=F2" if stage == "stage2" else "evidence_first"
        return f"\nFUSION_HANDOFF[{stage}] {self.library_path} F2 probability=0.8 {instruction}\n"

    def performance(self, *_args, **_kwargs):
        self.events.append("stage6")
        return True, "", str(self.report)

    def patches(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("sys.argv", [
            "orchestrator.py", "--task-dir", str(self.work / "task"),
            "--work-dir", str(self.work), "--config", str(self.work / "synthetic_config.yaml"),
        ]))
        stack.enter_context(patch.object(orchestrator, "load_config", return_value={
            "paths": {"cannbench_repo": str(self.work / "absent_cannbench")},
            "fusion_selection": {"top_n": 2},
        }))
        for name in ("setup_logger", "setup_state_logger", "setup_history_logger", "setup_node_logger"):
            stack.enter_context(patch.object(orchestrator, name, return_value=Mock()))
        stack.enter_context(patch.object(orchestrator.atexit, "register"))
        stack.enter_context(patch.object(orchestrator.signal, "signal"))
        stack.enter_context(patch.object(orchestrator.os, "symlink"))
        stack.enter_context(patch.object(orchestrator, "detect_triton_runtime", return_value={
            "framework": "Triton", "backend": "triton-ascend", "driver_backend": "npu",
            "target_arch": "synthetic", "programming_model": "Triton block program (SPMD)",
            "runtime_versions": {"triton": "synthetic", "triton_ascend": "synthetic",
                                 "torch": "synthetic", "torch_npu": "synthetic"},
        }))
        stack.enter_context(patch.object(orchestrator, "detect_npu_device",
                                         return_value=json.loads((self.work / "device_info.json").read_text(encoding="utf-8"))))
        stack.enter_context(patch("lib.jev_client.load_settings",
                                 side_effect=AssertionError("No credential reads in offline tests")))
        stack.enter_context(patch("lib.jev_client.evaluate",
                                 side_effect=AssertionError("No Jev calls in offline tests")))
        stack.enter_context(patch.object(orchestrator, "run_agent", side_effect=self.agent))
        self.selector = stack.enter_context(patch.object(orchestrator, "run_fusion_selection",
                                                        side_effect=self.selection))
        self.formatter = stack.enter_context(patch.object(orchestrator, "format_fusion_library_for_prompt",
                                                         side_effect=self.format_library))
        self.perf = stack.enter_context(patch.object(orchestrator, "run_perf_eval", side_effect=self.performance))
        return stack

    def test_new_run_selects_before_first_implementation(self):
        self.checkpoint()
        with self.patches():
            orchestrator.main()
            self.selector.assert_called_once_with(
                str(self.work), config_path=str(self.work / "synthetic_config.yaml"), top_n=2)
            self.perf.assert_not_called()
        self.assertEqual(self.events, ["stage1", "stage1.5", "stage2", "stage10"])
        self.assertIn("fusion_requirements.en.json", self.prompts["stage1"])
        self.assertIn("highest_probability=F2", self.prompts["stage2"])
        self.assertIn(str(self.library_path), self.prompts["stage2"])

    def test_selection_failure_stops_before_implementation_and_evaluation(self):
        self.checkpoint(max_iterations=1)
        self.selection_error = RuntimeError("synthetic Jev failure")
        with self.patches():
            with self.assertRaisesRegex(RuntimeError, "synthetic Jev failure"):
                orchestrator.main()
            self.perf.assert_not_called()
            self.formatter.assert_not_called()
        self.assertEqual(self.events, ["stage1", "stage1.5"])
        self.assertEqual(self.read_state()["current_stage"], "iter0_stage1.5")

    def test_resume_selection_does_not_repeat_stage1(self):
        self.checkpoint("iter0_stage1.5")
        self.prepare_requirements()
        with self.patches():
            orchestrator.main()
        self.assertEqual(self.events, ["stage1.5", "stage2", "stage10"])
        self.assertIn("highest_probability=F2", self.prompts["stage2"])

    def test_failed_analysis_does_not_call_selection(self):
        self.checkpoint(max_iterations=1)
        self.stage1_success = False
        with self.patches():
            with self.assertRaisesRegex(RuntimeError, "Stage1 failed"):
                orchestrator.main()
            self.selector.assert_not_called()
            self.perf.assert_not_called()
        self.assertEqual(self.events, ["stage1"])
        self.assertEqual(self.read_state()["current_stage"], "N1_phase1")

    def test_underperforming_path_passes_library_to_all_five_consumers(self):
        self.checkpoint(max_iterations=1)
        with self.patches():
            orchestrator.main()
            self.assertEqual(self.perf.call_count, 1)
            self.assertEqual([call.args[1] for call in self.formatter.call_args_list],
                             ["stage2", "stage7", "stage8", "stage9", "stage3"])
        self.assertEqual(self.events, ["stage1", "stage1.5", "stage2", "stage4", "stage5",
                                      "stage6", "stage7", "stage8", "stage9", "stage3", "stage10"])
        for stage in ("stage2", "stage3", "stage7", "stage8", "stage9"):
            with self.subTest(stage=stage):
                self.assertIn(f"FUSION_HANDOFF[{stage}]", self.prompts[stage])
                self.assertIn(str(self.library_path), self.prompts[stage])
                self.assertIn("F2 probability=0.8", self.prompts[stage])
        # Verify actual injected prompts, not just the static role documentation.
        input_paths = {
            "stage1": "task", "stage2": "ANALYSIS.md", "stage3": "eval/iter1/perf_result.json",
            "stage4": "impl", "stage5": "task", "stage7": "eval/iter1/perf_result.json",
            "stage8": "profile/iter1/bottleneck_analysis.md", "stage9": "knowledge/history.json",
            "stage10": ".state.json",
        }
        for stage, relative in input_paths.items():
            with self.subTest(file_stage=stage):
                lines = self.prompts[stage].splitlines()
                description = next(line for line in lines if f"相对工作目录：`{relative}`" in line)
                self.assertIn("用途：", description)
                self.assertIn("怎么看：", description)
                if "iter1" in relative:
                    self.assertIn(relative.replace("iter1", "<iter>"), description)
        self.assertIn("相对项目根目录：`knowledge/anti_cheat_reference.md`", self.prompts["stage9"])
        self.assertEqual(self.read_state()["stopped_by"], "max_iterations")

    def test_stage_numbers_are_exact_not_prefix_matches(self):
        cases = {
            "stage1": 1, "iter0_stage1": 1,
            "stage1.5": 1.5, "iter0_stage1.5": 1.5,
            "stage10": 10, "iter3_stage10": 10,
            "N1_phase1": None, "stage1.50": None,
            "stage1.5suffix": None, "iter1_stage10_extra": None,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(orchestrator.stage_number(value), expected)


if __name__ == "__main__":
    unittest.main()

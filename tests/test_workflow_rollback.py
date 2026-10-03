"""Offline smoke checks for the restored workflow and checkpoint boundary."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import orchestrator
from lib.history_manager import load_history, save_history
from lib.state import State


class RestoredWorkflowTests(unittest.TestCase):
    def exercise_main(self, *, exit_decision=True, report_only=False):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / "task").mkdir()
            (work / "workflow_target.json").write_text(json.dumps({
                "schema_version": 1, "framework": "Triton", "backend": "triton-ascend",
            }), encoding="utf-8")
            template = work / "absent_cannbench/examples/triton_ascend_cann_example/cann_bench"
            template.mkdir(parents=True)
            (template / "__init__.py").write_text("# synthetic template\n", encoding="utf-8")
            state = State.load_or_create(folder, "synthetic_op", 1)
            state.current_stage = "iter1_stage10" if report_only else "ready_for_iter"
            state.iteration = 1 if report_only else 0
            state.flush()
            device = dict(chip_model="test", soc_version="test", npu_arch="test",
                          ai_core_num=1, ub_size_kb=1, l1_size_kb=1, l0a_size_kb=1, device_id=0,
                          detect_method="synthetic")
            (work / "device_info.json").write_text(json.dumps(device), encoding="utf-8")
            report = work / "cannbench_fixture.json"
            report.write_text(json.dumps({"operators": [{
                "score": 90, "avg_speedup": 1.25, "total_cases": 1,
                "passed_cases": 1, "failed_cases": 0,
                "cases": [{"case_id": "synthetic_op_1", "status": "success",
                           "baseline_perf_us": 100, "elapsed_us": 80, "speedup": 1.25}],
            }]}), encoding="utf-8")
            events = []

            def agent(_agent, role, _work, _prompt, **_kwargs):
                name = Path(role).name
                events.append(name)
                if name == "n2_stage4_build.md":
                    (work / "build/iter1/build.log").write_text("STATUS: SUCCESS\n", encoding="utf-8")
                elif name == "n2_stage5_precision_eval.md":
                    (work / "eval/iter1/precision_result.json").write_text(
                        json.dumps({"precision_overall": True, "total_cases": 1, "passed_cases": 1}),
                        encoding="utf-8")
                return True

            def tech_lead(*_args, **kwargs):
                self.assertIn("perf_pass_semantic_review", kwargs["fail_reason"])
                events.append("stage9")
                _args[4].current_stage = "iter1_stage9"
                _args[4].flush()
                history = load_history(folder)
                history["exit_decision"] = exit_decision
                save_history(folder, history)

            def stage3(*_args, **kwargs):
                self.assertEqual(kwargs["reason"], "perf_pass_optimize")
                events.append("stage3")
                _args[4].current_stage = "iter1_stage3"
                _args[4].flush()

            with ExitStack() as stack:
                stack.enter_context(patch("sys.argv", ["orchestrator.py", "--task-dir", str(work / "task"),
                                                        "--work-dir", folder]))
                stack.enter_context(patch.object(orchestrator, "load_config", return_value={
                    "paths": {"cannbench_repo": str(work / "absent_cannbench")}}))
                for name in ("setup_logger", "setup_state_logger", "setup_history_logger", "setup_node_logger"):
                    stack.enter_context(patch.object(orchestrator, name, return_value=Mock()))
                stack.enter_context(patch.object(orchestrator.atexit, "register"))
                stack.enter_context(patch.object(orchestrator.signal, "signal"))
                stack.enter_context(patch.object(orchestrator.os, "symlink"))
                stack.enter_context(patch.object(orchestrator, "detect_npu_device", return_value=device))
                stack.enter_context(patch.object(orchestrator, "detect_triton_runtime", return_value={
                    "framework": "Triton", "backend": "triton-ascend", "driver_backend": "npu",
                    "target_arch": "synthetic", "programming_model": "Triton block program (SPMD)",
                    "runtime_versions": {"triton": "synthetic", "triton_ascend": "synthetic",
                                         "torch": "synthetic", "torch_npu": "synthetic"},
                }))
                stack.enter_context(patch.object(orchestrator, "run_agent", side_effect=agent))
                stack.enter_context(patch.object(orchestrator, "run_tech_lead", side_effect=tech_lead))
                stack.enter_context(patch.object(orchestrator, "self_goto_stage3", side_effect=stage3))
                perf = stack.enter_context(patch.object(orchestrator, "run_perf_eval",
                                                       return_value=(True, "", str(report))))
                jev = stack.enter_context(patch("lib.jev_client.evaluate",
                                                side_effect=AssertionError("Workflow must not call Jev")))
                orchestrator.main()
                jev.assert_not_called()
                self.assertEqual(perf.call_count, 0 if report_only else 1)
            saved = json.loads((work / ".state.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["current_stage"], "iter1_stage10")
            self.assertNotIn("fusion", saved)
            self.assertNotIn("workflow", saved)
            return events, saved

    def test_tech_lead_true_cannot_skip_semantic_exit_window(self):
        events, state = self.exercise_main()
        self.assertEqual(events, ["n2_stage4_build.md", "n2_stage5_precision_eval.md",
                                  "stage9", "stage3", "n5_stage10_kerminal_report.md"])
        self.assertEqual(state["stopped_by"], "max_iterations")

    def test_tech_lead_false_preserves_optimization_route(self):
        events, state = self.exercise_main(exit_decision=False)
        self.assertEqual(events, ["n2_stage4_build.md", "n2_stage5_precision_eval.md",
                                  "stage9", "stage3", "n5_stage10_kerminal_report.md"])
        self.assertEqual(state["stopped_by"], "max_iterations")

    def test_final_report_checkpoint_does_not_restart_stage1(self):
        events, _ = self.exercise_main(report_only=True)
        self.assertEqual(events, ["n5_stage10_kerminal_report.md"])

    def test_experimental_checkpoint_is_rejected_without_rewriting_it(self):
        for extra in ({"schema_version": 2}, {"fusion": {}}, {"workflow": {}}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / ".state.json"
                checkpoint.write_text(json.dumps({"current_stage": "iter1_stage10", **extra}), encoding="utf-8")
                before = checkpoint.read_bytes()
                with self.assertRaisesRegex(ValueError, "withdrawn experimental routing"):
                    State.load_or_create(folder)
                self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

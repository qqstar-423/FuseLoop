"""Exercise emergency imports through the real control plane, without providers."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import orchestrator
from lib.fusion_evidence import implementation_hash, load_evidence
from test_init_impl import make_source, write_json


class PauseFixture(RuntimeError):
    pass


class InitImplRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = make_source(self.root)
        self.runtime = {
            "framework": "Triton", "backend": "triton-ascend", "driver_backend": "npu",
            "target_arch": "test-v1", "programming_model": "Triton block program (SPMD)",
            "runtime_versions": {"triton": "test", "triton_ascend": "test", "torch": "test", "torch_npu": "test"},
        }
        self.update_source_device()
        self.work = self.root / "runs/new_run"
        self.events, self.prompts = [], {}
        self.stop_at = "stage7"
        self.log = Mock()
        self.config = {"paths": {"cannbench_repo": str(self.root / "bench")},
                       "workflow": {"max_iterations": 1}}

    def update_source_device(self):
        self.device = json.loads((self.source / "device_info.json").read_text())
        self.device.update(l0a_size_kb=64, device_id=0, **self.runtime)
        write_json(self.source / "device_info.json", self.device)
        sha = hashlib.sha256((self.source / "device_info.json").read_bytes()).hexdigest()
        path = self.source / "fusion/jev_request.json"
        request = json.loads(path.read_text())
        request["state"]["hardware"] = self.device
        write_json(path, request)
        for name in ("fusion_library.json", "ranking.json"):
            path = self.source / "fusion" / name
            document = json.loads(path.read_text())
            document["sources"]["hardware"]["sha256"] = sha
            document["fingerprint_sha256"] = hashlib.sha256(json.dumps({
                "protocol": document["protocol"], "request": request, "jev_config": document["jev_config"],
                "source_sha256": {key: value["sha256"] for key, value in document["sources"].items()},
            }, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
            write_json(path, document)

    def agent(self, agent, role, work_dir, prompt, **kwargs):
        name = Path(role).stem
        stage = next((f"stage{n}" for n in (10, 1, 2, 3, 4, 5, 7, 8, 9)
                      if f"stage{n}_" in name), name)
        self.events.append(stage)
        self.prompts[stage] = prompt
        self.assertEqual(Path(work_dir), self.work)
        if stage == self.stop_at:
            raise PauseFixture(stage)
        if stage == "stage4":
            (self.work / "build/iter1/build.log").write_text("STATUS: SUCCESS\n", encoding="utf-8")
        elif stage == "stage5":
            write_json(self.work / "eval/iter1/precision_result.json", {
                "precision_overall": True, "total_cases": 2, "passed_cases": 2, "failed_cases": 0})
        else:
            raise AssertionError(f"Unexpected agent stage: {stage}")
        return True

    def performance(self, *args, **kwargs):
        self.events.append("stage6")
        report = self.work / "eval/iter1/source.json"
        write_json(report, {"operators": [{
            "total_cases": 2, "passed_cases": 2, "failed_cases": 0,
            "score": 60, "performance_score": 60, "avg_speedup": .8,
            "cases": [{"case_id": f"case_{i}", "status": "success", "speedup": .8,
                       "elapsed_us": 125, "baseline_perf_us": 100, "perf_score": .8,
                       "t_hw_us": 10, "op_times": [{"name": "kernel", "time_us": 125}]}
                      for i in (1, 2)],
        }]})
        return True, "", str(report)

    def patches(self, *, resume=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        argv = ["orchestrator.py", "--task-dir", str(self.source / "task"),
                "--config", str(self.root / "synthetic.yaml")]
        argv += (["--work-dir", str(self.work)] if resume else ["--init-impl", str(self.source / "impl")])
        stack.enter_context(patch("sys.argv", argv))
        stack.enter_context(patch.object(orchestrator, "load_config", return_value=self.config))
        stack.enter_context(patch.object(orchestrator, "allocate_work_directory", return_value=str(self.work)))
        for name in ("setup_logger", "setup_state_logger", "setup_history_logger", "setup_node_logger"):
            stack.enter_context(patch.object(orchestrator, name, return_value=self.log))
        stack.enter_context(patch.object(orchestrator.atexit, "register"))
        stack.enter_context(patch.object(orchestrator.signal, "signal"))
        # Portable transport double for directory symlinks; source metadata and
        # all import/state/evidence/measurement decisions run without mocks.
        stack.enter_context(patch.object(orchestrator, "ensure_task_link", side_effect=lambda work, task:
            shutil.copytree(task, Path(work) / "task", dirs_exist_ok=True)))
        stack.enter_context(patch.object(orchestrator, "ensure_example_link"))
        stack.enter_context(patch.object(orchestrator.os, "symlink"))
        stack.enter_context(patch.object(orchestrator, "detect_npu_device", side_effect=lambda **kwargs: dict(self.device)))
        stack.enter_context(patch.object(orchestrator, "detect_triton_runtime", return_value=self.runtime))
        stack.enter_context(patch.object(orchestrator, "run_agent", side_effect=self.agent))
        stack.enter_context(patch.object(orchestrator, "run_perf_eval", side_effect=self.performance))
        self.jev = stack.enter_context(patch.object(orchestrator, "run_fusion_selection",
            side_effect=AssertionError("Emergency import must never call Jev")))
        stack.enter_context(patch.object(orchestrator, "fusion_requirements_byte_budget",
            side_effect=AssertionError("Emergency import must skip Stage1")))
        return stack

    def test_implementation_without_triton_stops_before_allocation_or_stages(self):
        path = self.source / "impl/cann_bench/__init__.py"
        path.write_text("from other_framework import kernel\n", encoding="utf-8")
        before = path.read_bytes()
        with self.patches(), patch.object(orchestrator, "load_config") as config, \
                patch.object(orchestrator, "allocate_work_directory") as allocate, \
                self.assertRaisesRegex(ValueError, "has no Triton source import"):
            orchestrator.main()
        config.assert_not_called()
        allocate.assert_not_called()
        self.assertEqual(self.events, [])
        self.assertFalse(self.work.exists())
        self.assertEqual(path.read_bytes(), before)

    def test_starts_with_build_then_fresh_measurement_and_best_without_old_history(self):
        old_hash = implementation_hash(self.source / "impl")
        for relative in ("knowledge/history.json", "eval/iter8/perf_result.json",
                         "human_review/inbox/old.json", "selection/best.json"):
            write_json(self.source / relative, {"old_sentinel": True})
        with self.patches(), self.assertRaisesRegex(PauseFixture, "stage7"):
            orchestrator.main()
        self.assertEqual(self.events, ["stage4", "stage5", "stage6", "stage7"])
        self.jev.assert_not_called()
        self.assertTrue(load_evidence(self.work)["eligible"])
        self.assertEqual(implementation_hash(self.source / "impl"), old_hash)
        self.assertEqual(implementation_hash(self.work / "impl"), old_hash)
        best = json.loads((self.work / "selection/best.json").read_text())
        self.assertEqual(best["iteration"], 1)
        self.assertEqual(best["avg_speedup"], .8)
        self.assertTrue(Path(best["implementation_dir"]).is_relative_to(self.work))
        self.assertFalse((self.work / "eval/iter8").exists())
        self.assertFalse((self.work / "human_review/inbox/old.json").exists())
        history = json.loads((self.work / "knowledge/history.json").read_text())
        self.assertEqual([row["iter"] for row in history["rounds"]], [1])
        self.assertNotIn("old_sentinel", history)
        self.assertIn("skipping Stage1", str(self.log.info.call_args_list))
        self.assertIn("copied", str(self.log.info.call_args_list))
        self.assertIn("init_impl_manifest.json", self.prompts["stage4"])

    def test_stage3_generated_source_keeps_matching_selftests_and_enters_build(self):
        self.source = make_source(self.root / "later", stage=3)
        self.update_source_device()
        with self.patches(), self.assertRaises(PauseFixture):
            orchestrator.main()
        self.assertEqual(self.events, ["stage4", "stage5", "stage6", "stage7"])
        evidence = load_evidence(self.work)
        self.assertTrue(evidence["eligible"], evidence)
        self.assertEqual(evidence["stage"], 3)
        self.assertTrue((self.work / "develop/iter0/fusion_scheme_rationale.md").is_file())
        self.assertFalse((self.work / "develop/iter2").exists())
        self.assertNotIn(str(self.work / "develop/iter0/design_rationale.md"), self.prompts["stage7"])
        self.assertTrue((self.work / "selection/best.json").is_file())

    def test_resume_build_keeps_emergency_mode_without_reimport_or_jev(self):
        self.stop_at = "stage4"
        with self.patches(), self.assertRaises(PauseFixture):
            orchestrator.main()
        manifest = (self.work / "init_impl_manifest.json").read_bytes()
        self.events.clear()
        self.stop_at = "stage5"
        with self.patches(resume=True), patch.object(orchestrator, "prepare_init_impl",
                side_effect=AssertionError("Resume must not copy source again")), self.assertRaises(PauseFixture):
            orchestrator.main()
        self.assertEqual(self.events, ["stage4", "stage5"])
        self.jev.assert_not_called()
        self.assertEqual((self.work / "init_impl_manifest.json").read_bytes(), manifest)
        self.assertEqual(json.loads((self.work / ".state.json").read_text())["iteration"], 1)
        self.events.clear()
        self.stop_at = "stage7"
        with self.patches(resume=True), self.assertRaises(PauseFixture):
            orchestrator.main()
        self.assertEqual(self.events, ["stage5", "stage6", "stage7"])
        self.jev.assert_not_called()
        self.assertEqual(json.loads((self.work / "selection/best.json").read_text())["iteration"], 1)

    def test_resume_after_device_probe_failure_still_skips_initial_stages(self):
        with self.patches(), patch.object(orchestrator, "detect_triton_runtime", side_effect=RuntimeError("offline probe")), \
                self.assertRaises(SystemExit):
            orchestrator.main()
        self.assertEqual(self.events, [])
        self.stop_at = "stage4"
        with self.patches(resume=True), self.assertRaises(PauseFixture):
            orchestrator.main()
        self.assertEqual(self.events, ["stage4"])

    def test_changed_hardware_stops_before_any_agent_and_preserves_scored_device_file(self):
        self.device["ub_size_kb"] += 1
        with self.patches(), self.assertRaisesRegex(ValueError, "chip/framework"):
            orchestrator.main()
        self.assertEqual(self.events, [])
        self.assertEqual((self.work / "device_info.json").read_bytes(),
                         (self.source / "device_info.json").read_bytes())

    def test_missing_materials_do_not_fall_back_to_initial_development_on_resume(self):
        (self.source / "fusion/ranking.json").unlink()
        with self.patches(), self.assertRaises(ValueError):
            orchestrator.main()
        with self.patches(resume=True), self.assertRaisesRegex(ValueError, "the previous import did not finish"):
            orchestrator.main()
        self.assertEqual(self.events, [])

    def test_conflicting_cli_rejected_before_any_directory_or_config_access(self):
        with patch("sys.argv", ["orchestrator.py", "--task-dir", str(self.source / "task"),
                   "--work-dir", str(self.work), "--init-impl", str(self.source / "impl")]), \
                patch.object(orchestrator, "load_config") as load, \
                patch.object(orchestrator, "allocate_work_directory") as allocate, self.assertRaises(SystemExit):
            orchestrator.main()
        load.assert_not_called()
        allocate.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_same_second_allocations_are_unique_and_never_overwrite_existing_content(self):
        first = Path(orchestrator.allocate_work_directory(self.root / "runs", "operator", "20260928_123456"))
        (first / "sentinel").write_text("preserve", encoding="utf-8")
        second = Path(orchestrator.allocate_work_directory(self.root / "runs", "operator", "20260928_123456"))
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir() and second.is_dir())
        self.assertEqual((first / "sentinel").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()

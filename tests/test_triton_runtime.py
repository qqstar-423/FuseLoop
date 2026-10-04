"""Offline checks for target identity, actual backend probing and device handoff."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import orchestrator
from lib import agent_runner, bench_parser
from lib.framework_target import (TARGET_IDENTITY, PROGRAMMING_MODEL,
                                  ensure_workflow_target, ensure_example_link, detect_triton_runtime,
                                  read_cann_toolchain)
from lib.knowledge_metadata import build_knowledge_environment
from lib.performance_comparison import compare_performance


class TritonRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="triton_runtime_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = {
            "framework": "Triton", "backend": "triton-ascend",
            "driver_backend": "npu", "target_arch": "Ascend910B3",
            "runtime_versions": {"triton": "test-3.2", "triton_ascend": "test-3.2-ascend",
                                 "torch": "test-2.6", "torch_npu": "test-2.6"},
        }

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def probe(self, runtime=None, device_id=2):
        runtime = self.runtime if runtime is None else runtime
        response = SimpleNamespace(returncode=0, stderr="", stdout=(
            "harmless startup banner\nWORKFLOW_TRITON_RUNTIME=" + json.dumps(runtime) + "\n"))
        initial = {"ASCEND_RT_VISIBLE_DEVICES": "4,7", "ASCEND_VISIBLE_DEVICES": "4,7"}
        toolkit = self.root / "cann"
        (toolkit / "compiler").mkdir(parents=True, exist_ok=True)
        (toolkit / "compiler/version.info").write_text("Version=synthetic-cann\n", encoding="utf-8")
        initial["ASCEND_HOME_PATH"] = str(toolkit)
        with patch("lib.cann_env.build_cann_env", return_value=initial) as env, \
                patch("lib.framework_target.subprocess.run", return_value=response) as run:
            result = detect_triton_runtime(device_id, str(self.root / "synthetic.yaml"))
        return result, env, run

    def test_active_npu_backend_and_versions_are_required_without_pypto(self):
        runtime, env, run = self.probe()
        self.assertEqual(runtime["backend"], "triton-ascend")
        self.assertEqual(runtime["driver_backend"], "npu")
        self.assertEqual(runtime["programming_model"], PROGRAMMING_MODEL)
        self.assertEqual(runtime["toolchain"]["version_files"][0]["version"], "synthetic-cann")
        script = run.call_args.args[0][2]
        self.assertIn("torch.npu.set_device(2)", script)
        self.assertIn("driver.active.get_current_target()", script)
        self.assertNotIn("pypto", script)
        env.assert_called_once_with({"WORKFLOW_NPU_DEVICE_ID": "2"},
                                    config_path=str(self.root / "synthetic.yaml"))
        self.assertEqual(run.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4,7")

    def test_wrong_backend_or_missing_version_fails_closed(self):
        for value in (dict(self.runtime, driver_backend="cuda"),
                      dict(self.runtime, driver_backend="hip"),
                      dict(self.runtime, runtime_versions={"triton": "3.2"})):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.probe(value)

    def test_device_id_validation_precedes_environment_or_subprocess_access(self):
        for value in (-1, True, "0", None):
            with self.subTest(value=value), patch("lib.cann_env.build_cann_env") as env, \
                    self.assertRaises(ValueError):
                detect_triton_runtime(value)
            env.assert_not_called()

    def test_empty_work_gets_explicit_target_identity_and_resumes(self):
        work = self.root / "fresh"
        marker = ensure_workflow_target(work)
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), TARGET_IDENTITY)
        (work / "impl").mkdir()
        (work / "impl/kernel.py").write_text("import triton\nimport triton.language as tl\n", encoding="utf-8")
        before = marker.read_bytes()
        self.assertEqual(ensure_workflow_target(work), marker)
        self.assertEqual(marker.read_bytes(), before)

    def test_unmarked_old_checkpoint_is_unchanged_and_not_adopted(self):
        state = self.write("old/.state.json", {"iteration": 8, "current_stage": "iter8_stage9"})
        before = state.read_bytes()
        with self.assertRaisesRegex(ValueError, "resuming an old PyPTO task directly is forbidden"):
            ensure_workflow_target(state.parent)
        self.assertEqual(state.read_bytes(), before)
        self.assertFalse((state.parent / "workflow_target.json").exists())

    def test_wrong_marked_framework_cannot_resume(self):
        marker = self.write("old/workflow_target.json", dict(TARGET_IDENTITY, framework="PyPTO Pro"))
        with self.assertRaisesRegex(ValueError, "Cross-framework checkpoint resume is forbidden"):
            ensure_workflow_target(marker.parent)

    def test_missing_example_fails_and_existing_wrong_directory_is_not_adopted(self):
        work = self.root / "work"
        work.mkdir()
        bench = self.root / "bench"
        with self.assertRaisesRegex(ValueError, "Missing the standard Triton Ascend example"):
            ensure_example_link(work, bench)
        source = bench / "examples/triton_ascend_cann_example"
        (source / "cann_bench").mkdir(parents=True)
        (source / "cann_bench/__init__.py").write_text("", encoding="utf-8")
        with patch("lib.framework_target.os.symlink") as link:
            self.assertEqual(ensure_example_link(work, bench), source.resolve())
            link.assert_called_once_with(source.resolve(), work / "example", target_is_directory=True)
        (work / "example").mkdir()
        with self.assertRaisesRegex(ValueError, "source does not match"):
            ensure_example_link(work, bench)

    def test_marking_old_pypto_implementation_as_triton_does_not_bypass_guard(self):
        work = self.root / "old"
        self.write("old/workflow_target.json", TARGET_IDENTITY)
        (work / "impl").mkdir()
        path = work / "impl/kernel.py"
        path.write_text("import torch, pypto_pro.language as arbitrary_alias\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "still imports the old"):
            ensure_workflow_target(work)
        self.assertIn("pypto_pro", path.read_text(encoding="utf-8"))

    def test_chip_probe_uses_configured_device_and_only_torch_npu_tbe(self):
        responses = [
            SimpleNamespace(returncode=0, stderr="", stdout=(
                "WORKFLOW_SOC=Ascend910B3\nCUBE_CORES=20\nVECTOR_CORES=40\nL2_BYTES=1048576\n")),
            SimpleNamespace(returncode=0, stderr="", stdout=(
                "UB_SIZE=196608\nL1_SIZE=524288\nL0A_SIZE=65536\nL0B_SIZE=65536\nL0C_SIZE=131072\nCORE_NUM=20\n")),
        ]
        with patch("lib.cann_env.build_cann_env", return_value={}), \
                patch("subprocess.run", side_effect=responses) as run:
            info = orchestrator.detect_npu_device(3, str(self.root / "synthetic.yaml"))
        self.assertEqual(info["device_id"], 3)
        self.assertEqual(info["chip_model"], "Ascend910B3")
        self.assertEqual(info["ub_size_kb"], 192)
        self.assertEqual(info["programming_model"], PROGRAMMING_MODEL)
        scripts = "\n".join(call.args[0][2] for call in run.call_args_list)
        self.assertIn("get_device_properties(3)", scripts)
        self.assertNotIn("pypto", scripts)
        self.assertNotIn("ARCH_MAP", scripts)

    def test_agent_and_evaluator_preserve_visible_mapping_and_share_device_id(self):
        initial = {"ASCEND_RT_VISIBLE_DEVICES": "4,7", "PYTHONPATH": "/synthetic"}
        with patch("lib.cann_env.build_cann_env", side_effect=lambda **kwargs: dict(initial)):
            env = agent_runner._agent_env({"device_id": 1, "workflow_config_path": "synthetic.yaml"})
        self.assertEqual(env["WORKFLOW_NPU_DEVICE_ID"], "1")
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "4,7")
        self.assertNotIn("TILE_FWK_DEVICE_ID", env)
        response = SimpleNamespace(returncode=0, stdout="", stderr="")
        for evaluate in (bench_parser.run_perf_eval, bench_parser.run_precision_eval):
            with self.subTest(evaluate=evaluate.__name__), \
                    patch.object(bench_parser, "_build_eval_env", side_effect=lambda _: dict(initial)), \
                    patch.object(bench_parser, "_find_latest_report", return_value=None), \
                    patch.object(bench_parser.subprocess, "run", return_value=response) as run:
                evaluate(str(self.root / "task"), str(self.root / "cannbench"), device_id=1)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--device-id") + 1], "1")
            self.assertEqual(run.call_args.kwargs["env"]["WORKFLOW_NPU_DEVICE_ID"], "1")
            self.assertEqual(run.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4,7")

    def test_runtime_and_framework_are_part_of_performance_comparison(self):
        task = self.root / "task"
        task.mkdir()
        hardware = {"chip_model": "synthetic", "runtime_versions": self.runtime["runtime_versions"]}
        context = orchestrator.comparison_context(task, hardware, {"paths": {"cannbench_repo": str(self.root / "bench")}})
        report = {"comparison_context": context, "comparison_context_stable": True,
                  "avg_speedup": 2, "total_cases": 1, "cases": [{"case_id": "case1", "status": "success",
                      "speedup": 2, "elapsed_us": 5, "baseline_perf_us": 10}]}
        self.assertTrue(compare_performance(report, deepcopy(report))["comparable"])
        for field, value in (("framework", "PyPTO Pro"), ("backend", "cuda"),
                             ("runtime_versions", dict(self.runtime["runtime_versions"], triton="new-version"))):
            changed = deepcopy(report)
            changed["comparison_context"][field] = value
            with self.subTest(field=field):
                outcome = compare_performance(report, changed)
                self.assertFalse(outcome["comparable"])
                self.assertIn("comparison_context." + field, outcome["mismatch_fields"])

    def test_historical_framework_label_is_not_replaced_by_current_target(self):
        report = self.write("old_report.json", {"comparison_context": {
            "framework": "PyPTO Pro", "backend": "historical-pypto", "hardware": {"chip_model": "old-chip"}}})
        environment = build_knowledge_environment(self.root, performance_report=report)
        self.assertEqual(environment["framework"], "PyPTO Pro")
        self.assertEqual(environment["framework_source"], "performance_report")
        self.write("old_report.json", {"comparison_context": {"hardware": {"chip_model": "old-chip"}}})
        self.assertEqual(build_knowledge_environment(self.root, performance_report=report)["framework"], "unknown")
        self.assertEqual(build_knowledge_environment(self.root)["framework"], "Triton")

    def test_cann_metadata_change_is_visible_in_comparison_context(self):
        toolkit = self.root / "cann"
        (toolkit / "compiler").mkdir(parents=True)
        metadata = toolkit / "compiler/version.info"
        metadata.write_text("Version=synthetic-cann-A\n", encoding="utf-8")
        first = read_cann_toolchain({"ASCEND_HOME_PATH": str(toolkit)})
        hardware = {**self.runtime, "toolchain": first}
        original = deepcopy(hardware)
        config = {"paths": {"cannbench_repo": str(self.root / "bench")}}
        before = orchestrator.comparison_context(self.root / "task", hardware, config)
        metadata.write_text("Version=synthetic-cann-B\n", encoding="utf-8")
        with patch.object(orchestrator, "detect_triton_runtime", side_effect=AssertionError("No runtime probe here")):
            after = orchestrator.comparison_context(self.root / "task", hardware, config)
        self.assertNotEqual(before, after)
        self.assertEqual(hardware, original)
        self.assertEqual(before["hardware"]["toolchain"]["version_files"][0]["version"], "synthetic-cann-A")
        self.assertEqual(after["hardware"]["toolchain"]["version_files"][0]["version"], "synthetic-cann-B")
        self.assertIsNot(before["hardware"], hardware)

    def test_disappearing_cann_metadata_stops_comparison_context_capture(self):
        toolkit = self.root / "cann"
        toolkit.mkdir()
        metadata = toolkit / "version.info"
        metadata.write_text("Version=synthetic-cann-A\n", encoding="utf-8")
        hardware = {**self.runtime, "toolchain": read_cann_toolchain({"ASCEND_HOME_PATH": str(toolkit)})}
        config = {"paths": {"cannbench_repo": str(self.root / "bench")}}
        before = orchestrator.comparison_context(self.root / "task", hardware, config)
        metadata.unlink()
        with self.assertRaisesRegex(RuntimeError, "Cannot verify the CANN toolchain version"):
            orchestrator.comparison_context(self.root / "task", hardware, config)
        self.assertEqual(before["hardware"]["toolchain"]["version_files"][0]["version"], "synthetic-cann-A")

    def test_missing_cann_metadata_is_not_assumed_stable(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot verify the CANN toolchain version"):
            read_cann_toolchain({"ASCEND_HOME_PATH": str(self.root / "missing")})

    def test_resume_refreshes_hardware_before_any_stage(self):
        from test_fusion_routing import FusionRoutingTests
        fixture = FusionRoutingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint(stage="iter1_stage10", max_iterations=0)
        stale = json.loads((fixture.work / "device_info.json").read_text(encoding="utf-8"))
        fresh = {**stale, "chip_model": "fresh-real-chip", "soc_version": "fresh-real-soc",
                 "ub_size_kb": 192, "l0a_size_kb": 64, "device_id": 0}
        with fixture.patches(), patch.object(orchestrator, "detect_npu_device", return_value=fresh) as probe:
            orchestrator.main()
        probe.assert_called_once_with(device_id=0, config_path=str(fixture.work / "synthetic_config.yaml"))
        archived = json.loads((fixture.work / "device_info.json").read_text(encoding="utf-8"))
        self.assertEqual(archived["chip_model"], "fresh-real-chip")
        self.assertEqual(archived["ub_size_kb"], 192)


if __name__ == "__main__":
    unittest.main()

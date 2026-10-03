"""Exercise real stage-to-runner wiring with synthetic configs and transports."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import orchestrator
from lib import agent_runner, bench_parser, cann_env
from lib.framework_target import ensure_task_link
import test_fusion_routing as fusion


class TritonRuntimeWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("_cached_cann_cfg", "_cached_cann_env", "_cached_config_path"):
            self.stack.enter_context(patch.object(cann_env, name, None))

    def config_file(self, name, toolkit):
        path = self.root / name
        path.write_text(json.dumps({"cann": {"toolkit_path": str(toolkit)}}), encoding="utf-8")
        return str(path)

    def test_all_stages_use_selected_config_cli_and_device_through_real_runner(self):
        fixture = fusion.FusionRoutingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint(max_iterations=1)
        hermes_cli = fixture.work / "synthetic-hermes"
        hermes_cli.write_text(f'#!"{Path(sys.executable).as_posix()}"\n'
                              "from hermes_cli.main import main\n"
                              "if __name__ == '__main__':\n    raise SystemExit(main())\n", encoding="utf-8")
        hermes_cli.chmod(0o700)
        config_path = fixture.work / "synthetic_config.yaml"
        config_path.write_text(json.dumps({
            "paths": {"cannbench_repo": str(fixture.work / "absent_cannbench")},
            "hardware": {"device_id": 3}, "fusion_selection": {"top_n": 2},
            "agents": {name: {"cli": str(hermes_cli) if name == "hermes" else f"synthetic-custom-{name}"}
                       for name in ("cannbot", "kerminal", "hermes")},
        }), encoding="utf-8")
        actual_load_config = orchestrator.load_config
        actual_run_agent = agent_runner.run_agent
        active = {}
        observed = []

        def forward(*args, **kwargs):
            active.update(args=args, kwargs=kwargs)
            return actual_run_agent(*args, **kwargs)

        def transport(command, *, env, **kwargs):
            args = active["args"]
            if args[0] == "hermes":
                self.assertIn(str(hermes_cli), command)
                self.assertEqual(Path(command[0]), Path(sys.executable))
            else:
                self.assertIsNotNone(kwargs.get("stdin_path"))
            observed.append((fusion.ROLES[Path(args[1]).name],
                             str(hermes_cli) if args[0] == "hermes" else command[0], dict(env)))
            success = fixture.agent(*args, **active["kwargs"])
            return SimpleNamespace(returncode=0 if success else 1, stdout="", stderr="")

        def pty_transport(command, **kwargs):
            result = transport(command, **kwargs)
            return result.returncode, result.stdout

        with fixture.patches(), \
                patch.object(orchestrator, "load_config", side_effect=actual_load_config), \
                patch.object(orchestrator, "run_agent", side_effect=forward), \
                patch.object(agent_runner, "_run_with_heartbeat", side_effect=transport), \
                patch.object(agent_runner, "_run_with_pty", side_effect=pty_transport), \
                patch.object(cann_env, "build_cann_env", side_effect=lambda **kwargs: {
                    "PATH": "/synthetic/bin", "ASCEND_RT_VISIBLE_DEVICES": "4,7,8,9"}) as build_env:
            orchestrator.main()
            self.assertTrue(all(call.kwargs.get("config_path") == str(config_path)
                                for call in build_env.call_args_list))
            self.assertEqual(fixture.perf.call_args.kwargs["config_path"], str(config_path))
            self.assertEqual(fixture.perf.call_args.kwargs["device_id"], 3)

        expected = {"stage1": "cannbot", "stage2": "cannbot", "stage3": "cannbot",
                    "stage4": "kerminal", "stage5": "kerminal", "stage7": "kerminal",
                    "stage8": "hermes", "stage9": "kerminal", "stage10": "kerminal"}
        self.assertEqual({stage for stage, _, _ in observed}, set(expected))
        for stage, cli, env in observed:
            with self.subTest(stage=stage):
                self.assertEqual(cli, str(hermes_cli) if expected[stage] == "hermes"
                                 else "synthetic-custom-" + expected[stage])
                self.assertEqual(env["WORKFLOW_NPU_DEVICE_ID"], "3")
                self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "4,7,8,9")
        self.assertIsNone(agent_runner._workflow_agent_configs.get())

    def test_agent_configuration_is_removed_after_failure(self):
        @agent_runner.isolated_agent_configuration
        def failed_run():
            agent_runner.bind_workflow_agent_configuration(
                {"hardware": {"device_id": 5}, "agents": {"cannbot": {"cli": "synthetic"}}},
                str(self.root / "synthetic.yaml"))
            self.assertEqual(agent_runner._load_agent_config("cannbot")["device_id"], 5)
            raise RuntimeError("synthetic interruption")
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            failed_run()
        self.assertIsNone(agent_runner._workflow_agent_configs.get())

    def test_task_directory_mismatch_is_rejected_before_agents_or_measurement(self):
        fixture = fusion.FusionRoutingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint()
        original = fixture.work / "task/desc.md"
        original.write_text("original task", encoding="utf-8")
        other = fixture.work / "different_task"
        other.mkdir()
        with fixture.patches(), patch("sys.argv", ["orchestrator.py", "--task-dir", str(other),
                "--work-dir", str(fixture.work)]), self.assertRaisesRegex(ValueError, "task 与 --task-dir 不一致"):
            orchestrator.main()
        self.assertEqual(fixture.events, [])
        self.assertEqual(original.read_text(encoding="utf-8"), "original task")

    def test_task_link_accepts_same_directory_and_requires_existing_source(self):
        work = self.root / "work"
        task = work / "task"
        task.mkdir(parents=True)
        self.assertEqual(ensure_task_link(work, task), task.resolve())
        with self.assertRaisesRegex(ValueError, "task 目录不存在"):
            ensure_task_link(work, self.root / "missing")

    def test_eval_preserves_full_probe_environment_and_ctypes_compatibility(self):
        selected = {"ASCEND_HOME_PATH": "/selected/cann", "ASCEND_OPP_PATH": "/selected/opp",
                    "TRITON_ASCEND_HOME_PATH": "/selected/backend", "PATH": "/selected/python/bin",
                    "PYTHONPATH": "/selected/python", "LD_LIBRARY_PATH": "/selected/lib",
                    "ASCEND_RT_VISIBLE_DEVICES": "5,7", "LD_PRELOAD": "remove-me"}
        with patch.object(cann_env, "build_cann_env", return_value=dict(selected)) as build, \
                patch.object(bench_parser, "_find_good_ctypes_dir", return_value="/compatible/ctypes"):
            env = bench_parser._build_eval_env("synthetic-selected.yaml")
        build.assert_called_once_with(config_path="synthetic-selected.yaml")
        for key, value in selected.items():
            if key not in {"LD_PRELOAD", "PYTHONPATH"}:
                self.assertEqual(env[key], value)
        self.assertEqual(env["PYTHONPATH"], "/compatible/ctypes:/selected/python")
        self.assertNotIn("LD_PRELOAD", env)

    def test_cann_cache_is_scoped_to_selected_config_and_fallback_replaces_old_toolkit(self):
        first = self.config_file("first.yaml", self.root / "first-toolkit")
        second = self.config_file("second.yaml", self.root / "second-toolkit")
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": "/unrelated/old", "CANN_PATH": "/unrelated/old"}), \
                patch.object(cann_env, "_source_set_env_sh", return_value={}) as source:
            first_env = cann_env.build_cann_env(config_path=first)
            second_env = cann_env.build_cann_env(config_path=second)
            reused_env = cann_env.build_cann_env(config_path=second)
        self.assertEqual(source.call_count, 2)
        self.assertEqual(first_env["ASCEND_HOME_PATH"], str(self.root / "first-toolkit"))
        self.assertEqual(second_env["ASCEND_HOME_PATH"], str(self.root / "second-toolkit"))
        self.assertEqual(second_env["CANN_PATH"], str(self.root / "second-toolkit"))
        self.assertEqual(second_env, reused_env)

    def test_failed_set_env_is_not_treated_as_success_or_silent_fallback(self):
        toolkit = self.root / "toolkit"
        toolkit.mkdir()
        (toolkit / "set_env.sh").write_text("return 1\n", encoding="utf-8")
        config = self.config_file("synthetic.yaml", toolkit)
        with patch.object(cann_env.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="", stderr="synthetic failure")) as run, \
                self.assertRaisesRegex(RuntimeError, "set_env.sh 执行失败"):
            cann_env.build_cann_env(config_path=config)
        self.assertIn("&& env -0", run.call_args.args[0][-1])
        self.assertIsNone(cann_env._cached_cann_env)

    def test_repeated_main_refreshes_same_config_path_and_parent_device_map(self):
        fixture = fusion.FusionRoutingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.checkpoint()
        config = fixture.work / "synthetic_config.yaml"
        device_info = json.loads((fixture.work / "device_info.json").read_text(encoding="utf-8"))
        observed = []

        def probe(*, config_path, **kwargs):
            env = cann_env.build_cann_env(config_path=config_path)
            observed.append((env["ASCEND_HOME_PATH"], env["ASCEND_RT_VISIBLE_DEVICES"]))
            return dict(device_info)

        with fixture.patches(), \
                patch.object(orchestrator, "detect_npu_device", side_effect=probe), \
                patch.object(cann_env, "_source_set_env_sh", return_value={}) as source:
            for name, visibility in (("first", "5,7"), ("second", "2,4")):
                config.write_text(json.dumps({"cann": {"toolkit_path": f"/synthetic/{name}"}}),
                                  encoding="utf-8")
                with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": visibility}):
                    orchestrator.main()
        self.assertEqual(observed, [("/synthetic/first", "5,7"), ("/synthetic/second", "2,4")])
        self.assertEqual(source.call_count, 2)

    def test_login_shell_cannot_replace_callers_visible_device_map(self):
        config = self.config_file("synthetic.yaml", self.root / "toolkit")
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "5,7"}), \
                patch.object(cann_env, "_source_set_env_sh", return_value={
                    "ASCEND_RT_VISIBLE_DEVICES": "0", "PATH": "/selected/bin"}):
            env = cann_env.build_cann_env(config_path=config)
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "5,7")
        self.assertIn("/usr/local/Ascend/driver/lib64/common", env["LD_LIBRARY_PATH"].split(":"))


if __name__ == "__main__":
    unittest.main()

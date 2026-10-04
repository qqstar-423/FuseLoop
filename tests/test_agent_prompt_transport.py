"""Exercise complete prompts through the real runner with a mocked child process."""

from contextlib import ExitStack
import errno
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from lib import agent_runner as runner


class AgentPromptTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / "work 中文 with spaces"
        self.work.mkdir()
        self.role = self.work / "stage3_modify.md"
        self.role.write_text("You are the operator development node.\nKeep all P0s.", encoding="utf-8")
        self.config = {"cli": "synthetic-cannbot", "cwd": str(self.work / "different cwd")}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "_agent_env", return_value={
            "PATH": "/synthetic/bin", "SYNTHETIC_SECRET": "do-not-log-this-value"}))

    def completed_process(self, returncode=0, stdout="", stderr=""):
        proc = Mock()
        proc.poll.return_value = returncode
        proc.returncode = returncode
        proc.wait.return_value = returncode
        proc.stdout = io.StringIO(stdout)
        proc.stderr = io.StringIO(stderr)
        proc.communicate.side_effect = AssertionError("Do not mix text reads with communicate")
        return proc

    def test_large_chinese_prompt_is_exact_stdin_not_an_argument(self):
        prompt = "Per-case analysis; file /tmp/a b; literal $(never-run) `never-run`\r\n" * 40000
        expected = (self.role.read_text(encoding="utf-8") + "\n\n---\n\n"
                    + prompt + runner.GLOBAL_CONSTRAINT).encode("utf-8")
        observed = {}
        node_log = Mock()

        def launch(command, **kwargs):
            stream = kwargs["stdin"]
            observed.update(stream=stream, path=Path(stream.name))
            self.assertFalse(stream.isatty())
            self.assertEqual(stream.read(), expected)
            self.assertEqual(command, ["synthetic-cannbot", "run", "--format", "json",
                                       "--dangerously-skip-permissions"])
            self.assertLess(sum(len(arg.encode("utf-8")) for arg in command), 200)
            self.assertEqual(kwargs["cwd"], self.config["cwd"])
            self.assertEqual(kwargs["env"]["SYNTHETIC_SECRET"], "do-not-log-this-value")
            self.assertNotIn("shell", kwargs)
            return self.completed_process()

        with patch.object(runner.subprocess, "Popen", side_effect=launch), \
                self.assertLogs(runner.log, level="INFO") as logs:
            self.assertTrue(runner.run_agent("cannbot", str(self.role), str(self.work), prompt,
                                             config=self.config, node_log=node_log))
        self.assertGreater(len(expected), 2 * 1024 * 1024)
        self.assertTrue(observed["stream"].closed)
        self.assertEqual(observed["path"].parent, self.work.resolve() / "log/prompts")
        self.assertEqual(observed["path"].read_bytes(), expected)
        self.assertTrue(observed["path"].name.startswith("cannbot_stage3_modify_"))
        if os.name == "posix":
            self.assertEqual(observed["path"].stat().st_mode & 0o777, 0o600)
        logged = "\n".join(logs.output)
        self.assertIn(str(observed["path"]), logged)
        self.assertIn(f"UTF-8 bytes={len(expected)}", logged)
        self.assertNotIn("never-run", logged)
        self.assertNotIn("do-not-log-this-value", logged)
        self.assertTrue(any(str(observed["path"]) in item.args[0]
                            for item in node_log.info.call_args_list))

    def test_repeated_stage_calls_preserve_separate_inputs(self):
        with patch.object(runner.subprocess, "Popen", side_effect=lambda *a, **kw: self.completed_process()):
            for prompt in ("first task", "same-round correction task"):
                self.assertTrue(runner.run_cannbot(str(self.role), str(self.work), prompt, self.config))
        paths = list((self.work / "log/prompts").glob("*.txt"))
        self.assertEqual(len(paths), 2)
        suffixes = {path.read_text(encoding="utf-8").split("\n\n---\n\n")[-1] for path in paths}
        self.assertEqual(suffixes, {"first task", "same-round correction task"})

    def test_failed_launch_closes_input_logs_sizes_and_propagates_error(self):
        opened = []
        failure = OSError(errno.E2BIG, "Argument list too long")

        def launch(command, **kwargs):
            opened.append(kwargs["stdin"])
            raise failure

        with patch.object(runner.subprocess, "Popen", side_effect=launch) as popen, \
                self.assertLogs(runner.log, level="INFO") as logs, \
                self.assertRaises(OSError) as caught:
            runner.run_cannbot(str(self.role), str(self.work), "synthetic task", self.config)
        self.assertIs(caught.exception, failure)
        popen.assert_called_once()
        self.assertTrue(opened[0].closed)
        self.assertTrue(Path(opened[0].name).exists())
        logged = "\n".join(logs.output)
        self.assertIn("process never started", logged)
        self.assertIn("argv bytes=", logged)
        self.assertIn("env bytes=", logged)
        self.assertNotIn("do-not-log-this-value", logged)

    def test_missing_executable_error_is_not_swallowed_or_retried(self):
        failure = FileNotFoundError(errno.ENOENT, "synthetic missing executable")
        with patch.object(runner.subprocess, "Popen", side_effect=failure) as popen, \
                self.assertRaises(FileNotFoundError) as caught:
            runner.run_cannbot(str(self.role), str(self.work), "task", self.config)
        self.assertIs(caught.exception, failure)
        popen.assert_called_once()

    def test_model_context_error_still_returns_failure_and_distinguishes_os_limit(self):
        proc = self.completed_process(returncode=1, stderr="Prompt exceeds max length")
        with patch.object(runner.subprocess, "Popen", return_value=proc), \
                self.assertLogs(runner.log, level="ERROR") as logs:
            self.assertFalse(runner.run_cannbot(str(self.role), str(self.work), "task", self.config))
        self.assertIn("model context limit exceeded", "\n".join(logs.output))

    def test_stdin_does_not_replace_stdout_streaming(self):
        path = self.work / "input.txt"
        path.write_bytes("complete task".encode("utf-8"))
        proc = self.completed_process(
            stdout='{"type":"text","part":{"text":"synthetic response"}}\ntail',
            stderr="separate error stream")
        with patch.object(runner.subprocess, "Popen", return_value=proc), \
                self.assertLogs(runner.log, level="INFO") as logs:
            result = runner._run_with_heartbeat(["synthetic"], str(self.work), {}, "test",
                                                stdin_path=str(path))
        self.assertEqual(result.returncode, 0)
        self.assertIn("synthetic response", result.stdout)
        self.assertTrue(result.stdout.endswith("tail"))
        self.assertEqual(result.stderr, "separate error stream")
        self.assertNotIn("separate error stream", result.stdout)
        self.assertTrue(proc.stdout.closed)
        self.assertTrue(proc.stderr.closed)
        self.assertIn("synthetic response", "\n".join(logs.output))

    def test_heartbeat_without_prompt_keeps_inherited_stdin(self):
        with patch.object(runner.subprocess, "Popen", return_value=self.completed_process()) as popen:
            runner._run_with_heartbeat(["synthetic"], str(self.work), {}, "test")
        self.assertIsNone(popen.call_args.kwargs["stdin"])

    def test_pty_startup_failure_closes_both_descriptors(self):
        fake_pty = SimpleNamespace(openpty=Mock(return_value=(101, 102)))
        fake_fcntl = SimpleNamespace(ioctl=Mock())
        fake_termios = SimpleNamespace(TIOCSWINSZ=1)
        with patch.object(runner, "pty", fake_pty), \
                patch.dict(sys.modules, {"fcntl": fake_fcntl, "termios": fake_termios}), \
                patch.object(runner.os, "close") as close, \
                patch.object(runner.subprocess, "Popen", side_effect=OSError(errno.E2BIG, "synthetic")), \
                self.assertRaises(OSError):
            runner._run_with_pty(["synthetic"], str(self.work), {})
        self.assertEqual(close.call_args_list, [call(101), call(102)])

    def test_missing_cli_fails_before_writing_prompt(self):
        with self.assertRaisesRegex(ValueError, "cli path is not configured"):
            runner.run_cannbot(str(self.role), str(self.work), "task", config={})
        self.assertFalse((self.work / "log/prompts").exists())

    def hermes_cli(self):
        path = self.work / "hermes console script"
        path.write_text(f'#!"{Path(sys.executable).as_posix()}"\n'
                        "from hermes_cli.main import main\n"
                        "if __name__ == '__main__':\n    raise SystemExit(main())\n", encoding="utf-8")
        path.chmod(0o700)
        return str(path)

    def test_kerminal_multi_megabyte_prompt_uses_exec_stdin_without_pty(self):
        prompt = "Plenty of history and human P0s; all text must be preserved.\r\n" * 80000
        expected = (self.role.read_text(encoding="utf-8") + "\n\n---\n\n"
                    + prompt + runner.GLOBAL_CONSTRAINT).encode("utf-8")
        self.assertGreater(len(expected), 2 * 1024 * 1024)
        config = {**self.config, "cli": "synthetic-kerminal"}
        opened = []

        def launch(command, **kwargs):
            self.assertEqual(command, ["synthetic-kerminal", "-a", "never", "exec",
                                       "--skip-git-repo-check", "-C", config["cwd"], "-"])
            self.assertEqual(kwargs["stdin"].read(), expected)
            self.assertEqual(kwargs["cwd"], config["cwd"])
            opened.append(kwargs["stdin"])
            return self.completed_process()

        with patch.object(runner.subprocess, "Popen", side_effect=launch), \
                patch.object(runner, "_run_with_pty", side_effect=AssertionError("PTY must not start")):
            self.assertTrue(runner.run_agent("kerminal", str(self.role), str(self.work), prompt,
                                             config=config))
        self.assertTrue(opened[0].closed)
        self.assertEqual(Path(opened[0].name).read_bytes(), expected)
        self.assertTrue(Path(opened[0].name).name.startswith("kerminal_stage3_modify_"))

    def test_hermes_multi_megabyte_prompt_never_enters_launch_arguments(self):
        prompt = "Plenty of history and search directions; keep newlines.\r\n" * 90000
        expected = (self.role.read_text(encoding="utf-8") + "\n\n---\n\n"
                    + prompt + runner.GLOBAL_CONSTRAINT).encode("utf-8")
        self.assertGreater(len(expected), 2 * 1024 * 1024)
        config = {**self.config, "cli": self.hermes_cli()}
        captured = []

        def launch(command, **kwargs):
            self.assertNotIn(prompt, command)
            self.assertLess(max(len(arg.encode("utf-8")) for arg in command), 4096)
            files = [Path(arg) for arg in command if str(arg).endswith(".txt")]
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), expected)
            self.assertEqual(command[-5:], ["-t", "web,file", "--yolo", "--in", config["cwd"]])
            self.assertEqual(kwargs["cwd"], config["cwd"])
            self.assertIsNone(kwargs["stdin"])
            self.assertIn(config["cli"], command)
            captured.append(files[0])
            return self.completed_process()

        with patch.object(runner.subprocess, "Popen", side_effect=launch):
            self.assertTrue(runner.run_agent("hermes", str(self.role), str(self.work), prompt,
                                             config=config))
        self.assertTrue(captured[0].name.startswith("hermes_stage3_modify_"))

    def test_kerminal_and_hermes_failures_cannot_turn_into_success(self):
        # A pre-existing search report must not hide a bridge/CLI failure.
        report = self.work / "search/iter1/SEARCH_REPORT.md"
        report.parent.mkdir(parents=True)
        report.write_text("old result", encoding="utf-8")
        for name in ("kerminal", "hermes"):
            with self.subTest(agent=name):
                config = {**self.config, "cli": self.hermes_cli() if name == "hermes" else "synthetic-kerminal"}
                with patch.object(runner.subprocess, "Popen", return_value=self.completed_process(returncode=2)):
                    self.assertFalse(runner.run_agent(name, str(self.role), str(self.work), "task", config=config))

    def test_kerminal_launch_error_keeps_prompt_and_propagates(self):
        opened = []

        def launch(command, **kwargs):
            opened.append(kwargs["stdin"])
            raise OSError(errno.E2BIG, "synthetic environment exceeds limit")

        with patch.object(runner.subprocess, "Popen", side_effect=launch), self.assertRaises(OSError):
            runner.run_kerminal(str(self.role), str(self.work), "synthetic task", {"cli": "synthetic-kerminal"})
        self.assertTrue(opened[0].closed)
        self.assertTrue(Path(opened[0].name).exists())


if __name__ == "__main__":
    unittest.main()

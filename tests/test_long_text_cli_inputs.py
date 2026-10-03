"""File-backed user text reaches existing prompts/manifests without argv growth."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import orchestrator
from tools import human_review
import test_init_impl_routing as routing_fixture


class LongTextCLIInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = "开始：完整保留。\n" + ("中文方向及 case1 精度要求，保留所有步骤。\n" * 7000) + "结束：不要截断。\n"
        self.text_file = self.root / "有 空格的方向说明.md"
        self.text_file.write_text(self.text, encoding="utf-8-sig")

    def routing(self):
        fixture = routing_fixture.InitImplRoutingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_hint_file_reaches_emergency_manifest_unchanged(self):
        fixture = self.routing()
        fixture.stop_at = "stage4"
        with fixture.patches(), patch("sys.argv", [
                "orchestrator.py", "--task-dir", str(fixture.source / "task"),
                "--init-impl", str(fixture.source / "impl"),
                "--optimize-hint-file", str(self.text_file)]), self.assertRaises(routing_fixture.PauseFixture):
            orchestrator.main()
        manifest = json.loads((fixture.work / "init_impl_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["optimize_hint"], self.text)
        self.assertEqual(fixture.events, ["stage4"])
        fixture.jev.assert_not_called()

    def test_hint_file_reaches_stage1_prompt_without_truncation(self):
        fixture = self.routing()
        prompts = []
        def stop_at_stage1(agent, role, work_dir, prompt, **kwargs):
            self.assertIn("stage1_", str(role))
            prompts.append(prompt)
            raise routing_fixture.PauseFixture("stage1")
        with fixture.patches(), patch("sys.argv", [
                "orchestrator.py", "--task-dir", str(fixture.source / "task"),
                "--work-dir", str(fixture.work), "--optimize-hint-file", str(self.text_file)]), \
                patch.object(orchestrator, "fusion_requirements_byte_budget", return_value=6000), \
                patch.object(orchestrator, "run_agent", side_effect=stop_at_stage1), \
                self.assertRaises(routing_fixture.PauseFixture):
            orchestrator.main()
        self.assertEqual(len(prompts), 1)
        self.assertIn(self.text, prompts[0])

    def test_existing_inline_hint_semantics_are_preserved(self):
        fixture = self.routing()
        fixture.stop_at = "stage4"
        hint = "短方向：保持当前融合方法"
        with fixture.patches(), patch("sys.argv", [
                "orchestrator.py", "--task-dir", str(fixture.source / "task"),
                "--init-impl", str(fixture.source / "impl"), "--optimize-hint", hint]), \
                self.assertRaises(routing_fixture.PauseFixture):
            orchestrator.main()
        manifest = json.loads((fixture.work / "init_impl_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["optimize_hint"], hint)

    def parse_failure(self, args):
        stderr = io.StringIO()
        with patch("sys.argv", ["orchestrator.py", "--task-dir", "synthetic-task", *args]), \
                patch.object(orchestrator, "load_config") as load_config, redirect_stderr(stderr), \
                self.assertRaises(SystemExit) as stopped:
            orchestrator.main()
        self.assertEqual(stopped.exception.code, 2)
        load_config.assert_not_called()
        self.assertNotIn(self.text, stderr.getvalue())
        return stderr.getvalue()

    def test_hint_sources_are_mutually_exclusive_without_echoing_text(self):
        message = self.parse_failure(["--optimize-hint", self.text, "--optimize-hint-file", str(self.text_file)])
        self.assertIn("not allowed with argument", message)
        self.assertIn("--optimize-hint", message)
        self.assertLess(len(message), 2000)

    def test_missing_hint_file_fails_before_loading_config(self):
        missing = self.root / "不存在 的文件.md"
        message = self.parse_failure(["--optimize-hint-file", str(missing)])
        self.assertIn("无法读取 UTF-8 文件", message)
        self.assertIn("FileNotFoundError", message)
        self.assertIn(str(missing), message)

    def test_invalid_utf8_hint_file_reports_type_without_contents(self):
        bad = self.root / "非 UTF8.bin"
        bad.write_bytes(b"\xff\xff PRIVATE_LONG_TEXT")
        message = self.parse_failure(["--optimize-hint-file", str(bad)])
        self.assertIn("UnicodeDecodeError", message)
        self.assertNotIn("PRIVATE_LONG_TEXT", message)

    def test_directory_is_not_accepted_as_hint_file(self):
        message = self.parse_failure(["--optimize-hint-file", str(self.root)])
        self.assertIn("无法读取 UTF-8 文件", message)

    def test_existing_human_file_input_delivers_all_text(self):
        output = io.StringIO()
        with patch.object(human_review, "submit_message", return_value={"message_id": "offline"}) as submit, \
                redirect_stdout(output):
            code = human_review.main(["--work-dir", str(self.root), "--file", str(self.text_file)])
        self.assertEqual(code, 0)
        self.assertEqual(submit.call_args.args[1], self.text)

    def test_human_help_explains_file_input_for_long_opinions(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            human_review.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("长意见请用 --file", output.getvalue())

    def test_human_inline_and_file_inputs_remain_mutually_exclusive(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), patch.object(human_review, "submit_message") as submit, \
                self.assertRaises(SystemExit) as stopped:
            human_review.main(["--work-dir", str(self.root), "--message", self.text, "--file", str(self.text_file)])
        self.assertEqual(stopped.exception.code, 2)
        submit.assert_not_called()
        self.assertNotIn(self.text, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

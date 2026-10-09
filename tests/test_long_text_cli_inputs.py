"""File-backed user text reaches existing prompts/manifests without argv growth."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import orchestrator
import test_init_impl_routing as routing_fixture


class LongTextCLIInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = "Start: keep everything.\n" + ("Chinese direction and case1 precision requirements, keep every step.\n" * 7000) + "End: do not truncate.\n"
        self.text_file = self.root / "direction notes with spaces.md"
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
        hint = "short direction: keep the current fusion method"
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
        missing = self.root / "does not exist.md"
        message = self.parse_failure(["--optimize-hint-file", str(missing)])
        self.assertIn("cannot read UTF-8 file", message)
        self.assertIn("FileNotFoundError", message)
        self.assertIn(str(missing), message)

    def test_invalid_utf8_hint_file_reports_type_without_contents(self):
        bad = self.root / "not-utf8.bin"
        bad.write_bytes(b"\xff\xff PRIVATE_LONG_TEXT")
        message = self.parse_failure(["--optimize-hint-file", str(bad)])
        self.assertIn("UnicodeDecodeError", message)
        self.assertNotIn("PRIVATE_LONG_TEXT", message)

    def test_directory_is_not_accepted_as_hint_file(self):
        message = self.parse_failure(["--optimize-hint-file", str(self.root)])
        self.assertIn("cannot read UTF-8 file", message)


if __name__ == "__main__":
    unittest.main()

"""A failed or dry run must never reuse an earlier successful Jev answer."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.jev_client import JevSettings
from tools.jev_smoke_test import main


class SmokeArtifactTests(unittest.TestCase):
    def test_success_failure_and_dry_run_have_isolated_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = ["jev_smoke_test.py", "--output-dir", folder]
            with patch("tools.jev_smoke_test.load_settings", return_value=JevSettings("fake-test-key")), \
                    patch("sys.argv", argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch("tools.jev_smoke_test.evaluate", return_value={"model": "test", "answers": {}}):
                    self.assertEqual(main(), 0)
                with patch("tools.jev_smoke_test.evaluate", side_effect=RuntimeError("synthetic failure")):
                    self.assertEqual(main(), 1)
                with patch("tools.jev_smoke_test.evaluate") as evaluate, patch("sys.argv", argv + ["--dry-run"]):
                    self.assertEqual(main(), 0)
                    evaluate.assert_not_called()
            runs = list(Path(folder).iterdir())
            self.assertEqual(len(runs), 3)
            by_status = {json.loads((p / "summary.json").read_text())["status"]: p for p in runs}
            self.assertEqual(set(by_status), {"success", "failed", "dry_run"})
            self.assertTrue((by_status["success"] / "response.json").exists())
            for status in ("failed", "dry_run"):
                self.assertFalse((by_status[status] / "response.json").exists())
            self.assertTrue(all((p / "request.json").exists() for p in runs))


if __name__ == "__main__":
    unittest.main()

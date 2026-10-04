import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.prompt_files import file_hint


class PromptFileHintTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.gettempdir()) / "prompt-file-hints" / "work"

    def test_fixed_file_keeps_actual_and_relative_locations(self):
        path = self.work / "fusion/fusion_library.json"
        hint = file_hint(self.work, path, "initial fusion library", "look at probability")
        self.assertIn(str(path), hint)
        self.assertIn("relative to working directory: `fusion/fusion_library.json`", hint)
        self.assertIn("Purpose: initial fusion library; how to read: look at probability", hint)
        self.assertNotIn("iteration template", hint)
        self.assertEqual(hint.count("\n"), 1)
        self.assertTrue(hint.endswith("\n"))

    def test_iteration_keeps_concrete_path_and_adds_template(self):
        hint = file_hint(self.work, self.work / "develop/iter13/self_test_result.json", "self-test results", "look at pass statuses")
        self.assertIn("develop/iter13/self_test_result.json", hint)
        self.assertIn("iteration template: `develop/<iter>/self_test_result.json`", hint)

    def test_snapshot_template_replaces_iteration_and_fingerprint(self):
        hint = file_hint(self.work, self.work / "selection/records/iter3-a1b2c3/manifest.json", "snapshot manifest", "verify metrics")
        self.assertIn("selection/records/iter3-a1b2c3/manifest.json", hint)
        self.assertIn("selection/records/<iter>-<fingerprint>/manifest.json", hint)

    def test_external_project_uses_its_explicit_base(self):
        project = self.work.parent / "project"
        hint = file_hint(self.work, project / "knowledge/fusion_options.json", "fusion method catalog", "look at method categories",
                         base_dir=project, base_label="project root")
        self.assertIn("relative to project root: `knowledge/fusion_options.json`", hint)
        self.assertNotIn("../project", hint)

    def test_task_path_is_lexical_and_never_resolves_link_target(self):
        # work/task may be a symlink or Windows junction. No filesystem access
        # or link resolution is needed to describe its intended prompt location.
        path = self.work / "task/input.json"
        with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve symlinks")), \
                patch.object(os.path, "realpath", side_effect=AssertionError("must not resolve symlinks")):
            hint = file_hint(self.work, path, "operator inputs", "look at cases")
        self.assertIn("relative to working directory: `task/input.json`", hint)
        self.assertIn(str(path), hint)


if __name__ == "__main__":
    unittest.main()

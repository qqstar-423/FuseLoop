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
        hint = file_hint(self.work, path, "初始融合库", "看 probability")
        self.assertIn(str(path), hint)
        self.assertIn("相对工作目录：`fusion/fusion_library.json`", hint)
        self.assertIn("用途：初始融合库；怎么看：看 probability", hint)
        self.assertNotIn("迭代模板", hint)
        self.assertEqual(hint.count("\n"), 1)
        self.assertTrue(hint.endswith("\n"))

    def test_iteration_keeps_concrete_path_and_adds_template(self):
        hint = file_hint(self.work, self.work / "develop/iter13/self_test_result.json", "自测结果", "看通过状态")
        self.assertIn("develop/iter13/self_test_result.json", hint)
        self.assertIn("迭代模板：`develop/<iter>/self_test_result.json`", hint)

    def test_snapshot_template_replaces_iteration_and_fingerprint(self):
        hint = file_hint(self.work, self.work / "selection/records/iter3-a1b2c3/manifest.json", "快照清单", "核对指标")
        self.assertIn("selection/records/iter3-a1b2c3/manifest.json", hint)
        self.assertIn("selection/records/<iter>-<指纹>/manifest.json", hint)

    def test_external_project_uses_its_explicit_base(self):
        project = self.work.parent / "project"
        hint = file_hint(self.work, project / "knowledge/fusion_options.json", "融合方法目录", "看方法分类",
                         base_dir=project, base_label="项目根目录")
        self.assertIn("相对项目根目录：`knowledge/fusion_options.json`", hint)
        self.assertNotIn("../project", hint)

    def test_task_path_is_lexical_and_never_resolves_link_target(self):
        # work/task may be a symlink or Windows junction. No filesystem access
        # or link resolution is needed to describe its intended prompt location.
        path = self.work / "task/input.json"
        with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve symlinks")), \
                patch.object(os.path, "realpath", side_effect=AssertionError("must not resolve symlinks")):
            hint = file_hint(self.work, path, "算子输入", "看 case")
        self.assertIn("相对工作目录：`task/input.json`", hint)
        self.assertIn(str(path), hint)


if __name__ == "__main__":
    unittest.main()

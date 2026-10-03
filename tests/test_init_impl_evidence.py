"""Imported self-tests must stay tied to the source revision that actually ran."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lib.fusion_evidence import (
    DECISION_FILE, begin_development, finalize_development,
    format_evidence_for_prompt, load_evidence, restore_imported_evidence,
)
from lib.init_impl import _rebaser


class ImportedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.work = Path(self.temp.name) / "new"
        self.develop = self.source / "develop/iter2"
        self.develop.mkdir(parents=True)
        (self.source / "impl").mkdir()
        (self.source / "impl/kernel.py").write_text("def kernel(x): return x + 1\n", encoding="utf-8")
        initial = {"schema_version": 1, "candidates": [
            {"rank": 1, "method": {"id": "pipeline", "name": "Pipeline"}, "probability": 0.8}]}
        library = {**deepcopy(initial), "selection": {
            "method_ids": ["pipeline"], "implementation_plan": "Tile the reduction.",
            "reason": "Avoid intermediate traffic.", "target_cases": ["all"],
            "actual_changes": ["Tile the reduction."], "expected_benefits": ["Less traffic."]}}
        self.self_test = {"schema_version": 1,
            "provided_cases": {"executed": True, "passed": True, "total": 2,
                "passed_cases": 2, "evidence_path": str(self.develop / "self_test.log")},
            "continuous_calls": {"executed": True, "passed": True, "same_shape": True,
                "call_count": 2, "reference_checked_each_call": True,
                "evidence_path": str(self.develop / "self_test.log"), "changes": {
                    "inputs": {"applicable": True, "changed": True, "passed": True},
                    "weights": {"applicable": False, "reason": "No weights."},
                    "bias": {"applicable": False, "reason": "No bias."}}}}
        self.write_json(self.source / "fusion/fusion_library.json", initial)
        self.write_json(self.develop / "fusion_library.json", library)
        self.write_json(self.develop / "self_test_result.json", self.self_test)
        for name in (DECISION_FILE, "design_rationale.md", "self_test_report.md", "self_test.log"):
            (self.develop / name).write_text("Actual executed developer evidence.\n", encoding="utf-8")
        self.make_import()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_import(self, stage=3):
        self.assertTrue(finalize_development(self.source, self.develop, stage)["eligible"])
        source_binding = json.loads((self.source / "selection/current_implementation.json").read_text(encoding="utf-8"))
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir()
        self.manifest = {"source_work_dir": str(self.source),
            "source_impl_dir": str(self.source / "impl"), "source_develop_dir": str(self.develop),
            "source_development_binding": source_binding, "copied_files": []}
        rebase = _rebaser(self.source, self.source / "impl", self.work, self.develop)
        for source_dir, target_dir in ((self.source / "impl", self.work / "impl"),
                (self.source / "fusion", self.work / "fusion"),
                (self.develop, self.work / "develop/iter0")):
            for source in source_dir.rglob("*"):
                if not source.is_file():
                    continue
                target = target_dir / source.relative_to(source_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if source.suffix == ".json":
                    value = json.loads(target.read_text(encoding="utf-8"))
                    self.write_json(target, rebase(value))
                self.manifest["copied_files"].append({"source": str(source),
                    "destination": target.relative_to(self.work).as_posix(),
                    "source_sha256": self.sha(source), "sha256": self.sha(target)})
        self.write_json(self.work / "init_impl_manifest.json", self.manifest)

    def restore(self):
        return restore_imported_evidence(self.work, self.manifest)

    def assert_ineligible(self, reason):
        result = self.restore()
        self.assertFalse(result["eligible"])
        self.assertIn(reason, result["reason"])
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertFalse((self.work / "selection/best.json").exists())
        return result

    def test_unchanged_stage3_code_restores_relocated_developer_binding_only(self):
        result = self.restore()
        self.assertTrue(result["eligible"])
        self.assertTrue(load_evidence(self.work)["eligible"])
        self.assertEqual(load_evidence(self.work)["stage"], 3)
        self.assertEqual(result["evidence_paths"]["decision_rationale"],
                         str(self.work / "develop/iter0" / DECISION_FILE))
        self.assertTrue(all(Path(path).is_relative_to(self.work)
                            for path in result["evidence_paths"].values()))
        self.assertEqual([path.name for path in (self.work / "selection").iterdir()],
                         ["current_implementation.json"])
        prompt = format_evidence_for_prompt(self.work)
        self.assertIn("对应代码与原开发证据已核对一致", prompt)
        self.assertIn("init_impl_manifest.json", prompt)
        self.assertIn("相对工作目录：`develop/iter0/", prompt)

    def test_stage2_and_relative_self_test_log_paths_also_restore(self):
        for item in (self.self_test["provided_cases"], self.self_test["continuous_calls"]):
            item["evidence_path"] = "self_test.log"
        self.write_json(self.develop / "self_test_result.json", self.self_test)
        self.make_import(stage=2)
        self.assertTrue(self.restore()["eligible"])
        self.assertTrue(load_evidence(self.work)["evidence_paths"]["decision_rationale"].endswith("design_rationale.md"))

    def test_changed_code_retains_reference_documents_without_reusing_passing_result(self):
        (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 2\n", encoding="utf-8")
        self.assert_ineligible("differs from the source")
        prompt = format_evidence_for_prompt(self.work)
        self.assertIn("原自测结果只作历史参考", prompt)
        self.assertIn("Stage9 应安排 Stage3", prompt)
        self.assertIn("不进入最佳记录和停滞窗口", prompt)
        self.assertIn(DECISION_FILE, prompt)

    def test_source_document_edit_after_binding_invalidates_import(self):
        for entry in self.manifest["copied_files"]:
            if entry["source"].endswith("self_test.log"):
                entry["source_sha256"] = "changed-before-import"
        self.assert_ineligible("changed before import")

    def test_copied_document_edit_invalidates_import(self):
        (self.work / "develop/iter0/self_test.log").write_text("Edited log", encoding="utf-8")
        self.assert_ineligible("changed after import")

    def test_changed_initial_probabilities_invalidate_import(self):
        initial_path = self.work / "fusion/fusion_library.json"
        value = json.loads(initial_path.read_text(encoding="utf-8"))
        value["candidates"][0]["probability"] = 0.9
        self.write_json(initial_path, value)
        self.assert_ineligible("Jev candidates differ")

    def test_unbound_or_failed_source_self_tests_do_not_gain_qualification(self):
        self.manifest["source_development_binding"]["eligible"] = False
        self.assert_ineligible("no eligible developer")
        self.manifest["source_development_binding"] = None
        self.assert_ineligible("no eligible developer")

    def test_source_binding_cannot_be_taken_from_another_developer_round(self):
        self.manifest["source_development_binding"]["develop_dir"] = str(self.source / "develop/iter3")
        self.assert_ineligible("does not match its original binding")

    def test_uncopied_external_self_test_log_does_not_gain_qualification(self):
        binding = self.manifest["source_development_binding"]
        binding["evidence_paths"]["selftest_log_provided"] = str(self.source / "eval/iter1/old.log")
        self.assert_ineligible("outside the copied developer directory")

    def test_destination_path_escape_or_misdirection_is_rejected(self):
        entry = next(item for item in self.manifest["copied_files"] if item["source"].endswith("self_test.log"))
        entry["destination"] = str(self.develop / "self_test.log")
        self.assert_ineligible("destination is invalid")
        entry["destination"] = "../source/develop/iter2/self_test.log"
        self.assert_ineligible("within the workflow directory")

    def test_source_summary_cannot_override_copied_fusion_decision(self):
        self.manifest["source_development_binding"]["implementation_plan"] = "A different decision"
        self.assert_ineligible("Copied fusion decisions differ")

    def test_path_only_relocation_inside_fusion_decisions_keeps_valid_binding(self):
        path = self.develop / "fusion_library.json"
        library = json.loads(path.read_text(encoding="utf-8"))
        library["selection"]["implementation_plan"] = f"Tile the kernel in {self.source / 'impl/kernel.py'} ."
        library["selection"]["actual_changes"] = [f"See {self.develop / DECISION_FILE} ."]
        self.write_json(path, library)
        self.make_import()
        self.assertTrue(self.restore()["eligible"])
        result = load_evidence(self.work)
        self.assertIn(str(self.work / "impl/kernel.py"), result["implementation_plan"])
        self.assertIn(str(self.work / "develop/iter0" / DECISION_FILE),
                      result["fusion_scheme"]["actual_changes"][0])

    def test_code_edit_while_rebinding_is_detected(self):
        original = finalize_development

        def changed_before_finalize(*args, **kwargs):
            (self.work / "impl/kernel.py").write_text("def kernel(x): return x + 3\n", encoding="utf-8")
            return original(*args, **kwargs)

        with patch("lib.fusion_evidence.finalize_development", side_effect=changed_before_finalize):
            self.assert_ineligible("changed while restoring")

    def test_new_stage3_development_removes_import_status_and_requires_fresh_artifacts(self):
        self.assertTrue(self.restore()["eligible"])
        begin_development(self.work, self.work / "develop/iter1")
        self.assertFalse(load_evidence(self.work)["eligible"])
        self.assertNotIn("应急导入的开发材料", format_evidence_for_prompt(self.work))

    def test_optimize_hint_is_context_for_review_not_an_approved_human_task(self):
        self.manifest["optimize_hint"] = "Explore a smaller tile on the longest case."
        self.assertTrue(self.restore()["eligible"])
        prompt = format_evidence_for_prompt(self.work)
        self.assertIn(self.manifest["optimize_hint"], prompt)
        self.assertIn("首轮先实测，不是已批准的 P0", prompt)


if __name__ == "__main__":
    unittest.main()

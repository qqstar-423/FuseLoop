"""Offline emergency-import boundaries, provenance and restart validation."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lib.fusion_evidence import implementation_hash, _initial_library_hash
from lib.init_impl import (MANIFEST, load_init_impl_manifest, prepare_init_impl,
                           validate_init_impl_inputs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_source(root, stage=2):
    """Build complete synthetic Stage1/1.5 plus the matching developer binding."""
    work = Path(root) / "source"
    for name in ("impl/cann_bench", "task", "fusion/translation", "develop/iter0", "selection"):
        (work / name).mkdir(parents=True, exist_ok=True)
    (work / "impl/cann_bench/__init__.py").write_text("import triton\n", encoding="utf-8")
    (work / "impl/build.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (work / "impl/setup.py").write_text("# synthetic fixture\n", encoding="utf-8")
    (work / "task/cases.yaml").write_text("cases: [1, 2]\n", encoding="utf-8")
    (work / "task/golden.py").write_text("def golden(x): return x\n", encoding="utf-8")
    (work / "ANALYSIS.md").write_text(f"Analysis {work / 'impl'}; old history {work / 'history.json'}\n", encoding="utf-8")
    requirements = {"language": "en", "operator_summary": "Synthetic operator", "semantics": "identity",
                    "case_groups": ["float16"], "implementation_constraints": ["Triton Ascend"],
                    "optimization_hint": ""}
    device = {"chip_model": "Ascend-test", "soc_version": "test-v1", "npu_arch": "test",
              "ai_core_num": 20, "ub_size_kb": 192, "l1_size_kb": 512, "detect_method": "fixture"}
    write_json(work / "fusion_requirements.en.json", requirements)
    write_json(work / "device_info.json", device)
    candidates = [{"rank": 1, "method": {"id": "vertical", "name": "Vertical fusion"}, "probability": .8},
                  {"rank": 2, "method": {"id": "split", "name": "Split kernels"}, "probability": .6}]
    response = {"answers": {item["method"]["id"]: {"type": "noul", "noul": item["probability"]} for item in candidates}}
    request = {"state": {"hardware": device}, "questions": {item["method"]["id"]: {"type": "noul"} for item in candidates}}
    response_hash = hashlib.sha256(json.dumps(response, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    metadata = {"schema_version": 1, "protocol": "fusion-suitability/v2-en", "jev_config": {"model": "jev-test"}, "response_sha256": response_hash,
                "sources": {key: {"path": str(work / filename), "sha256": hashlib.sha256((work / filename).read_bytes()).hexdigest()}
                            for key, filename in (("stage1_analysis", "ANALYSIS.md"),
                                                  ("stage1_requirements", "fusion_requirements.en.json"),
                                                  ("hardware", "device_info.json"))},
                "english_inputs_path": str(work / "fusion/translation/english_inputs.json")}
    metadata["fingerprint_sha256"] = hashlib.sha256(json.dumps({
        "protocol": metadata["protocol"], "request": request, "jev_config": metadata["jev_config"],
        "source_sha256": {key: value["sha256"] for key, value in metadata["sources"].items()},
    }, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write_json(work / "fusion/jev_request.json", request)
    write_json(work / "fusion/jev_response.json", response)
    write_json(work / "fusion/translation/english_inputs.json", {"hardware": device})
    write_json(work / "fusion/ranking.json", {**metadata, "candidates": candidates})
    library = {**metadata, "top_n": 1, "candidates": candidates[:1]}
    write_json(work / "fusion/fusion_library.json", library)
    develop = work / "develop" / ("iter0" if stage == 2 else "iter2")
    develop.mkdir(parents=True, exist_ok=True)
    rationale = "design_rationale.md" if stage == 2 else "融合方案选择决策依据.md"
    (develop / rationale).write_text(f"Implement {work / 'impl'}; source {develop / 'fusion_library.json'}", encoding="utf-8")
    (develop / "self_test_report.md").write_text("Two genuine synthetic test records.", encoding="utf-8")
    (develop / "self_test.log").write_text("Synthetic tests pass", encoding="utf-8")
    tests = {"schema_version": 1,
             "provided_cases": {"executed": True, "passed": True, "total": 2, "passed_cases": 2, "evidence_path": str(develop / "self_test.log")},
             "continuous_calls": {"executed": True, "passed": True, "same_shape": True,
                                  "reference_checked_each_call": True, "call_count": 2,
                                  "evidence_path": str(develop / "self_test.log"),
                                  "changes": {"inputs": {"applicable": True, "changed": True, "passed": True},
                                              "weights": {"applicable": False, "reason": "identity"},
                                              "bias": {"applicable": False, "reason": "identity"}}}}
    write_json(develop / "self_test_result.json", tests)
    selection = {"method_ids": ["vertical"], "implementation_plan": "Identity pointwise kernel", "reason": "No reduction",
                 "target_cases": ["all"], "actual_changes": ["vector load/store"], "expected_benefits": ["fewer launches"]}
    write_json(develop / "fusion_library.json", {**library, "selection": selection})
    evidence = {"fusion_library": str(develop / "fusion_library.json"), "decision_rationale": str(develop / rationale),
                "self_test_report": str(develop / "self_test_report.md"), "self_test_result": str(develop / "self_test_result.json"),
                "selftest_log_provided": str(develop / "self_test.log"), "selftest_log_continuous": str(develop / "self_test.log")}
    write_json(work / "selection/current_implementation.json", {
        "schema_version": 1, "stage": stage, "develop_dir": str(develop), "eligible": True,
        "impl_sha256": implementation_hash(work / "impl"), "evidence_paths": evidence,
        "initial_library_sha256": _initial_library_hash(work / "fusion/fusion_library.json"),
        "implementation_plan": selection["implementation_plan"],
        "fusion_scheme": {**selection, "methods": candidates[:1]},
        "document_sha256": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in evidence.items()}})
    return work


class InitImplTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = make_source(self.root)
        self.work = self.root / "destination"
        self.device = json.loads((self.source / "device_info.json").read_text())

    def prepare(self, **kwargs):
        return prepare_init_impl(self.source / "impl", self.work, self.source / "task", **kwargs)

    def test_import_preserves_code_probabilities_and_rebases_only_copied_paths(self):
        manifest = self.prepare(optimize_hint="remember direction")
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["optimize_hint"], "remember direction")
        self.assertEqual((self.work / "impl/cann_bench/__init__.py").read_bytes(),
                         (self.source / "impl/cann_bench/__init__.py").read_bytes())
        library = json.loads((self.work / "fusion/fusion_library.json").read_text(encoding="utf-8"))
        self.assertEqual(library["candidates"][0]["probability"], .8)
        self.assertEqual(library["sources"]["hardware"]["path"], str(self.work / "device_info.json"))
        self.assertEqual((self.work / "fusion/jev_request.json").read_bytes(), (self.source / "fusion/jev_request.json").read_bytes())
        analysis = (self.work / "ANALYSIS.md").read_text(encoding="utf-8")
        self.assertIn(str(self.work / "impl"), analysis)
        self.assertIn(str(self.source / "history.json"), analysis)
        result = json.loads((self.work / "develop/iter0/self_test_result.json").read_text())
        self.assertEqual(result["provided_cases"]["evidence_path"], str(self.work / "develop/iter0/self_test.log"))
        self.assertEqual(load_init_impl_manifest(self.work), manifest)
        validate_init_impl_inputs(self.work, {**self.device, "detect_method": "new detector"}, self.source / "task")

    def test_no_history_performance_human_or_other_iteration_is_copied(self):
        for relative in ("history.json", "selection/best.json", "eval/iter1/perf_result.json", "human_review/state.json",
                         "develop/iter2/secret.md", "develop/iter0/human_feedback.json", "develop/iter0/question.md"):
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        self.prepare()
        for relative in ("history.json", "selection", "eval", "human_review", "develop/iter2",
                         "develop/iter0/human_feedback.json", "develop/iter0/question.md"):
            self.assertFalse((self.work / relative).exists(), relative)

    def test_stage3_binding_selects_that_development_and_preserves_name(self):
        self.source = make_source(self.root / "stage3", stage=3)
        manifest = self.prepare()
        self.assertEqual(manifest["source_development_binding"]["stage"], 3)
        self.assertEqual(manifest["source_develop_dir"], str(self.source / "develop/iter2"))
        self.assertTrue((self.work / "develop/iter0/融合方案选择决策依据.md").is_file())
        self.assertFalse((self.work / "develop/iter0/design_rationale.md").exists())
        validate_init_impl_inputs(self.work, self.device, self.source / "task")

    def test_modified_source_code_cannot_reuse_binding(self):
        (self.source / "impl/cann_bench/__init__.py").write_text("import triton\n# modified", encoding="utf-8")
        manifest = self.prepare()
        self.assertIsNone(manifest["source_development_binding"])

    def test_missing_binding_does_not_require_best(self):
        (self.source / "selection/current_implementation.json").unlink()
        manifest = self.prepare()
        self.assertIsNone(manifest["source_development_binding"])

    def test_missing_stage_input_stops_and_leaves_failed_marker(self):
        (self.source / "ANALYSIS.md").unlink()
        with self.assertRaisesRegex(ValueError, "必要材料"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "上次导入未完成"):
            load_init_impl_manifest(self.work)

    def test_mismatching_task_stops(self):
        other = self.root / "other-task"
        other.mkdir()
        (other / "cases.yaml").write_text("cases: [3]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "任务.*不一致"):
            prepare_init_impl(self.source / "impl", self.work, other)

    def test_invalid_probability_is_rejected_before_copy(self):
        path = self.source / "fusion/fusion_library.json"
        library = json.loads(path.read_text())
        library["candidates"][0]["probability"] = .99
        write_json(path, library)
        with self.assertRaisesRegex(ValueError, "Top N"):
            self.prepare()
        self.assertFalse((self.work / "impl").exists())

    def test_hardware_mismatch_rejected_without_jev(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "chip_model"):
            validate_init_impl_inputs(self.work, {**self.device, "chip_model": "different"}, self.source / "task")

    def test_cann_toolchain_change_cannot_reuse_imported_selftests(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "toolchain"):
            validate_init_impl_inputs(self.work, {**self.device, "toolchain": {"version": "different"}}, self.source / "task")

    def test_modified_imported_metadata_is_rejected_on_resume(self):
        self.prepare()
        (self.work / "ANALYSIS.md").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "被修改"):
            validate_init_impl_inputs(self.work, self.device, self.source / "task")

    def test_later_stage3_changes_to_impl_do_not_block_resume(self):
        self.prepare()
        (self.work / "impl/cann_bench/__init__.py").write_text("import triton\n# valid later iteration", encoding="utf-8")
        validate_init_impl_inputs(self.work, self.device, self.source / "task")

    def test_existing_target_files_are_never_overwritten(self):
        (self.work / "impl").mkdir(parents=True)
        (self.work / "impl/user.py").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "必须为空"):
            self.prepare()
        self.assertEqual((self.work / "impl/user.py").read_text(), "keep")

    def test_interrupted_copy_cannot_fall_through_to_regular_workflow(self):
        original = Path.read_bytes
        def fail(path):
            if path == self.source / "impl/setup.py" and (self.work / MANIFEST).exists():
                raise OSError("synthetic copy failure")
            return original(path)
        with patch.object(Path, "read_bytes", fail):
            with self.assertRaises(OSError):
                self.prepare()
        with self.assertRaisesRegex(ValueError, "上次导入未完成"):
            load_init_impl_manifest(self.work)

    def test_copy_log_lists_sources_and_skipped_stages(self):
        with self.assertLogs("triton-ascend-workflow", level="INFO") as captured:
            self.prepare()
        output = "\n".join(captured.output)
        self.assertIn("跳过 Stage1、Stage1.5、Stage2", output)
        self.assertIn(str(self.source / "develop/iter0"), output)
        self.assertIn(str(self.work / "fusion"), output)

    def test_original_request_tampering_is_rejected(self):
        path = self.source / "fusion/jev_request.json"
        request = json.loads(path.read_text())
        request["state"]["hardware"]["chip_model"] = "other chip"
        write_json(path, request)
        with self.assertRaisesRegex(ValueError, "请求与来源摘要"):
            self.prepare()

    def test_source_link_is_rejected_before_contents_are_copied(self):
        from lib.init_impl import _linked
        blocked = self.source / "fusion"
        with patch("lib.init_impl._linked", side_effect=lambda path: path == blocked or _linked(path)):
            with self.assertRaisesRegex(ValueError, "符号链接"):
                self.prepare()
        self.assertFalse((self.work / "impl").exists())

    def test_original_fusion_source_change_rejected(self):
        (self.source / "ANALYSIS.md").write_text("new task definition", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "原始输入不一致"):
            self.prepare()

    def test_specified_snapshot_code_is_never_replaced_by_best(self):
        snapshot = self.source / "selection/records/custom/impl"
        snapshot.mkdir(parents=True)
        (snapshot / "kernel.py").write_text("import triton\n# explicitly selected", encoding="utf-8")
        write_json(self.source / "selection/best.json", {"implementation_dir": "must-not-read-this"})
        manifest = prepare_init_impl(snapshot, self.work, self.source / "task")
        self.assertIsNone(manifest["source_development_binding"])
        self.assertEqual((self.work / "impl/kernel.py").read_bytes(), (snapshot / "kernel.py").read_bytes())
        self.assertFalse((self.work / "impl/cann_bench").exists())

    def test_three_successive_imports_keep_original_scoring_and_current_copy_hashes(self):
        from lib.fusion_evidence import restore_imported_evidence
        original_request = (self.source / "fusion/jev_request.json").read_bytes()
        score_hashes = None
        for count in range(1, 4):
            manifest = self.prepare()
            restored = restore_imported_evidence(self.work, manifest)
            self.assertTrue(restored["eligible"], restored["reason"])
            shutil.copytree(self.source / "task", self.work / "task")
            validate_init_impl_inputs(self.work, self.device, self.work / "task")
            self.assertEqual((self.work / "fusion/jev_request.json").read_bytes(), original_request)
            if score_hashes is None:
                score_hashes = manifest["fusion_input_hashes"]
            self.assertEqual(manifest["fusion_input_hashes"], score_hashes)
            analysis_entry = next(entry for entry in manifest["copied_files"] if entry["destination"] == "ANALYSIS.md")
            if count > 1:
                self.assertNotEqual(analysis_entry["source_sha256"], score_hashes["ANALYSIS.md"])
            self.source, self.work = self.work, self.root / f"destination{count + 1}"

    def test_reimport_checks_prior_import_integrity(self):
        manifest = self.prepare()
        shutil.copytree(self.source / "task", self.work / "task")
        (self.work / "ANALYSIS.md").write_text("corrupted imported source", encoding="utf-8")
        self.source, self.work = self.work, self.root / "destination2"
        with self.assertRaisesRegex(ValueError, "被修改"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "上次导入未完成"):
            load_init_impl_manifest(self.work)

    def test_impl_copy_uses_same_exclusions_as_existing_evidence_hash(self):
        from lib.fusion_evidence import restore_imported_evidence
        cache = self.source / "impl/.pytest_cache/README.md"
        cache.parent.mkdir()
        cache.write_text("pre-existing file covered by the original binding", encoding="utf-8")
        pointer = self.source / "selection/current_implementation.json"
        binding = json.loads(pointer.read_text())
        binding["impl_sha256"] = implementation_hash(self.source / "impl")
        write_json(pointer, binding)
        manifest = self.prepare()
        self.assertEqual(implementation_hash(self.work / "impl"), binding["impl_sha256"])
        self.assertTrue(restore_imported_evidence(self.work, manifest)["eligible"])


if __name__ == "__main__":
    unittest.main()
